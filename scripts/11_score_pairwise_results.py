from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
PAIRWISE_DIR = PROJECT_ROOT / "outputs" / "pairwise_compare"

BLIND_CSV = Path(os.environ.get("BLIND_EVAL_CSV", str(PAIRWISE_DIR / "blind_eval_filled.csv")))
ANSWER_KEY_CSV = Path(os.environ.get("ANSWER_KEY_CSV", str(PAIRWISE_DIR / "answer_key.csv")))
OUT_JSON = Path(os.environ.get("PAIRWISE_SCORE_JSON", str(PAIRWISE_DIR / "pairwise_results_summary.json")))


VALID_WINNERS = {"a", "b", "tie", "invalid"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def main() -> None:
    console.print(Panel.fit("Scoring pairwise blind evaluation", style="bold cyan"))

    if not BLIND_CSV.exists():
        raise FileNotFoundError(f"Missing blind eval CSV: {BLIND_CSV}")
    if not ANSWER_KEY_CSV.exists():
        raise FileNotFoundError(f"Missing answer key CSV: {ANSWER_KEY_CSV}")

    blind_rows = read_csv(BLIND_CSV)
    key_rows = read_csv(ANSWER_KEY_CSV)

    key_by_pair_id = {row["pair_id"]: row for row in key_rows}

    per_comparison = defaultdict(lambda: {
        "total_pairs": 0,
        "a_wins": 0,
        "b_wins": 0,
        "ties": 0,
        "invalid": 0,
        "model_stats": defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "invalid": 0}),
    })

    overall_model_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "invalid": 0})
    decoded_rows = []

    for blind in blind_rows:
        pair_id = blind["pair_id"]
        winner_raw = blind.get("winner", "").strip().lower()

        if winner_raw not in VALID_WINNERS:
            console.print(f"[yellow]Skipping pair_id={pair_id} with invalid winner='{winner_raw}'[/yellow]")
            continue

        if pair_id not in key_by_pair_id:
            console.print(f"[yellow]Skipping pair_id={pair_id}; not found in answer key[/yellow]")
            continue

        key = key_by_pair_id[pair_id]
        comparison = blind["comparison"]
        model_a = key["model_a"]
        model_b = key["model_b"]

        stats = per_comparison[comparison]
        stats["total_pairs"] += 1

        decoded = {
            "pair_id": pair_id,
            "comparison": comparison,
            "prompt_id": blind["prompt_id"],
            "winner_label": winner_raw,
            "model_a": model_a,
            "model_b": model_b,
            "winner_model": "",
            "loser_model": "",
            "notes": blind.get("notes", ""),
        }

        if winner_raw == "a":
            stats["a_wins"] += 1
            stats["model_stats"][model_a]["wins"] += 1
            stats["model_stats"][model_b]["losses"] += 1
            overall_model_stats[model_a]["wins"] += 1
            overall_model_stats[model_b]["losses"] += 1
            decoded["winner_model"] = model_a
            decoded["loser_model"] = model_b

        elif winner_raw == "b":
            stats["b_wins"] += 1
            stats["model_stats"][model_b]["wins"] += 1
            stats["model_stats"][model_a]["losses"] += 1
            overall_model_stats[model_b]["wins"] += 1
            overall_model_stats[model_a]["losses"] += 1
            decoded["winner_model"] = model_b
            decoded["loser_model"] = model_a

        elif winner_raw == "tie":
            stats["ties"] += 1
            stats["model_stats"][model_a]["ties"] += 1
            stats["model_stats"][model_b]["ties"] += 1
            overall_model_stats[model_a]["ties"] += 1
            overall_model_stats[model_b]["ties"] += 1

        elif winner_raw == "invalid":
            stats["invalid"] += 1
            stats["model_stats"][model_a]["invalid"] += 1
            stats["model_stats"][model_b]["invalid"] += 1
            overall_model_stats[model_a]["invalid"] += 1
            overall_model_stats[model_b]["invalid"] += 1

        decoded_rows.append(decoded)

    summary = {
        "blind_eval_csv": str(BLIND_CSV),
        "answer_key_csv": str(ANSWER_KEY_CSV),
        "comparisons": {},
        "overall_model_stats": overall_model_stats,
        "decoded_rows": decoded_rows,
    }

    # Build JSON-friendly dicts and compute rates
    for comparison, stats in per_comparison.items():
        model_stats = dict(stats["model_stats"])

        valid_non_tie = stats["a_wins"] + stats["b_wins"]

        comp_summary = {
            "total_pairs": stats["total_pairs"],
            "a_wins": stats["a_wins"],
            "b_wins": stats["b_wins"],
            "ties": stats["ties"],
            "invalid": stats["invalid"],
            "valid_non_tie_pairs": valid_non_tie,
            "model_stats": model_stats,
        }

        # For 2-model comparisons, compute explicit win rates
        models = list(model_stats.keys())
        if len(models) == 2:
            m1, m2 = models[0], models[1]
            m1_wins = model_stats[m1]["wins"]
            m2_wins = model_stats[m2]["wins"]

            comp_summary["win_rates_valid_non_tie"] = {
                m1: safe_float(m1_wins, valid_non_tie),
                m2: safe_float(m2_wins, valid_non_tie),
            }

        summary["comparisons"][comparison] = comp_summary

    # Convert defaultdict to regular dict
    summary["overall_model_stats"] = {k: dict(v) for k, v in overall_model_stats.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[bold green]Saved summary to:[/bold green] {OUT_JSON}")

    # Pretty print comparison summaries
    for comparison, comp in summary["comparisons"].items():
        table = Table(title=f"Comparison: {comparison}")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("total_pairs", str(comp["total_pairs"]))
        table.add_row("a_wins", str(comp["a_wins"]))
        table.add_row("b_wins", str(comp["b_wins"]))
        table.add_row("ties", str(comp["ties"]))
        table.add_row("invalid", str(comp["invalid"]))
        table.add_row("valid_non_tie_pairs", str(comp["valid_non_tie_pairs"]))

        if "win_rates_valid_non_tie" in comp:
            for model_name, rate in comp["win_rates_valid_non_tie"].items():
                table.add_row(f"{model_name} win rate", f"{rate:.2%}")

        console.print(table)

    overall_table = Table(title="Overall model stats")
    overall_table.add_column("Model")
    overall_table.add_column("Wins")
    overall_table.add_column("Losses")
    overall_table.add_column("Ties")
    overall_table.add_column("Invalid")

    for model_name, stats in summary["overall_model_stats"].items():
        overall_table.add_row(
            model_name,
            str(stats["wins"]),
            str(stats["losses"]),
            str(stats["ties"]),
            str(stats["invalid"]),
        )

    console.print(overall_table)


if __name__ == "__main__":
    main()