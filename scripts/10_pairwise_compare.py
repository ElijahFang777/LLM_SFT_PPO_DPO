from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
GEN_DIR = PROJECT_ROOT / "outputs" / "eval_generations"
OUT_DIR = PROJECT_ROOT / "outputs" / "pairwise_compare"

SFT_FILE = GEN_DIR / "sft_outputs.jsonl"
PPO_FILE = GEN_DIR / "ppo_hh_outputs.jsonl"
DPO_FILE = GEN_DIR / "dpo_hh_outputs.jsonl"

SEED = int(os.environ.get("PAIRWISE_SEED", "42"))
PAIRWISE_LIMIT_PER_COMPARISON = int(os.environ.get("PAIRWISE_LIMIT_PER_COMPARISON", "0"))
SKIP_IDENTICAL_RESPONSES = os.environ.get("SKIP_IDENTICAL_RESPONSES", "false").lower() == "true"


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def index_by_prompt_id(rows: List[Dict]) -> Dict[int, Dict]:
    out = {}
    for row in rows:
        out[int(row["prompt_id"])] = row
    return out


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def build_pair(
    prompt_id: int,
    prompt: str,
    model_x: str,
    response_x: str,
    model_y: str,
    response_y: str,
    rng: random.Random,
    comparison_name: str,
    pair_id: int,
) -> Tuple[Dict, Dict]:
    same_response = normalize_text(response_x) == normalize_text(response_y)

    if rng.random() < 0.5:
        public_row = {
            "pair_id": pair_id,
            "comparison": comparison_name,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response_a": response_x,
            "response_b": response_y,
            "winner": "",   # fill with A / B / tie / invalid
            "notes": "",
        }
        key_row = {
            "pair_id": pair_id,
            "comparison": comparison_name,
            "prompt_id": prompt_id,
            "model_a": model_x,
            "model_b": model_y,
            "identical_response": same_response,
        }
    else:
        public_row = {
            "pair_id": pair_id,
            "comparison": comparison_name,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response_a": response_y,
            "response_b": response_x,
            "winner": "",
            "notes": "",
        }
        key_row = {
            "pair_id": pair_id,
            "comparison": comparison_name,
            "prompt_id": prompt_id,
            "model_a": model_y,
            "model_b": model_x,
            "identical_response": same_response,
        }

    return public_row, key_row


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit("Building blind pairwise comparison files", style="bold cyan"))

    for path in [SFT_FILE, PPO_FILE, DPO_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing generation file: {path}")

    sft_rows = read_jsonl(SFT_FILE)
    ppo_rows = read_jsonl(PPO_FILE)
    dpo_rows = read_jsonl(DPO_FILE)

    sft = index_by_prompt_id(sft_rows)
    ppo = index_by_prompt_id(ppo_rows)
    dpo = index_by_prompt_id(dpo_rows)

    common_prompt_ids = sorted(set(sft) & set(ppo) & set(dpo))
    if not common_prompt_ids:
        raise ValueError("No common prompt_id values found across SFT/PPO/DPO files.")

    console.print(f"Common prompt_ids: {len(common_prompt_ids)}")

    pair_specs = [
        ("sft_vs_ppo", "sft", "ppo_hh", sft, ppo),
        ("sft_vs_dpo", "sft", "dpo_hh", sft, dpo),
        ("ppo_vs_dpo", "ppo_hh", "dpo_hh", ppo, dpo),
    ]

    public_rows: List[Dict] = []
    key_rows: List[Dict] = []
    stats = defaultdict(int)

    pair_id = 1

    for comparison_name, model_x, model_y, idx_x, idx_y in pair_specs:
        prompt_ids = common_prompt_ids[:]
        rng.shuffle(prompt_ids)

        kept = 0
        for prompt_id in prompt_ids:
            row_x = idx_x[prompt_id]
            row_y = idx_y[prompt_id]

            prompt = row_x["prompt"]
            response_x = row_x["response"]
            response_y = row_y["response"]

            same_response = normalize_text(response_x) == normalize_text(response_y)
            if SKIP_IDENTICAL_RESPONSES and same_response:
                stats[f"{comparison_name}_skipped_identical"] += 1
                continue

            public_row, key_row = build_pair(
                prompt_id=prompt_id,
                prompt=prompt,
                model_x=model_x,
                response_x=response_x,
                model_y=model_y,
                response_y=response_y,
                rng=rng,
                comparison_name=comparison_name,
                pair_id=pair_id,
            )

            public_rows.append(public_row)
            key_rows.append(key_row)

            stats[f"{comparison_name}_kept"] += 1
            if same_response:
                stats[f"{comparison_name}_identical"] += 1

            pair_id += 1
            kept += 1

            if PAIRWISE_LIMIT_PER_COMPARISON > 0 and kept >= PAIRWISE_LIMIT_PER_COMPARISON:
                break

    blind_jsonl = OUT_DIR / "blind_eval.jsonl"
    blind_csv = OUT_DIR / "blind_eval.csv"
    key_jsonl = OUT_DIR / "answer_key.jsonl"
    key_csv = OUT_DIR / "answer_key.csv"
    manifest_json = OUT_DIR / "manifest.json"

    write_jsonl(blind_jsonl, public_rows)
    write_jsonl(key_jsonl, key_rows)

    write_csv(
        blind_csv,
        public_rows,
        fieldnames=[
            "pair_id",
            "comparison",
            "prompt_id",
            "prompt",
            "response_a",
            "response_b",
            "winner",
            "notes",
        ],
    )

    write_csv(
        key_csv,
        key_rows,
        fieldnames=[
            "pair_id",
            "comparison",
            "prompt_id",
            "model_a",
            "model_b",
            "identical_response",
        ],
    )

    manifest = {
        "seed": SEED,
        "common_prompt_ids": len(common_prompt_ids),
        "pairwise_limit_per_comparison": PAIRWISE_LIMIT_PER_COMPARISON,
        "skip_identical_responses": SKIP_IDENTICAL_RESPONSES,
        "public_files": {
            "jsonl": str(blind_jsonl),
            "csv": str(blind_csv),
        },
        "private_key_files": {
            "jsonl": str(key_jsonl),
            "csv": str(key_csv),
        },
        "stats": dict(stats),
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print("[bold green]Done.[/bold green]")
    pprint(manifest)


if __name__ == "__main__":
    main()