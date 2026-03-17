from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "pairwise_compare" / "pairwise_results_summary.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "pairwise_compare" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0

    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = (
        z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    )
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)
    return low, high


def main() -> None:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    comparisons = summary["comparisons"]

    rows = []
    for comp_name, comp in comparisons.items():
        valid_n = comp["valid_non_tie_pairs"]
        model_stats = comp["model_stats"]

        for model_name, stats in model_stats.items():
            wins = stats["wins"]
            losses = stats["losses"]
            ties = stats["ties"]
            invalid = stats["invalid"]

            rate = wins / valid_n if valid_n > 0 else 0.0
            ci_low, ci_high = wilson_interval(wins, valid_n)

            rows.append(
                {
                    "comparison": comp_name,
                    "model": model_name,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "invalid": invalid,
                    "valid_non_tie_pairs": valid_n,
                    "win_rate": rate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    df = pd.DataFrame(rows)

    comparison_order = list(comparisons.keys())
    model_order = ["sft", "ppo_hh", "dpo_hh"]

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", n_colors=len(model_order))
    color_map = dict(zip(model_order, palette))

    fig, ax = plt.subplots(figsize=(12, 7))

    group_centers = list(range(len(comparison_order)))
    bar_width = 0.24

    # fixed offsets for up to 3 possible models
    offsets = {
        "sft": -bar_width,
        "ppo_hh": 0.0,
        "dpo_hh": bar_width,
    }

    legend_handles = {}
    plotted_labels = set()

    for _, row in df.iterrows():
        comp_idx = comparison_order.index(row["comparison"])
        model = row["model"]

        if model not in offsets:
            continue

        x = group_centers[comp_idx] + offsets[model]
        y = float(row["win_rate"])

        color = color_map.get(model, "gray")
        label = model if model not in plotted_labels else None

        bar = ax.bar(
            x,
            y,
            width=bar_width * 0.9,
            color=color,
            label=label,
            zorder=3,
        )

        if label is not None:
            plotted_labels.add(model)
            legend_handles[model] = bar

        # Clamp error bar widths to be non-negative
        yerr_low = max(0.0, y - float(row["ci_low"]))
        yerr_high = max(0.0, float(row["ci_high"]) - y)

        ax.errorbar(
            x=x,
            y=y,
            yerr=[[yerr_low], [yerr_high]],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )

        ax.text(
            x,
            min(y + 0.04, 1.02),
            f"{int(row['wins'])}/{int(row['valid_non_tie_pairs'])}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Pairwise Preference Win Rates with 95% Wilson Confidence Intervals")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Win rate (excluding ties and invalids)")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(comparison_order)
    ax.set_ylim(0, 1.05)

    # only show models that actually appeared
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Model", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_win_rates_wilson_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "pairwise_win_rates_wilson_ci.pdf", bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUT_DIR / "pairwise_win_rates_table.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "pairwise_win_rates_wilson_ci.png")
    print(OUT_DIR / "pairwise_win_rates_wilson_ci.pdf")
    print(OUT_DIR / "pairwise_win_rates_table.csv")


if __name__ == "__main__":
    main()