from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_from_disk


PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "datasets" / "hh_rlhf"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_explicit"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_samples.jsonl"

VAL_SIZE = 0.02
SEED = 42

SPEAKER_PATTERN = re.compile(r"\n\n(Human|Assistant):")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def empty_row(reason: str) -> Dict[str, Any]:
    return {
        "prompt": "",
        "chosen": "",
        "rejected": "",
        "prompt_messages": [],
        "chosen_messages": [],
        "rejected_messages": [],
        "num_prompt_turns": 0,
        "valid": False,
        "reason": reason,
    }


def parse_hh_dialogue(text: str) -> List[Tuple[str, str]]:
    text = clean_text(text)
    if not text:
        return []

    if not text.startswith("\n\n"):
        text = "\n\n" + text

    matches = list(SPEAKER_PATTERN.finditer(text))
    turns: List[Tuple[str, str]] = []

    for i, match in enumerate(matches):
        speaker = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append((speaker, content))

    return turns


def role_map(speaker: str) -> str:
    s = speaker.strip().lower()
    if s == "human":
        return "user"
    if s == "assistant":
        return "assistant"
    return s


def render_prompt_text(turns: List[Tuple[str, str]]) -> str:
    chunks = []
    for speaker, content in turns:
        if speaker == "Human":
            chunks.append(f"Human: {content}")
        elif speaker == "Assistant":
            chunks.append(f"Assistant: {content}")
        else:
            chunks.append(f"{speaker}: {content}")
    return "\n\n".join(chunks).strip()


def turns_to_messages(turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    return [{"role": role_map(s), "content": c} for s, c in turns]


def convert_pref_example(example: Dict[str, Any]) -> Dict[str, Any]:
    chosen_text = clean_text(example.get("chosen", ""))
    rejected_text = clean_text(example.get("rejected", ""))

    if not chosen_text or not rejected_text:
        return empty_row("empty_pair")

    chosen_turns = parse_hh_dialogue(chosen_text)
    rejected_turns = parse_hh_dialogue(rejected_text)

    if len(chosen_turns) < 2 or len(rejected_turns) < 2:
        return empty_row("too_few_turns")

    if chosen_turns[-1][0] != "Assistant" or rejected_turns[-1][0] != "Assistant":
        return empty_row("last_turn_not_assistant")

    chosen_prompt_turns = chosen_turns[:-1]
    rejected_prompt_turns = rejected_turns[:-1]

    if chosen_prompt_turns != rejected_prompt_turns:
        return empty_row("prompt_mismatch")

    chosen_answer = chosen_turns[-1][1].strip()
    rejected_answer = rejected_turns[-1][1].strip()

    if not chosen_answer or not rejected_answer:
        return empty_row("empty_completion")

    if chosen_answer == rejected_answer:
        return empty_row("identical_completion")

    prompt_text = render_prompt_text(chosen_prompt_turns)
    prompt_messages = turns_to_messages(chosen_prompt_turns)

    return {
        "prompt": prompt_text,
        "chosen": chosen_answer,
        "rejected": rejected_answer,
        "prompt_messages": prompt_messages,
        "chosen_messages": [{"role": "assistant", "content": chosen_answer}],
        "rejected_messages": [{"role": "assistant", "content": rejected_answer}],
        "num_prompt_turns": len(chosen_prompt_turns),
        "valid": True,
        "reason": "ok",
    }


def process_split(ds, split_name: str) -> Dataset:
    total = len(ds)
    valid_rows: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}

    print(f"\nProcessing split: {split_name} (total={total})")

    for i, example in enumerate(ds):
        row = convert_pref_example(example)
        reason = row["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if row["valid"]:
            valid_rows.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "prompt_messages": row["prompt_messages"],
                "chosen_messages": row["chosen_messages"],
                "rejected_messages": row["rejected_messages"],
                "num_prompt_turns": row["num_prompt_turns"],
            })

        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"  processed {i + 1}/{total}")

    print(f"\n{split_name} reason counts:")
    for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    print(f"\n{split_name}: kept {len(valid_rows)}/{total} rows")

    if not valid_rows:
        raise ValueError(f"No valid rows produced for split '{split_name}'.")

    return Dataset.from_list(valid_rows)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = load_from_disk(str(RAW_PATH))
    print("Loaded raw HH-RLHF:")
    print(raw)

    if "train" not in raw or "test" not in raw:
        raise ValueError("Expected HH-RLHF to contain 'train' and 'test' splits.")

    processed_train_full = process_split(raw["train"], "train")
    processed_test = process_split(raw["test"], "test")

    split_train = processed_train_full.train_test_split(
        test_size=VAL_SIZE,
        seed=SEED,
        shuffle=True,
    )

    final_ds = DatasetDict({
        "train": split_train["train"],
        "validation": split_train["test"],
        "test": processed_test,
    })

    final_ds.save_to_disk(str(OUT_PATH))

    print("\nSaved processed HH preference dataset to:")
    print(OUT_PATH)
    print("\nFinal splits:")
    print(final_ds)
    print(f"train: {len(final_ds['train'])}")
    print(f"validation: {len(final_ds['validation'])}")
    print(f"test: {len(final_ds['test'])}")

    sample_count = min(5, len(final_ds["train"]))
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for i in range(sample_count):
            row = final_ds["train"][i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {sample_count} sample rows to: {SAMPLE_PATH}")

    sample = final_ds["train"][0]
    print("\nExample processed preference row:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
