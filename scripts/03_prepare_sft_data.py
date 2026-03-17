from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import DatasetDict, load_from_disk


PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "datasets" / "daring_anteater"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "sft" / "daring_anteater_sft"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sft" / "daring_anteater_sft_samples.jsonl"

VAL_SIZE = 0.02
SEED = 42


def normalize_role(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm in {"user", "human"}:
        return "user"
    if role_norm in {"assistant", "gpt", "bot"}:
        return "assistant"
    if role_norm == "system":
        return "system"
    return role_norm


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Daring-Anteater row into:
      - messages: list[{role, content}]
      - source_dataset
      - raw_mask
      - num_messages
      - num_assistant_turns
      - valid
    """
    messages: List[Dict[str, str]] = []

    system_text = clean_text(example.get("system", ""))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    conversations = example.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = normalize_role(clean_text(turn.get("from", "")))
        content = clean_text(turn.get("value", ""))
        if not role or not content:
            continue
        if role not in {"user", "assistant", "system"}:
            continue
        messages.append({"role": role, "content": content})

    num_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    num_user_turns = sum(1 for m in messages if m["role"] == "user")

    valid = (
        len(messages) >= 2
        and num_user_turns >= 1
        and num_assistant_turns >= 1
    )

    return {
        "messages": messages,
        "source_dataset": clean_text(example.get("dataset", "")),
        "raw_mask": example.get("mask", None),
        "num_messages": len(messages),
        "num_assistant_turns": num_assistant_turns,
        "valid": valid,
    }


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = load_from_disk(str(RAW_PATH))
    if "train" not in raw:
        raise ValueError("Expected a 'train' split in Daring-Anteater.")

    print("Loaded raw dataset:")
    print(raw)

    processed_train = raw["train"].map(
        convert_example,
        remove_columns=raw["train"].column_names,
        desc="Converting Daring-Anteater to messages format",
    )

    processed_train = processed_train.filter(
        lambda x: x["valid"],
        desc="Filtering invalid SFT rows",
    )

    processed_train = processed_train.remove_columns(["valid"])

    split_ds = processed_train.train_test_split(
        test_size=VAL_SIZE,
        seed=SEED,
        shuffle=True,
    )

    final_ds = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"],
    })

    final_ds.save_to_disk(str(OUT_PATH))

    print("\nSaved processed SFT dataset to:")
    print(OUT_PATH)
    print("\nFinal splits:")
    print(final_ds)
    print(f"train: {len(final_ds['train'])}")
    print(f"validation: {len(final_ds['validation'])}")

    # Write a few readable samples for manual inspection
    sample_count = min(5, len(final_ds["train"]))
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for i in range(sample_count):
            row = final_ds["train"][i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {sample_count} sample rows to: {SAMPLE_PATH}")

    # Print one sample nicely
    if len(final_ds["train"]) > 0:
        sample = final_ds["train"][0]
        print("\nExample processed SFT row:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
