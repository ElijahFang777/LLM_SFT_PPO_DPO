from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from peft import PeftConfig, PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
BASE_MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"

SFT_PATH = PROJECT_ROOT / "checkpoints" / "sft" / "qwen25_15b_lora_sft" / "final"
PPO_PATH = PROJECT_ROOT / "checkpoints" / "ppo_hh" / "qwen25_15b_ppo_hh_full" / "final"
DPO_PATH = PROJECT_ROOT / "checkpoints" / "dpo_hh" / "qwen25_15b_dpo_hh_full" / "final"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_explicit"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_generations"

SEED = 42
MAX_INPUT_LENGTH = 384
MAX_NEW_TOKENS = 128

# Optional overrides from shell
# export EVAL_PROMPT_LIMIT=100
# export EVAL_SPLIT=test
EVAL_PROMPT_LIMIT = int(os.environ.get("EVAL_PROMPT_LIMIT", "100"))
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")


def is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(model_path: Path):
    bnb_config = build_bnb_config()

    if is_adapter_checkpoint(model_path):
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(model_path), is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    model.config.use_cache = True
    return model


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_eval_prompts(limit: int, split: str) -> List[Dict[str, Any]]:
    ds = load_from_disk(str(DATA_PATH))
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    split_ds = ds[split]

    # Keep prompt text + conversational prompt for proper chat templating
    split_ds = split_ds.select_columns(["prompt", "prompt_messages"])

    # Deduplicate by prompt text so repeated prompts do not dominate evaluation
    seen = set()
    rows: List[Dict[str, Any]] = []
    for row in split_ds:
        prompt_text = row["prompt"]
        if prompt_text in seen:
            continue
        seen.add(prompt_text)
        rows.append({
            "prompt_id": len(rows),
            "prompt": prompt_text,
            "prompt_messages": row["prompt_messages"],
            "source_split": split,
        })
        if len(rows) >= limit:
            break

    return rows


def messages_to_model_input(tokenizer, prompt_messages: List[Dict[str, str]]) -> str:
    # Best path for chat models
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback if chat template is unavailable
    parts = []
    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n\n".join(parts)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt_messages: List[Dict[str, str]]) -> Dict[str, Any]:
    model_input_text = messages_to_model_input(tokenizer, prompt_messages)

    device = get_model_device(model)
    enc = tokenizer(
        model_input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    output_ids = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # deterministic comparison
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][enc["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "rendered_prompt": model_input_text,
        "response": completion,
        "input_token_count": int(enc["input_ids"].shape[1]),
        "output_token_count": int(new_tokens.shape[0]),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_model_generation(model_name: str, model_path: Path, prompts: List[Dict[str, Any]]) -> Path:
    console.print(Panel.fit(f"Generating outputs: {model_name}", style="bold cyan"))

    tokenizer = load_tokenizer()
    model = load_causal_model(model_path)

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(prompts, start=1):
        gen = generate_one(model, tokenizer, row["prompt_messages"])
        out_rows.append({
            "model_name": model_name,
            "model_path": str(model_path),
            "prompt_id": row["prompt_id"],
            "source_split": row["source_split"],
            "prompt": row["prompt"],
            "prompt_messages": row["prompt_messages"],
            "response": gen["response"],
            "input_token_count": gen["input_token_count"],
            "output_token_count": gen["output_token_count"],
        })

        if i % 10 == 0 or i == len(prompts):
            console.print(f"[green]{model_name}[/green]: generated {i}/{len(prompts)}")

    out_path = OUTPUT_DIR / f"{model_name}_outputs.jsonl"
    write_jsonl(out_path, out_rows)

    # free memory before loading next model
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit("Generate evaluation outputs: SFT / PPO / DPO", style="bold green"))

    prompts = prepare_eval_prompts(EVAL_PROMPT_LIMIT, EVAL_SPLIT)
    console.print(f"Prepared {len(prompts)} unique prompts from split='{EVAL_SPLIT}'")

    prompts_path = OUTPUT_DIR / "eval_prompts.jsonl"
    write_jsonl(prompts_path, prompts)

    console.print("\n[bold]Prompt set summary[/bold]")
    pprint({
        "prompt_limit": EVAL_PROMPT_LIMIT,
        "split": EVAL_SPLIT,
        "saved_prompt_file": str(prompts_path),
        "sft_path": str(SFT_PATH),
        "ppo_path": str(PPO_PATH),
        "dpo_path": str(DPO_PATH),
    })

    sft_out = run_model_generation("sft", SFT_PATH, prompts)
    ppo_out = run_model_generation("ppo_hh", PPO_PATH, prompts)
    dpo_out = run_model_generation("dpo_hh", DPO_PATH, prompts)

    manifest = {
        "prompt_file": str(prompts_path),
        "outputs": {
            "sft": str(sft_out),
            "ppo_hh": str(ppo_out),
            "dpo_hh": str(dpo_out),
        },
        "count_prompts": len(prompts),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "deterministic_decoding": True,
    }

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print("\n[bold green]Generation complete.[/bold green]")
    pprint(manifest)


if __name__ == "__main__":
    main()