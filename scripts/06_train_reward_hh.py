from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardConfig, RewardTrainer


console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_explicit"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "rm_hh" / "qwen25_15b_reward_lora"
LOG_DIR = PROJECT_ROOT / "logs" / "rm_hh"

MAX_LENGTH = 512
SEED = 42

# Optional debug limits:
#   export RM_TRAIN_LIMIT=5000
#   export RM_VAL_LIMIT=500
RM_TRAIN_LIMIT = int(os.environ.get("RM_TRAIN_LIMIT", "0"))
RM_VAL_LIMIT = int(os.environ.get("RM_VAL_LIMIT", "0"))


def join_prompt_and_response(prompt: str, response: str) -> str:
    """
    Build a full dialogue text for reward modeling.

    The processed HH dataset stores `prompt` (context turns) and `chosen/rejected`
    (assistant completion only) separately. Reward training should score complete
    prompt+assistant continuations, not the assistant completion in isolation.
    """
    prompt = str(prompt).strip()
    response = str(response).strip()

    if prompt.endswith("Assistant:"):
        return f"{prompt} {response}".strip()
    return f"{prompt}\n\nAssistant: {response}".strip()


def format_reward_pair(example):
    return {
        "prompt": example["prompt"],
        "chosen": join_prompt_and_response(example["prompt"], example["chosen"]),
        "rejected": join_prompt_and_response(example["prompt"], example["rejected"]),
    }




def main() -> None:
    console.print(Panel.fit("Reward model training: HH-RLHF", style="bold cyan"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading processed HH preference dataset...[/bold]")
    ds = load_from_disk(str(DATA_PATH))
    console.print(ds)

    # Keep only the columns RewardTrainer needs.
    ds = DatasetDict({
        "train": ds["train"].select_columns(["prompt", "chosen", "rejected"]),
        "validation": ds["validation"].select_columns(["prompt", "chosen", "rejected"]),
        "test": ds["test"].select_columns(["prompt", "chosen", "rejected"]),
    })

    ds = DatasetDict({
        "train": ds["train"].map(format_reward_pair, desc="Formatting train pairs for RM"),
        "validation": ds["validation"].map(format_reward_pair, desc="Formatting validation pairs for RM"),
        "test": ds["test"].map(format_reward_pair, desc="Formatting test pairs for RM"),
    })


    if RM_TRAIN_LIMIT > 0:
        ds["train"] = ds["train"].select(range(min(RM_TRAIN_LIMIT, len(ds["train"]))))
    if RM_VAL_LIMIT > 0:
        ds["validation"] = ds["validation"].select(range(min(RM_VAL_LIMIT, len(ds["validation"]))))

    console.print(f"train rows: {len(ds['train'])}")
    console.print(f"validation rows: {len(ds['validation'])}")
    console.print(f"test rows: {len(ds['test'])}")

    console.print("\n[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.print(f"Tokenizer pad token: {tokenizer.pad_token}")
    console.print(f"Tokenizer eos token: {tokenizer.eos_token}")

    console.print("\n[bold]Configuring 4-bit quantization...[/bold]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    console.print("\n[bold]Loading sequence-classification reward model...[/bold]")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    console.print("[bold]Configuring LoRA...[/bold]")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )

    console.print("[bold]RewardTrainer configuration...[/bold]")
    reward_config = RewardConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=MAX_LENGTH,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        weight_decay=0.0,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=8,
        remove_unused_columns=True,
        center_rewards_coefficient=1e-2,
    )

    console.print("\n[bold]Config summary[/bold]")
    pprint({
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "max_length": MAX_LENGTH,
        "train_rows": len(ds["train"]),
        "val_rows": len(ds["validation"]),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    console.print("\n[bold]Building RewardTrainer...[/bold]")
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    console.print(Panel.fit("Starting reward-model training", style="bold green"))
    train_result = trainer.train()

    console.print(Panel.fit("Running final validation", style="bold yellow"))
    eval_metrics = trainer.evaluate()

    console.print(Panel.fit("Saving final reward model", style="bold magenta"))
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(ds["train"])
    train_metrics["eval_samples"] = len(ds["validation"])

    with open(final_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    with open(final_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    console.print("\n[bold green]Reward-model training complete.[/bold green]")
    console.print(f"Saved model to: {final_dir}")
    console.print("\nTrain metrics:")
    pprint(train_metrics)
    console.print("\nEval metrics:")
    pprint(eval_metrics)


if __name__ == "__main__":
    main()
