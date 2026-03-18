from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from peft import PeftConfig, PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
BASE_MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"
SFT_PATH = PROJECT_ROOT / "checkpoints" / "sft" / "qwen25_15b_lora_sft" / "final"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_explicit"
# OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "dpo_hh" / "qwen25_15b_dpo_hh" # somke test
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "dpo_hh" / "qwen25_15b_dpo_hh_full" 
LOG_DIR = PROJECT_ROOT / "logs" / "dpo_hh"

SEED = 42
MAX_LENGTH = 512

# Optional quick limits for smoke tests:
#   export DPO_TRAIN_LIMIT=5000
#   export DPO_VAL_LIMIT=500
DPO_TRAIN_LIMIT = int(os.environ.get("DPO_TRAIN_LIMIT", "0"))
DPO_VAL_LIMIT = int(os.environ.get("DPO_VAL_LIMIT", "0"))


def is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_causal_model(model_path: Path, trainable: bool):
    bnb_config = build_bnb_config()

    if is_adapter_checkpoint(model_path):
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(model_path), is_trainable=trainable)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model


def main() -> None:
    console.print(Panel.fit("DPO-HH training", style="bold cyan"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )

    # Keep DPO tokenization aligned with SFT training for Qwen chat templates.
    # In SFT we explicitly use <|im_end|> as EOS, so DPO should do the same.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id >= 0:
        tokenizer.eos_token = "<|im_end|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"Tokenizer pad token: {tokenizer.pad_token}")
    console.print(f"Tokenizer eos token: {tokenizer.eos_token}")

    console.print("\n[bold]Loading processed HH preference dataset...[/bold]")
   
    #dataset-boundary problem
    # ds = load_from_disk(str(DATA_PATH))
    # ds = DatasetDict({
    #     "train": ds["train"].select_columns(["prompt", "chosen", "rejected"]),
    #     "validation": ds["validation"].select_columns(["prompt", "chosen", "rejected"]),
    #     "test": ds["test"].select_columns(["prompt", "chosen", "rejected"]),
    # })

    ds = load_from_disk(str(DATA_PATH))
    def to_conversational(split):
        split = split.select_columns(["prompt_messages", "chosen_messages", "rejected_messages"])
        split = split.rename_column("prompt_messages", "prompt")
        split = split.rename_column("chosen_messages", "chosen")
        split = split.rename_column("rejected_messages", "rejected")
        return split
    ds = DatasetDict({
        "train": to_conversational(ds["train"]),
        "validation": to_conversational(ds["validation"]),
        "test": to_conversational(ds["test"]),
    })

    if DPO_TRAIN_LIMIT > 0:
        ds["train"] = ds["train"].shuffle(seed=SEED).select(range(min(DPO_TRAIN_LIMIT, len(ds["train"]))))
    if DPO_VAL_LIMIT > 0:
        ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(min(DPO_VAL_LIMIT, len(ds["validation"]))))

    console.print(f"train rows: {len(ds['train'])}")
    console.print(f"validation rows: {len(ds['validation'])}")
    console.print(f"test rows: {len(ds['test'])}")

    console.print("\n[bold]Loading policy model from SFT checkpoint...[/bold]")
    model = load_causal_model(SFT_PATH, trainable=True)

    console.print("[bold]Loading frozen reference model from same SFT checkpoint...[/bold]")
    ref_model = freeze_model(load_causal_model(SFT_PATH, trainable=False))

    # dpo_config = DPOConfig(
    #     output_dir=str(OUTPUT_DIR),
    #     max_length=MAX_LENGTH,
    #     truncation_mode="keep_start",
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=5e-6,
    #     num_train_epochs=1,
    #     lr_scheduler_type="cosine",
    #     warmup_ratio=0.03,
    #     logging_steps=25,
    #     save_steps=1000,
    #     eval_steps=1000,
    #     eval_strategy="steps",
    #     save_strategy="steps",
    #     save_total_limit=2,
    #     bf16=True,
    #     fp16=False,
    #     gradient_checkpointing=False,
    #     max_grad_norm=1.0,
    #     weight_decay=0.0,
    #     report_to="none",
    #     seed=SEED,
    #     dataloader_num_workers=8,
    #     remove_unused_columns=True,
    #     disable_dropout=True,
    #     beta=0.1,
    #     precompute_ref_log_probs=True,
    #     precompute_ref_batch_size=8,
    # )

    dpo_config = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    max_length=384,                  # was 512
    truncation_mode="keep_start",

    per_device_train_batch_size=2,   # was 4
    per_device_eval_batch_size=2,    # was 4
    gradient_accumulation_steps=8,   # keep effective batch size similar

    learning_rate=5e-6,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,

    bf16=True,
    fp16=False,
    gradient_checkpointing=True,     # was False
    max_grad_norm=1.0,
    weight_decay=0.0,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=4,        # was 8
    remove_unused_columns=True,
    disable_dropout=True,

    beta=0.1,
    precompute_ref_log_probs=True,
    precompute_ref_batch_size=2,     # was 8
    )



    console.print("\n[bold]Config summary[/bold]")
    pprint({
        "base_model_path": str(BASE_MODEL_PATH),
        "sft_path": str(SFT_PATH),
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "max_length": MAX_LENGTH,
        "train_rows": len(ds["train"]),
        "val_rows": len(ds["validation"]),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    console.print("\n[bold]Building DPOTrainer...[/bold]")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    console.print(Panel.fit("Starting DPO-HH training", style="bold green"))
    train_result = trainer.train()

    console.print(Panel.fit("Running final validation", style="bold yellow"))
    eval_metrics = trainer.evaluate()

    console.print(Panel.fit("Saving final DPO checkpoint", style="bold magenta"))
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()

    train_metrics = {
        "train_rows": len(ds["train"]),
        "eval_rows": len(ds["validation"]),
    }

    if train_result is not None and hasattr(train_result, "metrics") and train_result.metrics is not None:
        train_metrics.update(dict(train_result.metrics))
    else:
        train_metrics["train_result_returned"] = False
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            if isinstance(last_log, dict):
                for k, v in last_log.items():
                    if isinstance(v, (int, float, str, bool)):
                        train_metrics[f"last_log/{k}"] = v

    with open(final_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    with open(final_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    with open(final_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    console.print("\n[bold green]DPO-HH training complete.[/bold green]")
    console.print(f"Saved model to: {final_dir}")
    console.print("\nTrain metrics:")
    pprint(train_metrics)
    console.print("\nEval metrics:")
    pprint(eval_metrics)


if __name__ == "__main__":
    main()
