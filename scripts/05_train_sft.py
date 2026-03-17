from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sft" / "daring_anteater_sft"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "sft" / "qwen25_15b_lora_sft"
LOG_DIR = PROJECT_ROOT / "logs" / "sft"

MAX_LENGTH = 512
SEED = 42


def main() -> None:
    console.print(Panel.fit("SFT training: Qwen2.5-1.5B + LoRA + 4-bit", style="bold cyan"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading processed SFT dataset...[/bold]")
    ds = load_from_disk(str(DATA_PATH))
    console.print(ds)
    console.print(f"train rows: {len(ds['train'])}")
    console.print(f"validation rows: {len(ds['validation'])}")

    console.print("\n[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )

    # Important for Qwen2.5 SFT chat-template alignment
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"Tokenizer pad token: {tokenizer.pad_token}")
    console.print(f"Tokenizer eos token: {tokenizer.eos_token}")

    console.print("\n[bold]Configuring 4-bit quantization...[/bold]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    console.print("\n[bold]Loading base model...[/bold]")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    console.print("[bold]Configuring LoRA...[/bold]")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    console.print("[bold]Trainer configuration...[/bold]")
    sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    max_length=MAX_LENGTH,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, #
    #beatch_size=16 4*4 = 16 
    learning_rate=2e-4,
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
    report_to=[],
    seed=SEED,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": False},
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

    console.print("\n[bold]Building trainer...[/bold]")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    console.print(Panel.fit("Starting SFT training", style="bold green"))
    train_result = trainer.train()

    console.print(Panel.fit("Saving final checkpoint", style="bold yellow"))
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(ds["train"])
    metrics["eval_samples"] = len(ds["validation"])

    metrics_path = OUTPUT_DIR / "final" / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    console.print("\n[bold green]SFT training complete.[/bold green]")
    console.print(f"Saved model to: {OUTPUT_DIR / 'final'}")
    console.print(f"Saved metrics to: {metrics_path}")
    console.print("\nFinal metrics:")
    pprint(metrics)


if __name__ == "__main__":
    main()
