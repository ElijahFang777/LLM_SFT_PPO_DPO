from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from accelerate import PartialState
from datasets import DatasetDict, load_from_disk
from peft import PeftConfig, PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
BASE_MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"
SFT_PATH = PROJECT_ROOT / "checkpoints" / "sft" / "qwen25_15b_lora_sft" / "final"
RM_PATH = PROJECT_ROOT / "checkpoints" / "rm_hh" / "qwen25_15b_reward_lora" / "final"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prefs_hh" / "hh_rlhf_explicit"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "ppo_hh" / "qwen25_15b_ppo_hh_full"
# OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "ppo_hh" / "qwen25_15b_ppo_hh" # Smoke test 
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_hh"

SEED = 42
MAX_PROMPT_LENGTH = 384

# Use environment variables for quick smoke tests or larger runs.
# Examples:
#   export PPO_TRAIN_LIMIT=5000
#   export PPO_VAL_LIMIT=500
#   export PPO_TOTAL_EPISODES=2000
PPO_TRAIN_LIMIT = int(os.environ.get("PPO_TRAIN_LIMIT", "0"))
PPO_VAL_LIMIT = int(os.environ.get("PPO_VAL_LIMIT", "0"))
PPO_TOTAL_EPISODES = int(os.environ.get("PPO_TOTAL_EPISODES","10000"))


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


def load_policy_model(model_path: Path, trainable: bool):
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


def load_seqcls_model(model_path: Path, trainable: bool):
    bnb_config = build_bnb_config()

    if is_adapter_checkpoint(model_path):
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            num_labels=1,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(model_path), is_trainable=trainable)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=1,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model


def prepare_prompt_dataset(ds, tokenizer, max_prompt_length: int):

    def render_prompt_messages(prompt_messages: List[Dict[str, str]]) -> str:
        # Preferred path for chat models (Qwen, etc.)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback if chat template is unavailable
        parts = [f"{m['role']}: {m['content']}" for m in prompt_messages]
        parts.append("assistant:")
        return "\n\n".join(parts)

    def tokenize(example: Dict[str, str]):
        prompt_text = render_prompt_messages(example["prompt_messages"])
        input_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids), "prompt_text": prompt_text}

    out = ds.map(
        tokenize,
        remove_columns=ds.column_names,
        desc="Tokenizing PPO prompt-only dataset",
    )
    out = out.filter(
        lambda x: x["lengths"] <= max_prompt_length,
        desc="Filtering long prompts",
    )
    return out


def main() -> None:
    console.print(Panel.fit("PPO-HH training", style="bold cyan"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match your SFT chat-template termination if available.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id < 0:
        stop_token_id = tokenizer.eos_token_id
    else:
        stop_token_id = im_end_id

    console.print(f"Tokenizer pad token: {tokenizer.pad_token}")
    console.print(f"Tokenizer eos token: {tokenizer.eos_token}")
    console.print(f"Using stop_token_id: {stop_token_id}")

    console.print("\n[bold]Loading processed HH dataset...[/bold]")
    ds = load_from_disk(str(DATA_PATH))
    ds = DatasetDict({
       "train": ds["train"].select_columns(["prompt_messages"]),
        "validation": ds["validation"].select_columns(["prompt_messages"]),
    })

    if PPO_TRAIN_LIMIT > 0:
        ds["train"] = ds["train"].select(range(min(PPO_TRAIN_LIMIT, len(ds["train"]))))
    if PPO_VAL_LIMIT > 0:
        ds["validation"] = ds["validation"].select(range(min(PPO_VAL_LIMIT, len(ds["validation"]))))

    console.print(f"raw train prompts: {len(ds['train'])}")
    console.print(f"raw validation prompts: {len(ds['validation'])}")

    with PartialState().local_main_process_first():
        train_dataset = prepare_prompt_dataset(ds["train"], tokenizer, MAX_PROMPT_LENGTH)
        eval_dataset = prepare_prompt_dataset(ds["validation"], tokenizer, MAX_PROMPT_LENGTH)

    # PPO example expects prompts not to end with EOS.
    if len(train_dataset) > 0:
        assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, \
            "The last token in the prompt should not be an EOS token."

    console.print(f"prepared train prompts: {len(train_dataset)}")
    console.print(f"prepared validation prompts: {len(eval_dataset)}")

    console.print("\n[bold]Loading policy model from SFT checkpoint...[/bold]")
    policy = load_policy_model(SFT_PATH, trainable=True)

    console.print("[bold]Loading frozen reference policy...[/bold]")
    ref_model = freeze_model(load_policy_model(SFT_PATH, trainable=False))

    console.print("[bold]Loading frozen reward model...[/bold]")
    reward_model = freeze_model(load_seqcls_model(RM_PATH, trainable=False))
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    console.print("[bold]Loading value model (trainable)...[/bold]")
    value_model = load_seqcls_model(RM_PATH, trainable=True)
    value_model.config.pad_token_id = tokenizer.pad_token_id

    total_episodes = min(PPO_TOTAL_EPISODES, len(train_dataset))
    console.print(f"\n[bold]PPO total episodes: {total_episodes}[/bold]")
    ppo_config = PPOConfig(
    output_dir=str(OUTPUT_DIR),
    sft_model_path=str(SFT_PATH),
    reward_model_path=str(RM_PATH),

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    learning_rate=3e-6,
    num_ppo_epochs=1,
    num_mini_batches=1,
    total_episodes=total_episodes,

    local_rollout_forward_batch_size=8,
    response_length=64,
    stop_token_id=stop_token_id,
    missing_eos_penalty=1.0,
    temperature=0.7,

    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    report_to="none",
    seed=SEED,

    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    dataloader_num_workers=8,
)

    
    console.print("\n[bold]Config summary[/bold]")
    pprint({
        "base_model_path": str(BASE_MODEL_PATH),
        "sft_path": str(SFT_PATH),
        "rm_path": str(RM_PATH),
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "response_length": 64,
        "train_prompts": len(train_dataset),
        "val_prompts": len(eval_dataset),
        "ppo_total_episodes": total_episodes,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    console.print("\n[bold]Building PPOTrainer...[/bold]")
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    console.print(Panel.fit("Starting PPO-HH training", style="bold green"))
    


    

    train_result = trainer.train()

    console.print(Panel.fit("Saving final PPO checkpoint", style="bold yellow"))
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    train_metrics = {
        "train_prompts": len(train_dataset),
        "eval_prompts": len(eval_dataset),
        "ppo_total_episodes": total_episodes,
    }

    if train_result is not None and hasattr(train_result, "metrics") and train_result.metrics is not None:
        train_metrics.update(dict(train_result.metrics))
    else:
        train_metrics["train_result_returned"] = False

        # Try to recover the last logged PPO metrics if available
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            if isinstance(last_log, dict):
                for k, v in last_log.items():
                    if isinstance(v, (int, float, str, bool)):
                        train_metrics[f"last_log/{k}"] = v

    with open(final_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    #save the PPO log history for the report
    with open(final_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)


    console.print("\n[bold green]PPO-HH training complete.[/bold green]")
    console.print(f"Saved model to: {final_dir}")
    console.print("\nFinal metrics:")
    pprint(train_metrics)

    # Optional: TRL example calls generate_completions() after training.
    try:
        console.print("\n[bold]Generating sample completions...[/bold]")
        trainer.generate_completions()
    except Exception as e:
        console.print(f"[yellow]Skipping generate_completions(): {e}[/yellow]")


if __name__ == "__main__":
    main()
