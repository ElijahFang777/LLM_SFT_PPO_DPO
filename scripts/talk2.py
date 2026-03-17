#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftConfig, PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

console = Console()

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
BASE_MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"

MODEL_REGISTRY: Dict[str, Path] = {
    "BASE": BASE_MODEL_PATH,
    "SFT": PROJECT_ROOT / "checkpoints" / "sft" / "qwen25_15b_lora_sft" / "final",
    "PPO": PROJECT_ROOT / "checkpoints" / "ppo_hh" / "qwen25_15b_ppo_hh_full" / "final",
    "DPO": PROJECT_ROOT / "checkpoints" / "dpo_hh" / "qwen25_15b_dpo_hh_full" / "final",
}

CONFIG_PATH = PROJECT_ROOT / "configs" / "talk2_config.json"

DEFAULT_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 256,
    "system_prompt": "You are a helpful, concise assistant.",
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        ensure_parent_dir(CONFIG_PATH)
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = DEFAULT_CONFIG.copy()
    config.update(data)
    return config


def save_config(config: Dict[str, Any]) -> None:
    ensure_parent_dir(CONFIG_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def show_current_config(config: Dict[str, Any]) -> None:
    table = Table(title="Current chat configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for key in ["temperature", "top_p", "max_new_tokens", "system_prompt"]:
        table.add_row(key, str(config[key]))
    table.add_row("config_path", str(CONFIG_PATH))
    console.print(table)


def interactive_config_editor() -> None:
    config = load_config()
    console.print(Panel.fit("talk2.py configuration editor", style="bold cyan"))
    show_current_config(config)

    temperature = FloatPrompt.ask(
        "Temperature",
        default=float(config["temperature"]),
    )
    top_p = FloatPrompt.ask(
        "top_p",
        default=float(config["top_p"]),
    )
    max_new_tokens = int(
        FloatPrompt.ask(
            "max_new_tokens",
            default=float(config["max_new_tokens"]),
        )
    )
    system_prompt = Prompt.ask(
        "System prompt",
        default=str(config["system_prompt"]),
    )

    new_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "system_prompt": system_prompt,
    }

    save_config(new_config)
    console.print(Panel.fit("Configuration saved.", style="bold green"))
    show_current_config(new_config)


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str):
    model_path = MODEL_REGISTRY[model_name]
    bnb_config = build_bnb_config()

    if is_adapter_checkpoint(model_path):
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
            is_trainable=False,
        )
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


def render_chat_input(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: List[str] = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append("assistant:")
    return "\n\n".join(parts)


@torch.inference_mode()
def generate_reply(
    model,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt_text = render_chat_input(tokenizer, messages)
    device = get_model_device(model)

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    do_sample = temperature > 0.0
    output_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][enc["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply


def print_header(model_name: str, config: Dict[str, Any]) -> None:
    table = Table(title="talk2 session")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", model_name)
    table.add_row("Temperature", str(config["temperature"]))
    table.add_row("top_p", str(config["top_p"]))
    table.add_row("max_new_tokens", str(config["max_new_tokens"]))
    table.add_row("Config file", str(CONFIG_PATH))
    console.print(table)
    console.print(
        Panel.fit(
            "Type your message and press Enter.\n"
            "Press Ctrl+C to exit.\n"
            "Type /reset to clear chat history.\n"
            "Type /params to view current parameters.",
            style="bold blue",
        )
    )


def chat_loop(model_name: str) -> None:
    config = load_config()
    print_header(model_name, config)

    console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = load_tokenizer()
    console.print("[bold]Loading model...[/bold]")
    model = load_model(model_name)
    console.print(Panel.fit(f"{model_name} loaded successfully.", style="bold green"))

    messages: List[Dict[str, str]] = []
    system_prompt = str(config["system_prompt"]).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    try:
        while True:
            user_text = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

            if not user_text:
                continue

            if user_text.lower() == "/reset":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                console.print(Panel.fit("Conversation reset.", style="yellow"))
                continue

            if user_text.lower() == "/params":
                show_current_config(config)
                continue

            messages.append({"role": "user", "content": user_text})

            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                max_new_tokens=int(config["max_new_tokens"]),
            )

            console.print(Panel(reply or "[empty response]", title=model_name, style="green"))
            messages.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        console.print()
        console.print(Panel.fit("Chat ended by Ctrl+C.", style="bold red"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with local SFT/PPO/DPO models in the terminal.")
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model to chat with: BASE, SFT, PPO, or DPO.",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Open interactive config editor for temperature/top_p/max_new_tokens/system_prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        interactive_config_editor()
        return

    if not args.model:
        console.print("[bold red]Error:[/bold red] provide --model or use --config.")
        console.print('Example: python talk2.py --model SFT')
        sys.exit(1)

    if args.model not in MODEL_REGISTRY:
        console.print(f"[bold red]Unknown model:[/bold red] {args.model}")
        sys.exit(1)

    chat_loop(args.model)


if __name__ == "__main__":
    main()