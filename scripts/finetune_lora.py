"""
Model-agnostic LoRA/QLoRA fine-tuning for HTR post-correction.

Works with any causal LM (Qwen, Llama, Mistral, Gemma, etc.).
Model, hyperparameters, and paths are configured via YAML.

Supports two backends:
    - "transformers": PEFT + TRL SFTTrainer (standard)
    - "unsloth": Unsloth (optimised, ~2x faster, less VRAM)

The fine-tuning framework is model-agnostic: switching models requires
only changing the model.name field in the YAML config file.

Usage:
    python scripts/finetune_lora.py --config configs/finetune_config.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset


# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": {
        "name": "Qwen/Qwen3-8B-Instruct",
        "quantization": None,
        "access_token": "",
        "dtype": "auto",
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    },
    "training": {
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "max_seq_length": 4096,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
    },
    "backend": {
        "engine": "transformers",
    },
    "paths": {
        "train": "data/finetuning/train.jsonl",
        "val": "data/finetuning/val.jsonl",
        "output": "models/finetuned",
    },
}


def load_config(config_path=None):
    """Load configuration from a YAML file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    return config


# ─── Data loading ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def format_chat(pair, tokenizer):
    """Convert a chat-format pair to the tokenizer's template."""
    return tokenizer.apply_chat_template(
        pair["messages"], tokenize=False, add_generation_prompt=False
    )


def prepare_dataset(path, tokenizer):
    """Load and format a JSONL dataset for training."""
    pairs = load_jsonl(path)
    texts = [format_chat(p, tokenizer) for p in pairs]
    return Dataset.from_dict({"text": texts})


# ─── Backend Transformers (PEFT + TRL) ────────────────────────────────────────

def train_transformers(config):
    """Fine-tune using the standard Transformers + PEFT + TRL stack."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    model_name = config["model"]["name"]
    token = config["model"].get("access_token") or None

    # Tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Datasets
    print("Loading datasets...", flush=True)
    train_dataset = prepare_dataset(config["paths"]["train"], tokenizer)
    val_dataset = prepare_dataset(config["paths"]["val"], tokenizer)
    print(f"  Train: {len(train_dataset)} exemples", flush=True)
    print(f"  Val:   {len(val_dataset)} exemples", flush=True)

    # Model
    print("Loading model...", flush=True)
    model_kwargs = {
        "token": token,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    # Dtype
    dtype_str = config["model"].get("dtype", "auto")
    if dtype_str == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype_str == "float16":
        model_kwargs["torch_dtype"] = torch.float16

    # Quantisation (QLoRA)
    if config["model"]["quantization"] == "4bit":
        print("  4-bit quantisation enabled (QLoRA)", flush=True)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # LoRA
    print("Configuring LoRA...", flush=True)
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config["training"].get("gradient_checkpointing"):
        model.enable_input_require_grads()

    # Training arguments
    output_dir = config["paths"]["output"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=10,
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy="epoch",
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\Starting training...", flush=True)
    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


# ─── Backend Unsloth ──────────────────────────────────────────────────────────

def train_unsloth(config):
    """Fine-tune using Unsloth (optimised, ~2x faster)."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model_name = config["model"]["name"]
    max_seq = config["training"]["max_seq_length"]
    use_4bit = config["model"]["quantization"] == "4bit"

    # Model + Tokenizer (Unsloth loads both together)
    print("Loading model with Unsloth...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=use_4bit,
        dtype=None,  # Unsloth auto-detects
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Datasets
    print("Loading datasets...", flush=True)
    train_dataset = prepare_dataset(config["paths"]["train"], tokenizer)
    val_dataset = prepare_dataset(config["paths"]["val"], tokenizer)
    print(f"  Train: {len(train_dataset)} exemples", flush=True)
    print(f"  Val:   {len(val_dataset)} exemples", flush=True)

    # LoRA via Unsloth
    print("Configuring LoRA (Unsloth)...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Training
    output_dir = config["paths"]["output"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=10,
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy="epoch",
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training (Unsloth)...", flush=True)
    trainer.train()

    # Save
    print(f"\Saving model to {output_dir}...", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(config_path=None):
    config = load_config(config_path)

    engine = config["backend"]["engine"]
    model_name = config["model"]["name"]
    output_dir = config["paths"]["output"]

    print("=" * 60)
    print("LoRA FINE-TUNING")
    print("=" * 60)
    print(f"Model:          {model_name}")
    print(f"Backend:        {engine}")
    print(f"Quantisation:   {config['model']['quantization'] or 'none (standard LoRA)'}")
    print(f"LoRA rank:      {config['lora']['r']}")
    print(f"LoRA alpha:     {config['lora']['alpha']}")
    print(f"Epochs:         {config['training']['epochs']}")
    print(f"Effective batch:  {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Learning rate:  {config['training']['learning_rate']}")
    print(f"Max seq length: {config['training']['max_seq_length']}")
    print(f"Output:         {output_dir}")
    print()

    if engine == "unsloth":
        result_dir = train_unsloth(config)
    else:
        result_dir = train_transformers(config)

    # Save config for reference
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(result_dir) / "finetune_config.json", "w") as f:
        json.dump({
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-agnostic LoRA fine-tuning")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    args = parser.parse_args()

    main(args.config)
