#!/usr/bin/env python3
# scripts/train_base.py
"""
Train script for tinyllama / causal LM using LoRA or QLoRA (4-bit).
Reads configs from configs/*.yaml and trains a causal LM with PEFT.
Designed for Colab T4 (8GB) usage — conservative defaults included.

Usage:
    python scripts/train_base.py \
        --config_dir configs \
        --train_file data/train.jsonl \
        --valid_file data/valid.jsonl

"""

import argparse
import os
import logging
from pathlib import Path
import json
import math
import random
from typing import Dict, List

import torch
import yaml
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(prompt: str, response: str, prompt_template: str) -> str:
    """Format prompt+response according to prompt_template from model_config.yaml"""
    return prompt_template.replace("{prompt}", prompt).replace("{response}", response)


def read_jsonl(path: Path) -> List[Dict]:
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    return objs


def make_dataset_from_jsonl(jsonl_path: Path, tokenizer, prompt_template: str, max_length: int):
    """Load JSONL where each item has 'prompt' and 'response', return HF Dataset tokenized."""
    raw = read_jsonl(jsonl_path)
    texts = []
    for item in raw:
        prompt = item.get("prompt", "").strip()
        response = item.get("response", "").strip()
        text = build_prompt(prompt, response, prompt_template)
        texts.append({"text": text})
    # Build a Dataset from list of dicts
    ds = Dataset.from_list(texts)

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Labels: for causal LM we can set labels = input_ids (Trainer will shift internally)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs", help="Directory containing YAML configs")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    training_cfg = load_yaml(cfg_dir / "training_args.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    eval_cfg = load_yaml(cfg_dir / "eval_config.yaml")

    # Seed
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    model_name_or_path = training_cfg.get("model_name_or_path", model_cfg.get("model_name_or_path"))
    if model_name_or_path is None:
        raise ValueError("Model name/path not specified in configs")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("tokenizer_name_or_path", model_name_or_path))
    # Ensure pad token exists (some causal models don't have pad_token)
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token, setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Decide quantization / bitsandbytes config
    use_qlora = training_cfg.get("qlora", {}).get("use_qlora", False)
    use_4bit = training_cfg.get("qlora", {}).get("use_4bit", False) and use_qlora

    bnb_config = None
    if use_4bit:
        # Setup BitsAndBytes config for 4-bit loading (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_cfg["qlora"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=training_cfg["qlora"].get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, training_cfg["qlora"].get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        logger.info(f"Using 4-bit QLoRA bitsandbytes config: {bnb_config}")

    # Load model (with or without 4-bit)
    try:
        if bnb_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    except Exception as e:
        logger.warning("Failed to load with device_map=auto or quant config, trying CPU load as fallback. Error: %s", e)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        model.to(device)

    # Prepare for k-bit training if using QLoRA
    if use_qlora and bnb_config is not None:
        logger.info("Preparing model for k-bit training (QLoRA flow).")
        model = prepare_model_for_kbit_training(model)

    # Build LoRA config if requested
    use_lora = training_cfg.get("lora", {}).get("use_lora", True)
    if use_lora:
        lora_cfg = training_cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", None),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA adapter attached to the model.")

    # Prepare datasets
    max_input_length = model_cfg.get("max_input_length", 512)
    train_ds = make_dataset_from_jsonl(Path(args.train_file), tokenizer, model_cfg.get("prompt_format", "{prompt}{response}"), max_input_length)
    valid_ds = make_dataset_from_jsonl(Path(args.valid_file), tokenizer, model_cfg.get("prompt_format", "{prompt}{response}"), max_input_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # TrainingArguments
    output_dir = training_cfg.get("output_dir", "models/finetuned_model")
    # Map fields from training_cfg into TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=training_cfg.get("num_train_epochs", 2),
        max_steps=training_cfg.get("max_steps", None),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        logging_steps=training_cfg.get("logging_steps", 50),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 200),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        fp16=training_cfg.get("fp16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "loss"),
        push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_ds))
    logger.info("  Num valid examples = %d", len(valid_ds))
    logger.info("  Output dir = %s", output_dir)

    trainer.train()
    logger.info("Training completed. Saving model...")

    # Save peft adapters & tokenizer properly
    # If using PEFT, save_pretrained will save adapter weights
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved model and tokenizer to %s", output_dir)

    # Optionally push to hub
    if training_args.push_to_hub:
        try:
            logger.info("Pushing model to the Hub...")
            trainer.push_to_hub()
            logger.info("Pushed to Hub.")
        except Exception as e:
            logger.warning("Failed to push to hub: %s", e)


if __name__ == "__main__":
    main()
