#!/usr/bin/env python3
# scripts/train_base.py
"""
Train script for causal LM using LoRA/QLoRA with Gradio integration
"""

import argparse
import logging
import json
import time
from pathlib import Path
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import numpy as np
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def check_model_parameters(model, prefix=""):
    """Check and log model parameter states"""
    trainable_params = 0
    frozen_params = 0
    all_params = 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"MODEL PARAMETER CHECK: {prefix}")
    logger.info(f"{'='*50}")
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            status = "TRAINABLE"
        else:
            frozen_params += param.numel()
            status = "FROZEN"
        
        logger.info(f"{status}: {name} | shape: {param.shape} | dtype: {param.dtype}")
    
    logger.info(f"\nSummary for {prefix}:")
    logger.info(f"Total parameters: {all_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} ({frozen_params/all_params*100:.2f}%)")
    logger.info(f"{'='*50}\n")
    
    return trainable_params, frozen_params, all_params

class TensorBoardCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / "tb_runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tb_subdir = time.strftime("%Y%m%d-%H%M%S")
        self.logdir = str(self.run_dir / tb_subdir)
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.records = []

    def _record_entry(self, step: int, entry: dict):
        entry_with_step = {"step": int(step), "timestamp": int(time.time())}
        entry_with_step.update(entry)
        self.records.append(entry_with_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = int(state.global_step)
        if "loss" in logs:
            self.writer.add_scalar("train/loss", float(logs["loss"]), step)
        if "learning_rate" in logs:
            lr = logs["learning_rate"][0] if isinstance(logs["learning_rate"], list) else logs["learning_rate"]
            self.writer.add_scalar("train/learning_rate", float(lr), step)
        if "epoch" in logs:
            self.writer.add_scalar("train/epoch", float(logs["epoch"]), step)

        entry = {}
        for k in ("loss", "learning_rate", "epoch"):
            if k in logs:
                val = logs[k][0] if isinstance(logs[k], list) else logs[k]
                entry[k] = float(val)
        if entry:
            self._record_entry(step, entry)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = int(state.global_step)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"eval/{k}", v, step)
        
        entry = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if entry:
            self._record_entry(step, {"eval": entry})

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass
        
        out_path = self.output_dir / "training_logs.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.records)} log entries to {out_path}")
            logger.info(f"TensorBoard logs in: {self.logdir}")
        except Exception as e:
            logger.error(f"Failed to write logs: {e}")

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(path: Path) -> list:
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objs.append(json.loads(line))
    return objs

def make_dataset_from_jsonl(jsonl_path: Path, tokenizer, max_length: int, max_samples: int = None):
    raw = read_jsonl(jsonl_path)
    if max_samples and len(raw) > max_samples:
        raw = raw[:max_samples]
    
    items = []
    for item in raw:
        p = item.get("prompt", "").strip()
        r = item.get("response", "").strip()
        full_text = f"{p}{r}"
        
        tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"][0].tolist()
        attention_mask = tokens["attention_mask"][0].tolist()
        
        # Mask prompt tokens in labels
        prompt_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        
        # Ensure we don't mask beyond the sequence length
        if prompt_len >= len(input_ids):
            # If prompt is longer than the sequence, just mask everything
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Truncate labels if needed
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        
        items.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })
    
    ds = Dataset.from_dict({
        "input_ids": [item["input_ids"] for item in items],
        "labels": [item["labels"] for item in items],
        "attention_mask": [item["attention_mask"] for item in items]
    })
    return ds.with_format(type="torch")

def compute_metrics(eval_pred):
    """
    Compute token-level loss for causal LM.
    Properly handles masked tokens (-100) that should be excluded from evaluation.
    """
    try:
        predictions, labels = eval_pred
        if predictions is None or predictions.ndim != 3:
            logger.debug("Skipping metric computation: invalid predictions shape")
            return {}
        
        # Convert to torch tensors
        logits = torch.from_numpy(predictions)
        labels_t = torch.from_numpy(labels).long()
        
        # Create mask for valid tokens (non -100)
        mask = labels_t != -100
        
        # Count valid tokens
        num_valid_tokens = mask.sum().item()
        if num_valid_tokens == 0:
            logger.warning("No valid tokens found for evaluation (all masked)")
            return {}
        
        # Filter out invalid tokens before gathering
        valid_logits = logits[mask]
        valid_labels = labels_t[mask]
        
        # Safety check - ensure we have valid tokens to evaluate
        if len(valid_labels) == 0:
            logger.warning("No valid tokens after filtering masked positions")
            return {}
        
        # Compute log probabilities only for valid tokens
        log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # Gather log probs for the true labels (now all are valid)
        labels_exp = valid_labels.unsqueeze(-1)
        # Additional safety check for label values
        if torch.any(valid_labels < 0) or torch.any(valid_labels >= logits.shape[-1]):
            logger.warning(f"Found invalid label values. Min: {valid_labels.min().item()}, Max: {valid_labels.max().item()}, Vocab size: {logits.shape[-1]}")
            # Clip labels to valid range as fallback
            valid_labels = torch.clamp(valid_labels, 0, logits.shape[-1] - 1)
            labels_exp = valid_labels.unsqueeze(-1)
        
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels_exp).squeeze(-1)
        
        # Compute negative log likelihood for valid tokens
        nll = -token_log_probs
        token_loss = nll.mean().item()
        
        logger.debug(f"Computed metrics: token_loss={token_loss:.4f}, valid_tokens={num_valid_tokens}")
        
        return {
            "token_loss": float(token_loss),
            "valid_tokens": num_valid_tokens  # Added for debugging
        }
    except Exception as e:
        logger.warning(f"Error in compute_metrics: {str(e)}")
        logger.debug(f"Full error details: {repr(e)}")
        logger.debug(f"Predictions shape: {predictions.shape if predictions is not None else 'None'}")
        logger.debug(f"Labels shape: {labels.shape if labels is not None else 'None'}")
        return {}

def main():
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument("--config_dir", type=str, default="configs", help="Directory containing YAML configs")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory if it exists")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use")
    parser.add_argument("--max_eval_samples", type=int, default=50, help="Maximum number of evaluation samples to use")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for best model selection")
    
    # LoRA/QLoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA (4-bit quantization)")
    
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=None, help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=None, help="Use BF16 mixed precision training")

    args = parser.parse_args()

    # Load configs
    cfg_dir = Path(args.config_dir)
    training_cfg = load_yaml(cfg_dir / "training_args.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    
    # Setup
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    if args.fp16 is None:
        args.fp16 = False if args.use_qlora else (device == "cuda")
    
    if args.bf16 is None:
        args.bf16 = True if args.use_qlora else False

    # Tokenizer setup
    tokenizer_name = model_cfg.get("tokenizer_name_or_path", model_cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Quantization config
    bnb_config = None
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info("Using 4-bit NF4 quantization")

    # Model loading
    logger.info(f"Loading model from: {model_cfg['model_name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.use_qlora else torch.float16,
    )
    
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    # Prepare for k-bit training before applying LoRA
    if args.use_qlora:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=True
        )
        logger.info("Model prepared for k-bit training")

    # Apply LoRA AFTER preparing for k-bit training
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        target_modules = model_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA adapter applied with rank={args.lora_r}, alpha={args.lora_alpha}")
        model.print_trainable_parameters()
    else:
        # If not using LoRA, ensure the model is trainable
        logger.info("No LoRA - setting all parameters to trainable")
        for param in model.parameters():
            param.requires_grad = True

    # CRITICAL: Disable caching when using gradient checkpointing
    model.config.use_cache = False
    logger.info("Disabled model caching (required for gradient checkpointing)")

    # Verify model parameter states
    check_model_parameters(model, "AFTER ALL PREPARATION")

    # Check if we have any trainable parameters
    has_trainable_params = any(p.requires_grad for p in model.parameters())
    if not has_trainable_params:
        logger.error("CRITICAL ERROR: No trainable parameters found!")
        logger.error("This will cause the 'does not require grad' error")
        logger.error("Please check your LoRA/QLoRA configuration")
        raise ValueError("No trainable parameters in model")

    # Datasets
    max_length = model_cfg.get("max_input_length", 512)
    train_ds = make_dataset_from_jsonl(Path(args.train_file), tokenizer, max_length, args.max_train_samples)
    valid_ds = make_dataset_from_jsonl(Path(args.valid_file), tokenizer, max_length, args.max_eval_samples)
    logger.info(f"Training set size: {len(train_ds)}")
    logger.info(f"Validation set size: {len(valid_ds)}")

    # Determine greater_is_better based on metric
    greater_is_better = False  # default for loss-based metrics
    metric_lower = args.metric_for_best_model.lower()
    if "token_loss" in metric_lower:
        greater_is_better = True

    # Training setup - with QLoRA-specific optimizations
    optim = "paged_adamw_32bit" if args.use_qlora else "adamw_torch"
    lr_scheduler_type = "cosine"
    
    logger.info(f"Using optimizer: {optim}")
    logger.info(f"Using LR scheduler: {lr_scheduler_type}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=not args.use_qlora and device == "cuda",  # Don't use fp16 with 4-bit
        bf16=args.use_qlora and device == "cuda",      # Use bf16 with 4-bit if available
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Critical for compatibility
        report_to="tensorboard",
        logging_dir=str(Path(args.output_dir) / "logs"),
        overwrite_output_dir=args.overwrite_output_dir,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=0.03,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(output_dir=args.output_dir)],
    )

    # Train
    logger.info("***** Starting training *****")
    logger.info(f"  Output directory = {args.output_dir}")
    logger.info(f"  Best model metric = {args.metric_for_best_model} (greater_is_better={greater_is_better})")
    logger.info(f"  Training arguments: {training_args}")
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except RuntimeError as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("This is likely due to gradient checkpointing issues with quantization")
        logger.error("Try the following fixes:")
        logger.error("1. Disable gradient checkpointing by setting gradient_checkpointing=False")
        logger.error("2. Reduce batch size")
        logger.error("3. Use smaller model")
        logger.error("4. Disable QLoRA and use regular LoRA")
        raise

    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # Also save adapter separately if using LoRA
    if args.use_lora:
        adapter_dir = Path(args.output_dir) / "adapter"
        adapter_dir.mkdir(exist_ok=True)
        model.save_pretrained(adapter_dir)
        logger.info(f"LoRA adapter saved to {adapter_dir}")

if __name__ == "__main__":
    main()