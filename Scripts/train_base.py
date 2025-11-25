#!/usr/bin/env python3
# scripts/train_base.py
"""
Train script for tinyllama / causal LM using LoRA or QLoRA (4-bit).
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import time
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class TensorBoardCallback(TrainerCallback):
    """
    Write training logs to TensorBoard and also keep an in-memory record saved as JSON at the end.
    Logs: train/loss, train/learning_rate, train/epoch, eval/* metrics (eval_loss, perplexity, token_loss...), wall_time
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / "tb_runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # writer will create a subdir under run_dir with timestamp
        tb_subdir = time.strftime("%Y%m%d-%H%M%S")
        self.logdir = str(self.run_dir / tb_subdir)
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.records = []
        self.last_log_step = None

    def _record_entry(self, step: int, entry: Dict):
        entry_with_step = {"step": int(step), "timestamp": int(time.time())}
        entry_with_step.update(entry)
        self.records.append(entry_with_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs example: {'loss': 2.345, 'learning_rate': 5e-05, 'epoch': 0.12}
        if logs is None:
            return
        step = int(state.global_step)
        # prefer logging at logging_steps multiples to avoid very frequent writes
        # but still accept any logs
        if "loss" in logs:
            try:
                self.writer.add_scalar("train/loss", float(logs["loss"]), step)
            except Exception:
                pass
        if "learning_rate" in logs:
            try:
                # sometimes learning_rate is a list
                lr = logs["learning_rate"]
                if isinstance(lr, list):
                    lr = lr[0]
                self.writer.add_scalar("train/learning_rate", float(lr), step)
            except Exception:
                pass
        if "epoch" in logs:
            try:
                self.writer.add_scalar("train/epoch", float(logs["epoch"]), step)
            except Exception:
                pass

        # record to JSON
        entry = {}
        for k in ("loss", "learning_rate", "epoch"):
            if k in logs:
                try:
                    entry[k] = float(logs[k]) if not isinstance(logs[k], list) else float(logs[k][0])
                except Exception:
                    entry[k] = logs[k]
        if entry:
            self._record_entry(step, entry)
            self.last_log_step = step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics is a dict returned by Trainer.evaluate() (contains eval_loss, plus compute_metrics outputs)
        if metrics is None:
            return
        step = int(state.global_step)
        # write all metrics as eval/<metric_name>
        for k, v in metrics.items():
            try:
                val = float(v)
                self.writer.add_scalar(f"eval/{k}", val, step)
            except Exception:
                # if not convertible to float, skip
                continue
        # also record them to JSON
        entry = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if entry:
            self._record_entry(step, {"eval": entry})

    def on_train_end(self, args, state, control, **kwargs):
        # flush writer and save JSON file
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass
        # Save records to JSON
        out_path = self.output_dir / "training_logs.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            print(f"[TensorBoardCallback] saved {len(self.records)} log entries to {out_path}")
            print(f"[TensorBoardCallback] TensorBoard logs in: {self.logdir}")
        except Exception as e:
            print(f"[TensorBoardCallback] failed to write logs: {e}")

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(prompt: str, response: str, prompt_template: str) -> str:
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
    """
    Load JSONL where each item has 'prompt' and 'response', return a HuggingFace Dataset
    prepared for causal LM training where prompt tokens are masked with -100 in labels.
    """
    from datasets import Dataset  # local import to avoid top-level import conflicts

    raw = read_jsonl(jsonl_path)
    items = []
    for item in raw:
        p = item.get("prompt", "").strip()
        r = item.get("response", "").strip()
        # if your prompt_template includes markers, build them here:
        # but we want prompt and response separately for masking
        items.append({"prompt": p, "response": r})

    # Tokenize prompts and responses separately (add_special_tokens=False to control concatenation)
    all_prompts = [it["prompt"] for it in items]
    all_resps = [it["response"] for it in items]

    # encode without truncation first
    enc_prompts = tokenizer(all_prompts, add_special_tokens=False)["input_ids"]
    enc_resps = tokenizer(all_resps, add_special_tokens=False)["input_ids"]

    input_ids_list = []
    labels_list = []
    attention_masks = []

    for p_ids, r_ids in zip(enc_prompts, enc_resps):
        # Build full sequence = prompt + response
        full = p_ids + r_ids

        # Truncate from the left if longer than max_length (we prefer to keep response)
        if len(full) > max_length:
            # Keep the last max_length tokens
            full = full[-max_length:]
            # Now compute where the response starts in the truncated sequence
            if len(r_ids) >= max_length:
                # response itself fills entire sequence, response starts at 0
                resp_start = 0
            else:
                # otherwise compute remaining prompt tokens in truncated sequence
                resp_start = max(0, len(full) - len(r_ids))
        else:
            resp_start = len(p_ids)

        # Build labels: mask prompt (-100), keep response token ids
        labels = [-100] * len(full)
        for i in range(resp_start, len(full)):
            labels[i] = full[i]

        # Right-pad to max_length with pad_token_id, and mask pads with -100 in labels
        pad_len = max_length - len(full)
        input_ids = full + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * len(full) + [0] * pad_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append(attention_mask)

    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_masks
    })
    # make PyTorch tensors when used by Trainer
    ds = ds.with_format(type="torch")
    return ds


def compute_metrics(eval_pred):
    """
    Compute token-level loss and perplexity for causal LM.
    eval_pred: tuple (predictions, labels)
      - predictions: logits np.array with shape (batch, seq_len, vocab_size) or (batch, seq_len) if prediction ids
      - labels: np.array with shape (batch, seq_len) with -100 mask
    Returns: dict with 'eval_loss' (if available), 'perplexity', 'token_loss'
    """
    try:
        predictions, labels = eval_pred
        # If predictions are token ids (greedy), we cannot compute token loss -> return empty
        if predictions is None:
            return {}
        # If predictions are logits (batch, seq_len, vocab_size)
        preds = np.array(predictions)
        lab = np.array(labels)

        # If predictions are (num_examples, ) of strings or ids -> cannot compute
        if preds.ndim != 3:
            return {}

        # convert to torch tensor for stable log_softmax
        logits = torch.from_numpy(preds)  # shape (B, S, V)
        labels_t = torch.from_numpy(lab).long()  # shape (B, S)

        # compute log-probs
        log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

        # flatten and mask
        B, S = labels_t.shape
        # create mask for valid tokens
        mask = labels_t != -100  # (B, S)
        if mask.sum().item() == 0:
            return {}

        # Gather log-probs of the true labels
        # gather requires expanding labels to last dim
        labels_exp = labels_t.unsqueeze(-1)  # (B, S, 1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels_exp).squeeze(-1)  # (B, S)

        # Negative log likelihood on non-masked tokens
        nll = -token_log_probs[mask]  # 1D tensor of nll per token
        token_loss = nll.mean().item()
        perplexity = float(torch.exp(torch.tensor(token_loss)).item())

        return {"token_loss": float(token_loss), "perplexity": perplexity}
    except Exception as e:
        # If anything goes wrong, return empty dict but don't crash training
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs", help="Directory containing YAML configs")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    
    # 添加可自定义参数
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--metric_for_best_model", type=str, default=None, help="Override metric for best model")
    
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    training_cfg = load_yaml(cfg_dir / "training_args.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    eval_cfg = load_yaml(cfg_dir / "eval_config.yaml")

    # 用命令行参数覆盖配置
    if args.num_train_epochs is not None:
        training_cfg["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        training_cfg["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        training_cfg["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        training_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps
    if args.metric_for_best_model is not None:
        training_cfg["metric_for_best_model"] = args.metric_for_best_model

    # 修复：确保 metric_for_best_model 是有效的
    valid_metrics = ['eval_loss', 'loss', 'epoch']
    metric_for_best_model = training_cfg.get("metric_for_best_model", "eval_loss")
    if metric_for_best_model not in valid_metrics:
        logger.warning(f"metric_for_best_model '{metric_for_best_model}' is not valid. Using 'eval_loss' instead.")
        metric_for_best_model = "eval_loss"
        training_cfg["metric_for_best_model"] = metric_for_best_model

    # 根据 metric 设置 greater_is_better
    greater_is_better = training_cfg.get("greater_is_better", False)
    if metric_for_best_model == "eval_loss":
        greater_is_better = False
    elif metric_for_best_model in ["accuracy", "f1"]:
        greater_is_better = True

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
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token, setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Decide quantization / bitsandbytes config
    use_qlora = training_cfg.get("qlora", {}).get("use_qlora", False)
    use_4bit = training_cfg.get("qlora", {}).get("use_4bit", False) and use_qlora

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_cfg["qlora"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=training_cfg["qlora"].get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, training_cfg["qlora"].get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        logger.info(f"Using 4-bit QLoRA bitsandbytes config: {bnb_config}")

    # Load model
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
    max_eval_samples = 50  # 只使用前50个样本进行评估
    if len(valid_ds) > max_eval_samples:
      valid_ds = valid_ds.select(range(max_eval_samples))
      logger.info(f"Limited validation set to {max_eval_samples} samples for memory optimization")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # TrainingArguments
    output_dir = training_cfg.get("output_dir", "models/finetuned_model")
    
    # 修复：确保 max_steps 和 num_train_epochs 的默认值
    max_steps = training_cfg.get("max_steps", -1)
    if max_steps is None:
        max_steps = -1

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=training_cfg.get("num_train_epochs", 2),
        max_steps=max_steps,
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 200),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        fp16=training_cfg.get("fp16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=metric_for_best_model,  # 使用修复后的值
        greater_is_better=greater_is_better,  # 根据指标类型设置
        push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
        report_to="tensorboard",            # 让 Trainer 也能上报（可选）
        logging_dir=str(Path(output_dir) / "tensorboard"),  # backup logging dir
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 添加计算指标的函数
        callbacks=[TensorBoardCallback(output_dir=output_dir)],
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_ds))
    logger.info("  Num valid examples = %d", len(valid_ds))
    logger.info("  Output dir = %s", output_dir)
    logger.info("  Metric for best model = %s", metric_for_best_model)

    trainer.train()
    logger.info("Training completed. Saving model...")

    # Save model
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
