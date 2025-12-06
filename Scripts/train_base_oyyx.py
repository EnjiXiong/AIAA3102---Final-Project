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
from typing import Dict
import subprocess
import threading
import psutil
import pynvml
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
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import time
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_cpu_ram():#获取当前进程的CPU内存使用情况，单位MB
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_ram():#获取当前进程的GPU内存使用情况，单位MB
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / 1024**2


class BenchmarkCallback(TrainerCallback):

    def __init__(self, interval_steps=50):
        """
        interval_steps: 每多少个 step 输出一次
        """
        self.interval_steps = interval_steps

    def on_step_end(self, args, state, control, **kwargs):

        # 每 interval_steps 输出一次
        if state.global_step % self.interval_steps == 0:

            cpu_ram = get_cpu_ram()
            gpu_ram = get_gpu_ram()

            print("\n========== TRAINING BENCHMARK ==========")
            print(f"Step: {state.global_step}")
            print(f"CPU RAM used: {cpu_ram:.1f} MB")
            print(f"GPU RAM used: {gpu_ram:.1f} MB")
            print("=========================================\n")


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


def read_jsonl(path: Path) -> list[Dict]:
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
    # 返回空字典，避免复杂的指标计算导致梯度问题
    # 使用 PerplexityCallback 来添加 perplexity 指标
    return {}

from transformers import TrainerCallback
import math

class PerplexityCallback(TrainerCallback):#计算 perplexity 并添加到training_logs中
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        # 不要 return metrics，Trainer 会自动使用修改后的 dict
        return control




def main():
    print('train begin')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs", help="Directory containing YAML configs")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")


    print("basic args loaded!")
    
    # 添加可自定义参数
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--metric_for_best_model", type=str, default=None, help="Override metric for best model")
    parser.add_argument("--base_dir", type=str, default=None)  # 这里就是可以覆盖的参数
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")

    
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    training_cfg = load_yaml(cfg_dir / "training_args.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    eval_cfg = load_yaml(cfg_dir / "eval_config.yaml")

    # base_dir = args.base_dir if args.base_dir is not None else ''

    print("Configs loaded!")

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
    if args.base_dir is not None:#覆盖 output_dir, 保证在任何路径下运行时输出都在 base_dir 下
        # print("Overriding output_dir with base_dir:", args.base_dir)
        # print(training_cfg.get("output_dir", "/Models/tinyllama_finetuned"))
        training_cfg["output_dir"] = os.path.join(args.base_dir, training_cfg.get("output_dir", "Models/tinyllama_finetuned"))
        # print("The output dir is overridden by base_dir:", training_cfg["output_dir"])
    
    # print("Training configs:", training_cfg)
        

    print("override configs loaded!")

    # 修复：确保 metric_for_best_model 是有效的
    valid_metrics = ['eval_loss', 'loss', 'epoch']
    metric_for_best_model = training_cfg.get("metric_for_best_model", "eval_loss")
    if metric_for_best_model not in valid_metrics:
        logger.warning(f"metric_for_best_model '{metric_for_best_model}' is not valid. Using 'eval_loss' instead.")
        metric_for_best_model = "eval_loss"
        training_cfg["metric_for_best_model"] = metric_for_best_model
    
    print("metric is valid!")

    # 根据 metric 设置 greater_is_better
    greater_is_better = training_cfg.get("greater_is_better", False)
    if metric_for_best_model == "eval_loss":
        greater_is_better = False
    elif metric_for_best_model in ["accuracy", "f1"]:
        greater_is_better = True

    print("greater is better set!")

    # Seed
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)

    print("seed set!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    print(f"Running on device: {device}")

    model_name_or_path = training_cfg.get("model_name_or_path", model_cfg.get("model_name_or_path"))
    if model_name_or_path is None:
        raise ValueError("Model name/path not specified in configs")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("tokenizer_name_or_path", model_name_or_path))
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token, setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    print("tokenizer loaded!")


    # Decide quantization / bitsandbytes config
    qlora_cfg = training_cfg.get("qlora", {})
    lora_cfg = training_cfg.get("lora", {})

    use_qlora = qlora_cfg.get("use_qlora", False)
    qlora_mode = qlora_cfg.get("mode", None)  # "4bit", "8bit", or None
    use_lora = lora_cfg.get("use_lora", False)

    bnb_config = None
    # QLoRA 配置
    if use_qlora:
        if qlora_mode == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
            )
            logger.info("Using 4-bit QLoRA.")
        elif qlora_mode == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit QLoRA.")
        else:
            raise ValueError("If use_qlora=true, qlora.mode must be '4bit' or '8bit'.")
        print(f"Using {qlora_mode} QLoRA bitsandbytes")

    # -------------------- Load model --------------------
# 推荐替换的加载代码
    try:
        if bnb_config is not None:
            # QLoRA path
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # LoRA-only or full-finetune -> DO NOT force torch_dtype=float16 here
            # Let Trainer autocast handle mixed precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=None,
                low_cpu_mem_usage=training_cfg.get("low_cpu_mem_usage", False),
                trust_remote_code=True,
            )
    except Exception as e:
        logger.warning("Failed to load with device_map=auto or quant config, trying fallback. Error: %s", e)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        # 把模型移动到 Trainer 将使用的设备（通常 trainer 在创建时会做）
        model.to(device)


    # -------------------- Prepare for k-bit training (QLoRA only) --------------------
    if use_qlora and bnb_config is not None:
        logger.info("Preparing model for k-bit training (QLoRA flow).")
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training!")

    # -------------------- Attach LoRA --------------------
    if use_lora:
        # LoRA-only 或 LoRA+QLoRA 都可以用
        lora_config_obj = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", None),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_obj)
        model.enable_input_require_grads()

        
        # 确保所有参数都有正确的梯度追踪
        model.train()

        # 检查可训练参数
        trainable_params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(n)
                p.data = p.data.float()  # 确保参数为 float32
        
        if len(trainable_params) == 0:
            raise RuntimeError(
                "LoRA adapter未成功插入，请检查 target_modules 是否正确匹配模型"
            )
        
        # print(f"LoRA adapters attached. Trainable params count: {len(trainable_params)}")
        # for param in trainable_params[:5]:  # 只打印前5个
        #     print(f"  - {param}")
        # if len(trainable_params) > 5:
        #     print(f"  ... and {len(trainable_params) - 5} more")

    print("LoRA/QLoRA config set!")

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

    print("datasets and data collator set!")

    # TrainingArguments
    output_dir = training_cfg.get("output_dir", "./tinyllama_finetuned")


    # print("output dir:", output_dir)
    
    # 修复：确保 max_steps 和 num_train_epochs 的默认值
    max_steps = training_cfg.get("max_steps", -1)
    if max_steps is None:
        max_steps = -1

    if use_qlora:
        # QLoRA: do NOT use fp16
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            num_train_epochs=training_cfg.get("num_train_epochs", 2),
            max_steps=max_steps,
            learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            logging_steps=training_cfg.get("logging_steps", 50),
            eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
            eval_steps=training_cfg.get("eval_steps", 200),
            eval_accumulation_steps=training_cfg.get("eval_accumulation_steps", 1),
            save_steps=training_cfg.get("save_steps", 200),
            save_total_limit=training_cfg.get("save_total_limit", 3),
            fp16=training_cfg.get("fp16", True),
            # 禁用 gradient_checkpointing（会导致梯度丢失）
            gradient_checkpointing=True,
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=metric_for_best_model,  # 使用修复后的值
            greater_is_better=greater_is_better,  # 根据指标类型设置
            push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
            report_to=training_cfg.get("report_to", "none"),
            remove_unused_columns=False,
            overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
            bf16=training_cfg.get("bf16", False),
            optim=training_cfg.get("optim", "paged_adamw_32bit"),
        )
    else:
        # LoRA-only or full finetune
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            num_train_epochs=training_cfg.get("num_train_epochs", 2),
            max_steps=max_steps,
            learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            logging_steps=training_cfg.get("logging_steps", 50),
            eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
            eval_steps=training_cfg.get("eval_steps", 200),
            eval_accumulation_steps=training_cfg.get("eval_accumulation_steps", 1),
            save_steps=training_cfg.get("save_steps", 200),
            save_total_limit=training_cfg.get("save_total_limit", 3),
            # 禁用 gradient_checkpointing（会导致梯度丢失）
            gradient_checkpointing=True,
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=metric_for_best_model,  # 使用修复后的值
            greater_is_better=greater_is_better,  # 根据指标类型设置
            push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
            report_to=training_cfg.get("report_to", "none"),
            remove_unused_columns=False,
            overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
            fp16=training_cfg.get("fp16", False),  # 允许 True
            bf16=training_cfg.get("bf16", False),
        )
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
    #     per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
    #     gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
    #     num_train_epochs=training_cfg.get("num_train_epochs", 2),
    #     max_steps=max_steps,
    #     learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
    #     weight_decay=training_cfg.get("weight_decay", 0.0),
    #     logging_steps=training_cfg.get("logging_steps", 50),
    #     eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
    #     eval_steps=training_cfg.get("eval_steps", 200),
    #     eval_accumulation_steps=training_cfg.get("eval_accumulation_steps", 1),
    #     save_steps=training_cfg.get("save_steps", 200),
    #     save_total_limit=training_cfg.get("save_total_limit", 3),
    #     fp16=training_cfg.get("fp16", True),
    #     # 禁用 gradient_checkpointing（会导致梯度丢失）
    #     gradient_checkpointing=True,
    #     load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
    #     metric_for_best_model=metric_for_best_model,  # 使用修复后的值
    #     greater_is_better=greater_is_better,  # 根据指标类型设置
    #     push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
    #     report_to=training_cfg.get("report_to", "none"),
    #     remove_unused_columns=False,
    #     overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
    # )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics  # 添加计算指标的函数
    )

    # 在训练前验证模型梯度状态
    grad_enabled_params = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters with requires_grad=True: {grad_enabled_params}")
    if grad_enabled_params == 0:
        raise RuntimeError("No parameters require gradients! Check your LoRA/model configuration.")

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_ds))
    logger.info("  Num valid examples = %d", len(valid_ds))
    logger.info("  Output dir = %s", output_dir)
    logger.info("  Metric for best model = %s", metric_for_best_model)

    trainer.add_callback(PerplexityCallback())
    trainer.add_callback(TensorBoardCallback(output_dir=output_dir))

    trainer.add_callback(BenchmarkCallback(interval_steps=1000))

    # ---------- Diagnostic helper: run one forward and inspect grad graph ----------

    def debug_forward_and_grad_check(model, train_dataset, tokenizer, batch_size=1, n_steps=1):
        model.train()  # ensure training mode

        # collect a small dataloader from train_dataset (works for datasets.Dataset or list)
        try:
            if hasattr(train_dataset, "select"):
                small_ds = train_dataset.select(range(min(len(train_dataset), batch_size * n_steps)))
                dl = DataLoader(small_ds, batch_size=batch_size, collate_fn=lambda x: data_collator(x), shuffle=False)
            else:
                dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        except Exception as e:
            # fallback: try simple DataLoader
            dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # find first trainable parameter and its device
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print("Trainable params count:", len(trainable_params))
        if len(trainable_params) == 0:
            print("ERROR: No trainable parameters found in model. LoRA not attached correctly or all params frozen.")
            return False

        first_param = trainable_params[0]
        device = first_param.device
        print("First trainable param device:", device)

        # helper to move inputs to the right device
        def move_batch_to_device(batch, device):
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                else:
                    out[k] = v
            return out

        # Run a couple of steps
        good = True
        for i, batch in enumerate(dl):
            if i >= n_steps:
                break
            batch = move_batch_to_device(batch, device)
            # ensure no accidental no_grad scope
            torch.set_grad_enabled(True)
            # Try forward (catch exceptions)
            try:
                outputs = model(**batch)
            except Exception as e:
                print("Forward failed with exception:", e)
                good = False
                break

            # try to obtain loss attribute (HF models usually return loss)
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                # try typical keys
                loss = None
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]

            print(f"Step {i}: loss tensor:", type(loss))
            if loss is None:
                print("No loss returned by model.forward(). Check model forward signature and data collator.")
                good = False
                break

            print("  loss.item() (if available):", None if (not hasattr(loss, "item")) else loss.item())
            print("  loss.requires_grad:", loss.requires_grad)
            print("  loss.grad_fn:", type(loss.grad_fn).__name__ if loss.grad_fn is not None else None)

            # Print a few trainable param devices and requires_grad
            sample_trainables = trainable_params[:6]
            for pidx, p in enumerate(sample_trainables):
                print(f"  param[{pidx}] device: {p.device}, requires_grad: {p.requires_grad}, shape: {tuple(p.shape)}")

            # Print input tensor devices/dtypes
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  input '{k}': device={v.device}, dtype={v.dtype}, shape={tuple(v.shape)}")

            if not loss.requires_grad or loss.grad_fn is None:
                print("\n=== LOSS has no grad_fn. Investigating common causes... ===")
                # check global grad status
                print("  torch.is_grad_enabled():", torch.is_grad_enabled())
                print("  Any param requires_grad?:", any(p.requires_grad for p in model.parameters()))
                print("  Number of params with requires_grad=True:", sum(1 for p in model.parameters() if p.requires_grad))
                print("  Are we accidentally in no_grad context? (try to enable and re-run one forward)")

                # As a pragmatic remedy, explicitly mark LoRA params as trainable (if present)
                corrected = False
                for n, p in model.named_parameters():
                    if "lora" in n and not p.requires_grad:
                        p.requires_grad = True
                        corrected = True
                        print(f"  - Fixed requires_grad for {n}")

                if corrected:
                    # re-run forward once after fix
                    torch.set_grad_enabled(True)
                    batch2 = move_batch_to_device(batch, first_param.device)
                    outputs2 = model(**batch2)
                    loss2 = outputs2.loss if hasattr(outputs2, "loss") else (outputs2["loss"] if isinstance(outputs2, dict) and "loss" in outputs2 else None)
                    print("After forcing LoRA params requires_grad=True -> loss.requires_grad:", loss2.requires_grad if loss2 is not None else None)
                    if loss2 is not None and loss2.requires_grad:
                        print("Fix worked: loss now has grad_fn.")
                        return True
                    else:
                        print("Fix did not work. Still no grad_fn for loss.")
                        good = False
                        break
                else:
                    print("No LoRA params were non-trainable to fix. Need deeper debugging (device mismatch or forward uses no-trainable path).")
                    good = False
                    break
            else:
                print("Loss OK (has grad_fn). You should be able to backward().")
                return True

        return good

    # Run the check
    ok = debug_forward_and_grad_check(model, train_ds, tokenizer, batch_size=1, n_steps=1)
    if not ok:
        raise RuntimeError("Sanity check failed: forward produced a loss without grad_fn. See printed diagnostics above.")
    else:
        print("Sanity check passed: loss has grad_fn and model ready for training.")

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