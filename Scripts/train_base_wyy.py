#!/usr/bin/env python3
# scripts/train_base.py
"""
Enhanced Train script for tinyllama / causal LM using LoRA or QLoRA (4-bit).
支持更多自定义参数和详细训练监控。
"""

import argparse
import os
import logging
from pathlib import Path
import json
import math
import random
from typing import Dict, List, Optional

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
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class EnhancedTensorBoardCallback(TrainerCallback):
    """
    增强的TensorBoard回调，支持更多训练指标和实时进度显示
    """
    
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / "tb_runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        tb_subdir = time.strftime("%Y%m%d-%H%M%S")
        self.logdir = str(self.run_dir / tb_subdir)
        self.writer = SummaryWriter(log_dir=self.logdir)
        
        self.records = []
        self.last_log_step = None
        self.start_time = time.time()
        
        # 训练进度跟踪
        self.progress_bar = None
        self.current_epoch = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时初始化进度条"""
        total_steps = state.max_steps if state.max_steps else args.num_train_epochs * state.num_train_examples // args.train_batch_size
        self.progress_bar = tqdm(total=total_steps, desc="Training", unit="step")
        logger.info(f"🚀 开始训练，总步数: {total_steps}")
        
    def on_step_end(self, args, state, control, **kwargs):
        """每一步结束时更新进度条"""
        if self.progress_bar:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                'loss': f"{state.log_history[-1].get('loss', 0):.4f}" if state.log_history else 'N/A',
                'epoch': state.epoch
            })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个epoch开始时更新"""
        self.current_epoch = state.epoch
        logger.info(f"📅 开始第 {state.epoch:.1f} 个epoch")
        
    def _record_entry(self, step: int, entry: Dict):
        entry_with_step = {
            "step": int(step), 
            "timestamp": int(time.time()),
            "wall_time": time.time() - self.start_time
        }
        entry_with_step.update(entry)
        self.records.append(entry_with_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        step = int(state.global_step)
        
        # 记录训练指标到TensorBoard
        metrics_to_log = {
            'loss': 'train/loss',
            'learning_rate': 'train/learning_rate', 
            'epoch': 'train/epoch',
            'grad_norm': 'train/grad_norm'
        }
        
        for log_key, tb_key in metrics_to_log.items():
            if log_key in logs:
                try:
                    value = logs[log_key]
                    if isinstance(value, list):
                        value = value[0]
                    self.writer.add_scalar(tb_key, float(value), step)
                except Exception as e:
                    logger.debug(f"Failed to log {log_key}: {e}")

        # 记录到JSON
        entry = {}
        for key in metrics_to_log.keys():
            if key in logs:
                try:
                    entry[key] = float(logs[key]) if not isinstance(logs[key], list) else float(logs[key][0])
                except Exception:
                    entry[key] = logs[key]
                    
        if entry:
            self._record_entry(step, entry)
            self.last_log_step = step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
            
        step = int(state.global_step)
        
        # 记录评估指标
        for k, v in metrics.items():
            try:
                val = float(v)
                self.writer.add_scalar(f"eval/{k}", val, step)
            except Exception:
                continue
                
        # 记录到JSON
        entry = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if entry:
            self._record_entry(step, {"eval": entry})
            
        logger.info(f"📊 评估结果 (step {step}): {metrics}")

    def on_train_end(self, args, state, control, **kwargs):
        # 关闭进度条
        if self.progress_bar:
            self.progress_bar.close()
            
        # 保存训练记录
        try:
            self.writer.flush()
            self.writer.close()
            
            # 保存详细的训练日志
            out_path = self.output_dir / "training_logs_detailed.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
                
            # 生成训练曲线图
            self._plot_training_curves()
            
            logger.info(f"📈 训练完成！日志保存至: {out_path}")
            logger.info(f"📊 TensorBoard日志在: {self.logdir}")
            
        except Exception as e:
            logger.error(f"保存训练日志失败: {e}")

    def _plot_training_curves(self):
        """生成训练损失曲线图"""
        try:
            if not self.records:
                return
                
            # 提取训练损失
            train_steps = []
            train_losses = []
            
            for record in self.records:
                if 'loss' in record:
                    train_steps.append(record['step'])
                    train_losses.append(record['loss'])
            
            if train_steps:
                plt.figure(figsize=(10, 6))
                plt.plot(train_steps, train_losses, 'b-', alpha=0.7, label='Training Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title('Training Loss Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图片
                plot_path = self.output_dir / "training_loss_curve.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"📸 训练曲线图保存至: {plot_path}")
                
        except Exception as e:
            logger.warning(f"生成训练曲线图失败: {e}")

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(path: Path) -> List[Dict]:
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    return objs

def make_dataset_from_jsonl(jsonl_path: Path, tokenizer, max_length: int, max_samples: Optional[int] = None):
    """
    创建训练数据集，支持最大样本数限制
    """
    from datasets import Dataset

    raw = read_jsonl(jsonl_path)
    
    # 限制样本数量
    if max_samples and len(raw) > max_samples:
        logger.info(f"📝 限制数据集从 {len(raw)} 到 {max_samples} 个样本")
        raw = raw[:max_samples]
    
    items = []
    for item in raw:
        p = item.get("prompt", "").strip()
        r = item.get("response", "").strip()
        items.append({"prompt": p, "response": r})

    # Tokenize
    all_prompts = [it["prompt"] for it in items]
    all_resps = [it["response"] for it in items]

    enc_prompts = tokenizer(all_prompts, add_special_tokens=False)["input_ids"]
    enc_resps = tokenizer(all_resps, add_special_tokens=False)["input_ids"]

    input_ids_list = []
    labels_list = []
    attention_masks = []

    for p_ids, r_ids in zip(enc_prompts, enc_resps):
        full = p_ids + r_ids

        if len(full) > max_length:
            full = full[-max_length:]
            if len(r_ids) >= max_length:
                resp_start = 0
            else:
                resp_start = max(0, len(full) - len(r_ids))
        else:
            resp_start = len(p_ids)

        labels = [-100] * len(full)
        for i in range(resp_start, len(full)):
            labels[i] = full[i]

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
    ds = ds.with_format(type="torch")
    
    logger.info(f"📊 数据集创建完成: {len(ds)} 个样本")
    return ds

def compute_metrics(eval_pred):
    """
    计算评估指标
    """
    try:
        predictions, labels = eval_pred
        if predictions is None:
            return {}
            
        preds = np.array(predictions)
        lab = np.array(labels)

        if preds.ndim != 3:
            return {}

        logits = torch.from_numpy(preds)
        labels_t = torch.from_numpy(lab).long()

        log_probs = F.log_softmax(logits, dim=-1)
        B, S = labels_t.shape
        mask = labels_t != -100
        
        if mask.sum().item() == 0:
            return {}

        labels_exp = labels_t.unsqueeze(-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels_exp).squeeze(-1)
        nll = -token_log_probs[mask]
        token_loss = nll.mean().item()
        perplexity = float(torch.exp(torch.tensor(token_loss)).item())

        return {"token_loss": float(token_loss), "perplexity": perplexity}
    except Exception as e:
        return {}

def main():
    parser = argparse.ArgumentParser(description="增强的LLM微调脚本")
    
    # 基础参数
    parser.add_argument("--config_dir", type=str, default="configs", help="配置文件目录")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--valid_file", type=str, required=True, help="验证数据文件")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="覆盖输出目录")
    parser.add_argument("--push_to_hub", action="store_true", help="推送到HuggingFace Hub")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--max_steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--metric_for_best_model", type=str, default=None, help="最佳模型指标")
    
    # 新增自定义参数
    parser.add_argument("--max_eval_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--use_lora", type=bool, default=None, help="是否使用LoRA")
    parser.add_argument("--use_qlora", type=bool, default=None, help="是否使用QLoRA")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()

    # 加载配置
    cfg_dir = Path(args.config_dir)
    training_cfg = load_yaml(cfg_dir / "training_args.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    eval_cfg = load_yaml(cfg_dir / "eval_config.yaml")

    # 用命令行参数覆盖配置
    config_overrides = {
        'num_train_epochs': args.num_train_epochs,
        'learning_rate': args.learning_rate,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_steps': args.max_steps,
        'metric_for_best_model': args.metric_for_best_model,
        'seed': args.seed,
    }
    
    for key, value in config_overrides.items():
        if value is not None:
            training_cfg[key] = value

    # LoRA参数覆盖
    if args.use_lora is not None:
        training_cfg["lora"]["use_lora"] = args.use_lora
    if args.use_qlora is not None:
        training_cfg["qlora"]["use_qlora"] = args.use_qlora
    if args.lora_rank is not None:
        training_cfg["lora"]["r"] = args.lora_rank
    if args.lora_alpha is not None:
        training_cfg["lora"]["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        training_cfg["lora"]["lora_dropout"] = args.lora_dropout

    # 设置随机种子
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    logger.info(f"🎲 设置随机种子: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  运行设备: {device}")

    # 加载模型和tokenizer
    model_name_or_path = training_cfg.get("model_name_or_path", model_cfg.get("model_name_or_path"))
    if model_name_or_path is None:
        raise ValueError("❌ 未在配置中指定模型名称/路径")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("tokenizer_name_or_path", model_name_or_path))
    if tokenizer.pad_token is None:
        logger.info("🔧 Tokenizer没有pad_token，设置pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置
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
        logger.info(f"🔧 使用4-bit QLoRA配置")

    # 加载模型
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
        logger.warning(f"❌ 使用device_map=auto加载失败，尝试CPU加载: {e}")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        model.to(device)

    # QLoRA准备
    if use_qlora and bnb_config is not None:
        logger.info("🔧 准备k-bit训练 (QLoRA)")
        model = prepare_model_for_kbit_training(model)

    # LoRA配置
    use_lora = training_cfg.get("lora", {}).get("use_lora", True)
    if use_lora:
        lora_cfg = training_cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"🎯 LoRA适配器已附加到模型 (rank={lora_config.r}, alpha={lora_config.lora_alpha})")

    # 准备数据集
    max_input_length = model_cfg.get("max_input_length", 512)
    
    # 使用自定义的最大样本数
    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples or 50  # 默认50个评估样本
    
    train_ds = make_dataset_from_jsonl(
        Path(args.train_file), 
        tokenizer, 
        max_input_length, 
        max_samples=max_train_samples
    )
    valid_ds = make_dataset_from_jsonl(
        Path(args.valid_file), 
        tokenizer, 
        max_input_length, 
        max_samples=max_eval_samples
    )
    
    logger.info(f"📊 训练集: {len(train_ds)} 个样本")
    logger.info(f"📊 验证集: {len(valid_ds)} 个样本")

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, 
        pad_to_multiple_of=8
    )

    # 训练参数
    output_dir = training_cfg.get("output_dir", "models/finetuned_model")
    
    # 修复训练参数
    max_steps = training_cfg.get("max_steps", -1)
    if max_steps is None:
        max_steps = -1

    # 设置最佳模型指标
    metric_for_best_model = training_cfg.get("metric_for_best_model", "eval_loss")
    greater_is_better = training_cfg.get("greater_is_better", False)
    if metric_for_best_model == "eval_loss":
        greater_is_better = False
    elif metric_for_best_model in ["accuracy", "f1", "perplexity"]:
        greater_is_better = True

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
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        push_to_hub=training_cfg.get("push_to_hub", False) or args.push_to_hub,
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        overwrite_output_dir=args.overwrite_output_dir or training_cfg.get("overwrite_output_dir", False),
        logging_dir=str(Path(output_dir) / "tensorboard"),
        # 新增参数以改善训练体验
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        warmup_steps=training_cfg.get("warmup_steps", 100),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EnhancedTensorBoardCallback(output_dir=output_dir)],
    )

    # 打印训练信息
    logger.info("🚀 开始训练")
    logger.info(f"  📝 训练样本数: {len(train_ds)}")
    logger.info(f"  📝 验证样本数: {len(valid_ds)}")
    logger.info(f"  📁 输出目录: {output_dir}")
    logger.info(f"  📊 最佳模型指标: {metric_for_best_model}")
    logger.info(f"  🔧 LoRA: {use_lora}")
    logger.info(f"  🔧 QLoRA: {use_qlora}")
    logger.info(f"  ⚙️  学习率: {training_args.learning_rate}")
    logger.info(f"  ⚙️  批次大小: {training_args.per_device_train_batch_size}")
    logger.info(f"  ⚙️  梯度累积: {training_args.gradient_accumulation_steps}")

    # 开始训练
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"✅ 训练完成！耗时: {training_time/60:.2f} 分钟")

    # 保存模型
    logger.info("💾 保存模型和tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练配置
    config_save_path = Path(output_dir) / "training_config.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_cfg': training_cfg,
            'model_cfg': model_cfg,
            'eval_cfg': eval_cfg,
            'training_time_minutes': training_time/60,
            'final_metrics': trainer.state.log_history[-1] if trainer.state.log_history else {}
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 模型和配置保存到: {output_dir}")

    # 可选：推送到Hub
    if training_args.push_to_hub:
        try:
            logger.info("🌐 推送模型到HuggingFace Hub...")
            trainer.push_to_hub()
            logger.info("✅ 推送完成")
        except Exception as e:
            logger.warning(f"❌ 推送到Hub失败: {e}")

    logger.info("🎉 所有任务完成！")


if __name__ == "__main__":
    main()
