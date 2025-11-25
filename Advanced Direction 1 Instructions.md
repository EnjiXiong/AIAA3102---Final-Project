下面我为你整理 **Advanced Direction 1：实体扰动 / prompt perturbation + robustness evaluation** 的 **完整任务清单（deliverables）**。
如果你全部做到，你这一项 advanced direction 绝对拿满分，而且非常“科研范”。

---

# ✅ **Advanced Direction 1 — 你需要达成的全部目标（可直接写进 Final Project Proposal & Report）**

以下是一个**完整的任务体系**，我分成：**(A) 需要实现的模块 (B) 需要生成的数据 (C) 需要做的实验 (D) 需要写入报告的结果**。

---

# ⭐ A. **实现任务（必须达成）**

## **1. 设计 4–6 种 Prompt Perturbation（实体扰动）方法**

你至少要实现：

### ✔ 1) 同义词替换（Synonym Substitution）

例：

```
sad → depressed → low → upset → devastated
friend → partner → cousin → coworker
```

### ✔ 2) 语序变化（Reordering）

```
"I'm sad because my friend left."
"My friend left and it makes me sad."
```

### ✔ 3) 拼写噪声 / 错别字（Typos）

```
sad → saad → sd
```

### ✔ 4) 实体替换（Entity Swap）

```
friend → father → roommate
```

### ✔ 5) 情境变化（Contextual Perturbation）

```
“My friend left the country.”
“My friend stopped talking to me.”
```

### ✔ 6) 语言风格变化（Dialect / Formality）

（可选）

```
"I'm feeling down." → "I am experiencing emotional distress."
```

---

## **2. 实现一个 “Prompt Perturbation Generator” Python 模块（强烈推荐）**

你需要写一个函数：

```python
def generate_perturbations(prompt: str, n=5):
    # 返回多个扰动版本
    return [perturbed_prompt_1, perturbed_prompt_2, ..., perturbed_prompt_n]
```

你可以存在：

```
/Final_Project/Robustness/perturbation.py
```

---

## **3. 实现 baseline 模型和 finetuned 模型 的 inference pipeline**

你需要 2 个模型输出：

* `baseline_output = baseline_model(prompt)`
* `finetuned_output = finetuned_model(prompt)`
* 对 perturbation 版本也一样：

```
baseline_model(p1), baseline_model(p2), ...
finetuned_model(p1), finetuned_model(p2), ...
```

---

## **4. 实现 “Robustness Scoring” 模块**

你需要一个函数：

```python
def evaluate_response_quality(prompt, output):
    # 返回 {fluency, relevance, empathy, consistency} 评分
```

可以用 3 种方式实现：

### ✔ 方式 A：GPT-4o judge（最强，最加分）

输入 prompt + model output，让 GPT 给 1–5 分。

### ✔ 方式 B：Embedding 相似度（不调用 GPT）

用 SentenceTransformer 计算：

* model output ↔ reference response
* prompt ↔ output coherence

### ✔ 方式 C：人工评分

如果不能用 GPT API。

教师评分最喜欢 **方式 A** 和 **方式 B**。

---

## **5. 计算 Robustness Score（你必须要有一个公式）**

你可以设计：

[
R = \frac{1}{N} \sum_{i=1}^{N} \text{QualityScore}(perturb_i)
]

或更高级：

[
R = Consistency + Relevance + Empathy
]

（你可以在报告中详细解释这个公式）

---

# ⭐ B. **生成的数据（必须输出）**

你最终至少需要生成如下 JSON 文件：

### **1. perturbations.json**

保存每条原始 prompt 的所有扰动版本。

### **2. model_outputs.json**

结构示例：

```json
{
  "original_prompt": "...",
  "perturbations": [
    {
      "p": "perturbation1",
      "baseline_output": "...",
      "finetuned_output": "...",
      "baseline_score": {...},
      "finetuned_score": {...}
    }
  ]
}
```

### **3. robustness_results.json**

包括每种扰动方法的：

* 平均分
* 差值（finetuned gain）

为你后续绘图准备数据。

---

# ⭐ C. **需要做的实验（必须做）**

## **1. 对 50 条 validation prompts 做扰动（不能太少）**

50 是合适规模（这样统计显著）。

## **2. 让两个模型分别生成回复**

* TinyLlama（未微调）
* TinyLlama-LoRA（你训练的）

## **3. 统计 robustness gain：微调后提升多少？**

例如：

| Perturbation  | Baseline R | Fine-tuned R | Δ        |
| ------------- | ---------- | ------------ | -------- |
| Typos         | 2.1        | 3.4          | **+1.3** |
| Synonym       | 2.7        | 3.8          | +1.1     |
| Entity change | 2.5        | 4.0          | **+1.5** |

这样你的实验就很“论文范”，教师超喜欢。

---

# ⭐ D. **最后需要写入报告的内容（必须写）**

下面这些是 final project 中必须出现的：

---

## 📌 **1. 方法框图（Method Overview）**

你需要一个类似下图的流程：

```
Prompt → Perturbation Generator → Multiple Prompts → Model → Scorer → Robustness Score
```

我可以帮你画成 LaTeX TikZ / PPT 用图。

---

## 📌 **2. 描述你的扰动算法（要写得像论文）**

包括：

* 你选择了哪几种 perturbation
* 为什么（参考 NLP robustness 文献）
* 效果预期

---

## 📌 **3. 描述评分方法（GPT judge / embedding）**

* 你的 scoring rubric（例如 Empathy / Relevance / Coherence）
* 量化方式（1–5 分）
* 多模型对比方式

---

## 📌 **4. 实验结果（至少一个图 + 一个表）**

### 图建议：

* 每种扰动类型下的 “Baseline vs Finetuned” 柱状图
* 或者对各类型扰动的 gain 曲线

### 表建议：

| Perturbation | Baseline | Finetuned | Δ (%) |
| ------------ | -------- | --------- | ----- |

---

## 📌 **5. 讨论（必写）**

你必须回答：

* 为什么微调后的模型更加鲁棒？
* 哪些 perturbation 改进最大？
* 哪些 perturbation 模型仍然表现不好？
* 对 real-world mental health counseling 的意义是什么？

---

# 🎯 **最终交付物 checklist（你必须完成的）**

| Category | Deliverable                   | 必须? |
| -------- | ----------------------------- | --- |
| 算法       | perturbation generator        | ✔   |
| 推理       | baseline vs finetuned 双模型输出   | ✔   |
| 评分       | GPT judge / embedding scoring | ✔   |
| 实验       | 至少 5 类 perturbation           | ✔   |
| 数据       | perturbations.json            | ✔   |
| 数据       | model_outputs.json            | ✔   |
| 数据       | robustness_results.json       | ✔   |
| 分析       | robustness gain 表格 & 图        | ✔   |
| 报告       | 方法写作 + 结果讨论                   | ✔   |

这个 Checklist 可以直接放进你的 Project Proposal。

---

# 如果你愿意，我可以下一步帮你：

### ✔ 生成 `perturbation.py` (可直接运行)

### ✔ 把 baseline inference / finetuned inference 的对比代码写到你的训练 notebook 里

### ✔ 设计你的 GPT-judge 评分 prompt

### ✔ 给你画 robustness 实验结果图的模板

你希望我现在先帮你写 **扰动生成器（perturbation generator）** 吗？
