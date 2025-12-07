# AIAA3102 Final Project — TinyLlama Fine-Tuning with LoRA/QLoRA

This repository contains the full implementation of Our group's **AIAA3102 Final Project**, where we do PEFT on **TinyLlama-1.1B** using **LoRA / QLoRA** on specialized datasets to enhance domain-specific reasoning ability.
The project includes training scripts, dataset preprocessing, evaluation metrics, and exploratory *Advanced Directions* beyond the requirements.

You can download complete codes and checkpoints in [Hugging Face](https://huggingface.co/datasets/EnjiXiong/AIAA3102_Final_Project_A/tree/main). The content in github just provide an accessible and executable version.

---

## 🚀 Project Overview

### 🔹 Goal

Improve TinyLlama’s capability on specific domains (e.g., **mental health counseling**, **code generation**) through:

* Lightweight **PEFT-based fine-tuning**
* Dataset preprocessing in JSONL prompt–response format
* Controlled evaluation (before vs. after fine-tuning)
* Logging & visualization via TensorBoard
* Advanced experiments exploring generalization, scaling laws, and LoRA design choices

The project is fully runnable on Google Colab with a **T4 GPU (16GB)**.

---

## 📂 Directory Structure

```
AIAA3102/Final_Project
│
├── .ipynb_checkpoints/
│
├── Configs/
│   ├── eval_config.yaml
│   ├── model_config.yaml
│   ├── training_args.yaml
│   ├── training_args0.yaml # previous arguments configurations
│   └── .ipynb_checkpoints/
│
├── Data/
│   ├── con_train_debug.jsonl
│   ├── con_train.jsonl
│   ├── con_unknown_test.jsonl
│   ├── con_valid_debug.jsonl
│   ├── con_valid.jsonl
|   |--------------------------------- Two different dataset
│   ├── train_debug.jsonl
│   ├── train.jsonl
│   ├── unknown_test.jsonl
│   └── valid.jsonl
│
├── Deliverables/ # Except for code, other materials such as README.md, Demo, and PPT
│
├── Models/
│   └── tinyllama_ai_finetuned/
│       ├── checkpoint-1600/
│       ├── checkpoint-1800/
│       ├── checkpoint-1875/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── tokenizer.model
│
├── Results/
│   ├── test_results.json
│   ├── test_results_scores.json
│   └── test_results_rouge.json
│
├── Scripts/
│   ├── train_base.py
│   ├── train_base0.py # Previous base code
│   └── .ipynb_checkpoints/
│
├── AIAA3102_FinalProject_counseling_dataset.ipynb # main ipynb, the others are our attempts and worth remembering 
├── AIAA3102_FinalProject_wyy_01.ipynb
├── AIAA3102_FinalProject_wyy_02.ipynb
├── AIAA3102_FinalProject_Awareness_1on1r.ipynb
└── File_creator.ipynb # Create jsonl and py files directly in colab instead of uploading
```

---

## 📘 Dataset

The project supports **any prompt–response dataset**, including:

### Currently used dataset

#### 🧠 **Amod/mental_health_counseling_conversations**

* ~1,600 conversation samples
* Focused on empathetic and supportive language
* Good for improving counseling-style reasoning

### Previously tested

#### 💻 **HuggingFaceH4/CodeAlpaca-20K**

* Code-generation and instruction-following dataset
* Helps model improve structured reasoning and code writing

All datasets are automatically transformed to:

```json
{"prompt": "...", "response": "..."}
```

and masked with `-100` for causal LM supervised fine-tuning.

---

## ⚙️ How to Run Training

### **1. Prepare configs & data**

Ensure your folder contains:

```
Configs/*.yaml
Data/train.jsonl
Data/valid.jsonl
```

### **2. Run training script**

```bash
python Scripts/train_base.py \
    --config_dir Configs \
    --train_file Data/train.jsonl \
    --valid_file Data/valid.jsonl
```

### **3. Optional overrides (command line)**

```bash
--num_train_epochs 5
--learning_rate 2e-4
--per_device_train_batch_size 4
--metric_for_best_model eval_loss
```

### **4. Training with GUI (optional)**

You can either choose to use CLI for training, or GUI for training. To be noticed, the models you tuned in training GUI will be saved to the `/content/drive/MyDrive/AIAA3102/Final_Project/Models` directory by default, if you want to do assessment on them, please use assessment GUI, in which you can select fine-tuned models in `Models` directory.

```
%cd /content/drive/MyDrive/AIAA3102/Final_Project/Modules
!python training_gui.py
```

---

## 📊 Evaluation & Metrics

The project logs:

* **Training loss**
* **Validation loss**
* **Perplexity**
* **Learning rate**
* **Token/s throughput**
* **Training time**
* **LoRA parameters**

All results are exported to:

```
Results/test_results.json
```

TensorBoard logging is enabled:

```bash
%load_ext tensorboard
%tensorboard --logdir runs

```

## 🧪 Analysis of Overfitting Control

To investigate potential overfitting during training, the Psyc dataset was used to evaluate how different optimization hyperparameters influence model stability. Although the training loss and perplexity curves converge around step ~1000, signs of overfitting are still observed.

### Configuration Setup

Five configurations were tested, varying learning rate and weight decay:

| Configuration | Learning Rate | Weight Decay |
|--------------|----------------|--------------|
| LR-1         | 2e-5           | 0            |
| LR-2         | 1e-5           | 0            |
| LR-3         | 5e-6           | 0            |
| WD-1         | 2e-5           | 0.01         |
| WD-2         | 2e-5           | 0.1          |

### Purpose of This Direction

This direction aims to understand how learning rate and weight decay affect overfitting, convergence behavior, and training smoothness. The comparison provides insights into how small models and counseling-oriented data respond to different regularization levels.

#### **How to Run**

After each training run, the program automatically saves a `training_log.json` file inside the `tinyllama_ai_finetuned` directory.  
This file contains all metrics required for overfitting analysis, including perplexity, training loss, and evaluation loss.

To visualize these metrics, follow the same procedure described in **Advanced Direction 3 – How to Run**, where the plotting scripts can load `training_log.json` and generate the corresponding convergence and overfitting figures.


---

## 🔬 **Advanced Directions**

This project extends beyond basic LoRA/QLoRA supervised fine-tuning by exploring three advanced research directions, focusing on robustness, safety, and intrinsic model improvement. These experiments aim to evaluate whether the model truly *learns counseling-style reasoning* rather than merely memorizing training data.

---

### **1. Generalization & Robustness Evaluation**

Although the fine-tuned model performs well on in-distribution counseling conversations, it is unclear whether it has learned *general principles* of empathetic counseling. To assess this, we introduce systematic perturbations to the prompts and evaluate model robustness.

#### **Method**

For each original counseling prompt, generate **3–6 perturbed variants** using controlled transformations:

* **Synonym substitution**
  *“sad” → “depressed” → “upset” → “feeling down”*
* **Syntax reordering**
  Rephrase while keeping meaning intact
* **Mild typos or noise**
  *“I feel horible”* → *“I feel horrible”*
* **Role variations**
  *“my friend left me”* → *“my partner / cousin left me”*
* **Cultural variants**
  *American English ↔ British English*

Each perturbed prompt is fed to both:

* **Baseline TinyLlama**
* **Fine-tuned TinyLlama + LoRA**

#### **Evaluation**

* **Robustness Score**
  Using GPT-as-a-judge to score:

  * Consistency
  * Stability
  * Empathy quality
  * Instruction alignment
* **Visualization**
  Plot model performance across perturbation types (radar charts / bar plots)
* **Qualitative comparison**
  Inspect whether fine-tuned model remains stable under distribution shifts.

This direction demonstrates whether the model learned *behavioral patterns* instead of memorizing examples.

---

### **2. Special Cases Detection**

Counseling conversations often involve high-risk topics such as self-harm, suicide ideation, or violence. A responsible counseling model must **refuse unsafe queries** and direct users to seek professional help.

#### **2.1 Safety & Legality Check**

The safety and legality check feature blocks requests containing dangerous or illegal content using precise word-boundary regex matching. The system identifies exact matches of harmful terms while avoiding false positives from substrings (e.g., detecting "suicide" but not "presuicide").

##### **Implementation Details:**

* Implementation Details:
  * **Dangerous content dictionary:** 21 terms covering self-harm and suicide ideation
  * **Illegal activities dictionary:** 24 terms related to weapons, drugs, and criminal behavior
* When triggered, the system responds with pre-defined safety messages appropriate to the violation type
* All matching is case-insensitive and uses word boundaries (`\b`) to ensure precise matching

##### **Example blocked inputs:**

* "I want to kill myself because nothing matters anymore."
* "How to make a bomb at home"

##### **Example response for dangerous content:**

```
⚠️ SAFETY BLOCK (Dangerous Content): I hear you're going through a difficult time, and I truly care about your well-being. Please reach out to a mental health professional or a crisis hotline immediately. You're not alone, and help is available. In the US, you can call or text 988 for the Suicide & Crisis Lifeline.
```


##### **2.2 Out-of-Domain Detection**

The out-of-domain detection mechanism ensures the model remains focused on psychological counseling by implementing context-aware filtering. Unlike simple keyword blocking, this approach first checks for counseling-related contextual indicators before evaluating domain-relevant keywords.

##### **Implementation Details:**

* **Two-stage decision process:**
  1. Stage 1: Check for strong counseling context using 48 psychological indicators (e.g., "anxiety," "stress," "sleep problems")
  2. Stage 2: Only if no counseling context is found, check for 32 technical terms spanning mathematics, programming, medicine, and legal topics
* Contextual intelligence allows legitimate discussions where technical subjects appear in psychological contexts
* Uses the same word-boundary regex matching for accurate detection

##### **Example allowed inputs (with context):**

* "I'm anxious about my upcoming math exam and it's affecting my sleep."
* "My stress about programming is affecting my relationships."

##### **Example blocked inputs (without context):**

* "How to solve this calculus problem"
* "Write me a Python program to calculate Fibonacci numbers"

##### **Example response for out-of-domain content:**

```
⚠️ DOMAIN RESTRICTION: I notice your question relates to math, programming, and more, which is outside my area of expertise as a counseling-focused AI assistant. I'm designed to help with mental health, emotional well-being, and personal challenges. For questions about specific technical topics, I'd recommend consulting with a specialist in that field.
```

#### **How to Run**

1. Launch the assessment interface:
```
%cd /content/drive/MyDrive/AIAA3102/Final_Project/Modules
!python assessment_gui.py
```
2. Using the interface:
    * Step 1: Select and load a trained model from the dropdown menu
    * Step 2: For validation, set the number of samples and click "Run Validation"
    * Step 3: For interactive chat, enter your prompt in the input box and click "Generate Response"
    * Step 4: Adjust temperature and top-p parameters to control response randomness
3. Testing safety features:
    * Try entering prompts containing dangerous keywords (e.g., "suicide," "bomb")
    * Try entering technical questions outside the counseling domain (e.g., "how to code in Python")
    * Try entering prompts with technical keywords in counseling context (e.g., "I'm stressed about my math exam")
4. Customization:
    * To modify blocked keywords, edit the `DANGEROUS_WORDS`, `ILLEGAL_WORDS`, `OOD_KEYWORDS`, and `STRONG_COUNSELING_INDICATORS` lists in `assessment_gui.py`
    * To change response messages, modify the corresponding message templates in the `check_dangerous_content` and `check_out_of_domain` functions

> To be noticed, if you use GUI to do the assessment, you can only select the models in your `/content/drive/MyDrive/AIAA3102/Final_Project/Models` directory

This module ensures that the fine-tuned model operates within safe and professional boundaries while maintaining flexibility for legitimate counseling discussions that may include technical terminology.

---

### **3. Experimental Directions**

This section introduces the three experimental directions explored in this study.  
For each direction, we describe the purpose of the investigation and the configurations used for comparison.


#### **3.1 Influence of Data Characteristics and LoRA Parameters**

**Purpose:**  
To examine how dataset characteristics (code vs. counseling) and different LoRA parameter configurations affect model behavior.  
All experiments were performed under identical conditions (same data volume, batch size, and number of epochs).

**What was done:**  
We trained models on two datasets:
- **Code dataset**
- **Counseling dataset**

Using three LoRA configurations:

| Configuration | r | Alpha | Dropout | Target Modules |
|--------------|---|--------|---------|----------------|
| **A (Small)**  | 4 | 16 | 0.1 | q, v |
| **B (Medium)** | 8 | 32 | 0.1 | q, k, v, o |
| **C (Strong)** | 16 | 64 | 0.2 | q, k, v, o |

**Compared aspects:**  
- LOUGE-L score  

#### **3.2 Impact of LoRA Configuration on Model Scores**

**Purpose:**  
To isolate and analyze the effect of LoRA architectural choices on model quality.

**What was done:**  
We varied LoRA parameters across Config A, Config B, Config C, log their quality score and visualization

**Compared aspects:**  
- Quality score variations across different LoRA structures

#### **3.3 Comparison of Full-Parameter, LoRA, and QLoRA Training**

**Purpose:**  
To evaluate the efficiency and resource requirements of different fine-tuning strategies.

**What was done:**  
We compared four training paradigms:
- **Full-parameter fine-tuning**
- **LoRA**
- **4-bit QLoRA**
- **8-bit QLoRA**

**Compared aspects:**  
- CPU memory usage  
- GPU memory usage  
- Model size  
- Training speed  

#### **How to Run**

1. **Train the model**
   - In the main notebook, run the training section.
   - This will produce a folder named `tinyllama_ai_finetuned`, which contains:
     - Model checkpoints
     - `training_log.json`
     - Other training-related files

2. **Run evaluation**
   - Execute the evaluation section in the notebook to generate:
     - `test_result.json`
   - Then run:
     - The ROUGE-L scoring module
     - The quality-score evaluation module(dataset-dependent)
   - These will generate:
     - `test_result_rouge.json`
     - `test_result_score.json`

3. **Organize evaluation outputs**
   - Move the following three files into your `tinyllama_ai_finetuned` folder:
     - `test_result.json`
     - `test_result_rouge.json`
     - `test_result_score.json`

4. **Download visualization toolkit**
   - Download the zip from:  
     **https://huggingface.co/datasets/EnjiXiong/AIAA3102_Final_Project_A/tree/main**
   - Extract the downloaded folder.z`
   - Copy your `tinyllama_ai_finetuned` folder into the \Final_Project\Models\Advanced folder.

5. **Visualize the results**
   - Use the Python scripts provided inside the downloaded folder to generate visualizations.
   - Specific usage instructions can be found in the folder’s included `README.md`.



---

## 🧩 Future Work

* Add reward modeling (RM) and DPO training
* Add multi-turn conversation support
* Explore mixture-of-LoRA adapters

---

## Contributions 

**Enji Xiong**:
1. Whole file structure creator and maintainer, including writing scripts, basic main.ipynb, data processing, training\_args.yaml
2. Add baseline comparison in evaluation
3. Design all Advanced Contributions
4. Complete Advanced direction 2: Perturbation

**Yuxuan Ouyang**:
1. Implemented the evaluation pipeline
2. Completed the “Analyzing Overfitting Control” section
3. Completed Advanced Direction 1: Hyperparameters Comparison
4. Contributed to the development of the main training script
   
**Yuk Yeung Wong**:
1. Convenient Hyperparameter Configuration
2. Training GUI
3. Web-Based Chat Interface
4. LoRA-Based Assessment
5. Advanced 3: Safety \& Legality + OOD Detection

---

## 🙌 Acknowledgements

* HKUST(GZ) AIAA3102 course staff
* HuggingFace Transformers & PEFT
* TinyLlama development team

---

## Git Upload Log (AIAA3102 Final Project)

```
从Colab上传到github
0. 在github里面创建此仓库

1. 初始化本地仓库
cd /content/drive/MyDrive/AIAA3102/Final_Project
git init

2. 配置全局身份，各位自行修改Github账号的name和所用油箱
git config --global user.name "EnjiXiong"
git config --global user.email "exiong092@connect.hkust-gz.edu.cn"

3. 添加文件，生成commits
git add .
git commit -m "Initial commit: Final Project files"

4. 添加远端仓库
git remote set-url origin https://<YOUR_TOKEN>@github.com/EnjiXiong/AIAA3102---Final-Project.git #Token 在github -> settings -> developer -> api里面找
git remote add origin https://github.com/EnjiXiong/AIAA3102---Final-Project.git
   
5. 将默认分支改为 main
git branch -M main

6. 创建的时候加入了README.md，导致需要文件合并，否则直接使用最后一行push即可。
git fetch origin
git pull --rebase origin main
git push -u origin main

```
