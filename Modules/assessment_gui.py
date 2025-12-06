# /content/drive/MyDrive/AIAA3102/Final_Project/Modules/assessment_gui.py
import gradio as gr
import torch
import json
import random
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig
import re
import time

# Define paths
BASE_DIR = "/content/drive/MyDrive/AIAA3102/Final_Project"
MODELS_DIR = f"{BASE_DIR}/Models"
DATA_DIR = f"{BASE_DIR}/Data"
VALID_FILE = f"{DATA_DIR}/con_valid_3000.jsonl"

# Global variables
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
current_model_path = None

# Safety and domain check configurations
DANGEROUS_WORDS = [
    "suicide", "suicidal", "kill myself", "end my life", 
    "self-harm", "self harm", "cut myself", "overdose",
    "no reason to live", "want to die", "hang myself"
]

ILLEGAL_WORDS = [
    "bomb", "explosive", "make a bomb", "create a bomb", "build a bomb",
    "drugs", "illegal drugs", "cocaine", "heroin", "meth", "fentanyl", "mdma",
    "weapon", "gun", "illegal weapon", "steal", "robbery", "hijack", "terrorism",
    "hack", "hacking", "ddos attack", "cyber attack"
]

OOD_KEYWORDS = [
    "math", "mathematics", "calculus", "algebra", "geometry", "trigonometry",
    "programming", "code", "python", "javascript", "java", "c++", "c#", "html", "css",
    "physics", "chemistry", "biology", "science experiment", "lab report",
    "recipe", "cooking", "how to cook", "how to bake", "food recipe",
    "car repair", "fix my car", "engine problem", "mechanic",
    "stock market", "investment advice", "financial planning", "portfolio",
    "legal advice", "lawyer", "court case", "lawsuit", "contract",
    "medical diagnosis", "prescribe", "medication dosage", "surgery", "symptom"
]

STRONG_COUNSELING_INDICATORS = [
    "anxious", "anxiety", "stress", "stressed", "depressed", "depression",
    "worried", "worry", "fear", "afraid", "panic", "overwhelmed", 
    "sleep", "insomnia", "can't sleep", "trouble sleeping", "poor sleep",
    "mental health", "emotional", "feelings", "mood", "therapist", "counselor",
    "burnout", "exhausted", "tired", "fatigue", "pressure", "overwhelmed",
    "academic pressure", "exam stress", "test anxiety", "nervous", "scared",
    "trauma", "traumatic", "grief", "loss", "bereavement", "mourning",
    "relationship problems", "family issues", "marriage counseling",
    "self-esteem", "confidence", "identity crisis", "self-worth",
    "coping strategies", "resilience", "crisis", "emergency", "suicidal",
    "self-harm", "addiction", "substance abuse", "eating disorder",
    "social anxiety", "loneliness", "isolation", "work stress", "job stress",
    "perfectionism", "imposter syndrome", "panic attack", "phobia",
    "ptsd", "ocd", "bipolar", "schizophrenia", "adhd", "autism",
    "therapy", "counseling", "psychiatrist", "psychologist", "meditation",
    "mindfulness", "breathing exercises", "grounding techniques", "relaxation"
]

def check_dangerous_content(text):
    """
    Check for dangerous or illegal content using exact match.
    Returns a dictionary with detection results.
    """
    text_lower = text.lower().strip()
    
    # Check for dangerous words (self-harm, suicide)
    for word in DANGEROUS_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return {
                "is_blocked": True,
                "type": "dangerous",
                "matched_word": word,
                "message": "I hear you're going through a difficult time, and I truly care about your well-being. Please reach out to a mental health professional or a crisis hotline immediately. You're not alone, and help is available. In the US, you can call or text 988 for the Suicide & Crisis Lifeline."
            }
    
    # Check for illegal words (bombs, drugs, etc.)
    for word in ILLEGAL_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return {
                "is_blocked": True,
                "type": "illegal",
                "matched_word": word,
                "message": "I cannot provide assistance with illegal activities or harmful content. My purpose is to offer support for mental health and emotional well-being within legal and ethical boundaries."
            }
    
    return {"is_blocked": False}

def check_out_of_domain(text):
    """
    Check for out-of-domain content using smarter context-aware matching.
    This version correctly handles cases like "anxious about math exam".
    """
    text_lower = text.lower().strip()
    
    # First, check if we have strong counseling context indicators
    has_strong_context = False
    for indicator in STRONG_COUNSELING_INDICATORS:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            has_strong_context = True
            break  # Only need one strong indicator
    
    # If we have strong counseling context, allow the request regardless of OOD keywords
    if has_strong_context:
        return {"is_blocked": False}
    
    # Only if no strong counseling context, check for OOD content
    matched_ood = []
    for keyword in OOD_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            matched_ood.append(keyword)
    
    # If no OOD keywords found, allow the request
    if not matched_ood:
        return {"is_blocked": False}
    
    # If we have OOD keywords and no strong counseling context, block the request
    display_keywords = matched_ood[:3]
    if len(matched_ood) > 3:
        display_keywords.append("and more")
    
    return {
        "is_blocked": True,
        "type": "out_of_domain",
        "matched_keywords": matched_ood,
        "message": f"I notice your question relates to {', '.join(display_keywords)}, which is outside my area of expertise as a counseling-focused AI assistant. I'm designed to help with mental health, emotional well-being, and personal challenges. For questions about specific technical topics, I'd recommend consulting with a specialist in that field."
    }

def get_available_models():
    """Get list of available model directories in Models folder"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        return []
    
    model_dirs = []
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            # Check if this directory contains model files
            has_model_files = any(
                f.endswith('.bin') or f.endswith('.safetensors') or f == 'adapter_model.bin'
                for f in os.listdir(item_path)
            )
            if has_model_files:
                model_dirs.append(item)
    
    return sorted(model_dirs, reverse=True)  # Most recent first

def load_model(model_name):
    """Load the selected model"""
    global model, tokenizer, current_model_path
    
    if model_name == current_model_path and model is not None:
        return f"Model {model_name} already loaded"
    
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        current_model_path = model_name
        
        # Check if this is a LoRA adapter
        is_lora = os.path.exists(os.path.join(model_path, "adapter_model.bin")) or \
                 os.path.exists(os.path.join(model_path, "adapter"))
        
        # Get base model name from config if available
        base_model_name = "TinyLlama/TinyLlama_v1.1"  # Default fallback
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if "base_model_name_or_path" in config:
                    base_model_name = config["base_model_name_or_path"]
            except Exception as e:
                print(f"Error reading config: {e}")
        
        print(f"Loading base model: {base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        # Check for quantization
        use_4bit = False
        quant_config_path = os.path.join(model_path, "quantization_config.json")
        if os.path.exists(quant_config_path):
            try:
                with open(quant_config_path, 'r') as f:
                    quant_config = json.load(f)
                use_4bit = quant_config.get("load_in_4bit", False)
            except Exception as e:
                print(f"Error reading quantization config: {e}")
        
        # Load model with appropriate configuration
        if use_4bit:
            print("Loading model in 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print("Loading model in standard precision")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Load LoRA adapter if present
        if is_lora:
            adapter_path = os.path.join(model_path, "adapter") if os.path.exists(os.path.join(model_path, "adapter")) else model_path
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("LoRA adapter successfully loaded")
        
        # Move model to device if not using device_map
        if not hasattr(model, "hf_device_map"):
            model.to(device)
        
        print(f"Model {model_name} loaded successfully on {device}")
        return f"✅ Model '{model_name}' loaded successfully!"
    
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg

def load_valid_data(num_samples=5):
    """Load and sample validation data"""
    if not os.path.exists(VALID_FILE):
        return [{"prompt": "Validation file not found", "response": "Please check the path to valid.jsonl"}]
    
    try:
        with open(VALID_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse JSON lines
        valid_data = []
        for line in lines:
            try:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if "prompt" in data and "response" in data:
                        valid_data.append({
                            "prompt": data["prompt"].strip(),
                            "response": data["response"].strip()
                        })
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
        
        if not valid_data:
            return [{"prompt": "No valid data found", "response": "The validation file may be empty or malformed"}]
        
        # Sample data
        if num_samples > len(valid_data):
            num_samples = len(valid_data)
        
        return random.sample(valid_data, num_samples)
    
    except Exception as e:
        return [{"prompt": f"Error loading validation data: {str(e)}", "response": "Please check the validation file format"}]

def generate_validation_results(num_samples=5):
    """Generate validation results with proper prompt format"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return [["❌ No model loaded", "", "", ""]]
    
    try:
        # Load validation data
        valid_data = load_valid_data(num_samples)
        results = []
        
        for i, item in enumerate(valid_data):
            # Format prompt properly
            prompt = f"Question:\n{item['prompt']}\nAnswer:\n"
            true_response = item["response"]
            
            # Generate response
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and clean up the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in generated_text:
                generated_response = generated_text.split("Answer:", 1)[1].strip()
            else:
                generated_response = generated_text.strip()
            
            # Clean up extra spaces and line breaks
            generated_response = re.sub(r'\s+', ' ', generated_response).strip()
            true_response = re.sub(r'\s+', ' ', true_response).strip()
            
            results.append([
                f"Sample {i+1}",
                item['prompt'][:100] + "..." if len(item['prompt']) > 100 else item['prompt'],
                generated_response[:200] + "..." if len(generated_response) > 200 else generated_response,
                true_response[:200] + "..." if len(true_response) > 200 else true_response
            ])
        
        return results
    
    except Exception as e:
        return [["❌ Generation Error", str(e), "", ""]]

def generate_response(user_input, temperature=0.7, top_p=0.9):
    """Generate response for user input with safety checks and proper prompt format"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ No model loaded. Please select and load a model first."
    
    if not user_input.strip():
        return "❌ Please enter a prompt."
    
    # Check for dangerous/illegal content first
    danger_check = check_dangerous_content(user_input)
    if danger_check["is_blocked"]:
        safety_type = "Dangerous Content" if danger_check["type"] == "dangerous" else "Illegal Content"
        return f"⚠️ SAFETY BLOCK ({safety_type}): {danger_check['message']}"
    
    # Check for out-of-domain content
    ood_check = check_out_of_domain(user_input)
    if ood_check["is_blocked"]:
        return f"⚠️ DOMAIN RESTRICTION: {ood_check['message']}"
    
    try:
        # Format prompt properly - THIS IS THE KEY FIX
        formatted_prompt = f"Question:\n{user_input.strip()}\nAnswer:\n"
        print(f"Formatted prompt: {formatted_prompt[:100]}...")  # Debug output
        
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw generated text: {generated_text[:200]}...")  # Debug output
        
        # Extract only the answer part after "Answer:"
        if "Answer:" in generated_text:
            response = generated_text.split("Answer:", 1)[1].strip()
        else:
            # Fallback: if no "Answer:" marker, take everything after the prompt
            if formatted_prompt in generated_text:
                response = generated_text.split(formatted_prompt, 1)[1].strip()
            else:
                response = generated_text.strip()
        
        # Clean up response
        response = re.sub(r'\s+', ' ', response).strip()
        print(f"Cleaned response: {response[:200]}...")  # Debug output
        
        return response
    
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

def create_assessment_gui():
    with gr.Blocks(title="AIAA 3102 Model Assessment") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4a6fa5 0%, #6a11cb 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; font-family: 'Arial', sans-serif; margin-bottom: 10px;">🔍 AIAA 3102 Model Assessment</h1>
            <p style="color: rgba(255,255,255,0.9); max-width: 800px; margin: 0 auto; line-height: 1.6;">
                Evaluate fine-tuned TinyLlama models on validation data and interactive conversations
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📂 Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    info="Choose a trained model from your Models directory"
                )
                load_btn = gr.Button("🔄 Load Selected Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                with gr.Accordion("ℹ️ Model Information", open=False):
                    gr.Markdown("""
                    **Model Loading Information:**
                    - Models are loaded from `/content/drive/MyDrive/AIAA3102/Final_Project/Models`
                    - Both full models and LoRA adapters are supported
                    - 4-bit quantized models will be detected automatically
                    - Loading may take 1-2 minutes depending on model size
                    """)
            
            with gr.Column():
                gr.Markdown("### ⚙️ Generation Parameters (for Chat)")
                with gr.Group():
                    temp_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature", 
                        info="Controls randomness: lower = more focused, higher = more creative"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p (Nucleus Sampling)",
                        info="Controls diversity: lower = more focused, higher = more diverse"
                    )
                
                with gr.Accordion("🛡️ Safety & Domain Features", open=False):
                    gr.Markdown("""
                    **Safety & Domain Protection:**
                    - **Dangerous Content Detection**: Blocks requests containing suicide, self-harm keywords
                    - **Illegal Content Detection**: Blocks requests about bombs, drugs, weapons
                    - **Domain Restriction**: Only allows psychology/counseling related topics
                    - **Context-Aware Filtering**: Allows OOD keywords when in counseling context (e.g., "math anxiety")
                    """)
        
        gr.Markdown("## 📊 Validation Results")
        with gr.Row():
            with gr.Column():
                num_samples = gr.Number(
                    value=5, minimum=1, maximum=20, step=1,
                    label="Number of Validation Samples",
                    info="How many samples to randomly select from valid.jsonl"
                )
                validate_btn = gr.Button("🧪 Run Validation", variant="primary")
            
            with gr.Column():
                gr.Markdown("""
                **Validation Output Format:**
                - **Sample #**: The sequence number of the sample
                - **Input Prompt**: The first 100 characters of the validation prompt
                - **Model Output**: The first 200 characters of the model's generated response
                - **Ground Truth**: The first 200 characters of the expected response
                """)
        
        # Validation results table
        validation_results = gr.Dataframe(
            headers=["Sample #", "Input Prompt", "Model Output", "Ground Truth"],
            datatype=["str", "str", "str", "str"],
            type="array"
        )
        
        gr.Markdown("## 💬 Interactive Chat")
        with gr.Row():
            with gr.Column(scale=2):
                chat_input = gr.Textbox(
                    lines=3, 
                    placeholder="Enter your prompt here (psychology/counseling topics only)...",
                    label="User Input"
                )
                chat_btn = gr.Button("🚀 Generate Response", variant="primary")
                gr.Markdown("""
                **Safety Examples (will be blocked):**
                - *"I want to kill myself"*
                - *"How to make a bomb"*
                - *"Help me with my calculus homework"*
                
                **Allowed Examples (counseling context):**
                - *"I'm anxious about my math exam"*
                - *"My stress about programming is affecting my sleep"*
                """)
            
            with gr.Column(scale=3):
                chat_output = gr.Textbox(
                    lines=10,
                    label="Model Response",
                    interactive=False
                )
        
        # Examples for chat
        gr.Examples(
            examples=[
                "I'm feeling overwhelmed with my coursework. How can I manage my stress better?",
                "I'm having trouble communicating with my partner. What are some ways to improve our communication?",
                "I feel like I'm not making progress in my career. What steps can I take to advance?",
                "I'm struggling to maintain a work-life balance. How can I set better boundaries?",
                "I'm anxious about my upcoming math exam and it's affecting my sleep.",
                "I want to kill myself because nothing matters anymore."
            ],
            inputs=chat_input,
            label="Example Prompts (try these!)"
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em; border-top: 1px solid #eee; margin-top: 20px;">
            <p><strong>⚠️ Important Note:</strong> This model is designed for educational purposes only and is not a substitute for professional mental health care. 
            In crisis situations, please contact emergency services or a mental health professional.</p>
            <p>AIAA 3102 Final Project Assessment Tool | Model Evaluation and Interactive Testing</p>
            <p style="font-size: 0.8em; color: #888;">Last updated: December 2025</p>
        </div>
        """)
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=model_dropdown,
            outputs=model_status
        )
        
        validate_btn.click(
            fn=generate_validation_results,
            inputs=num_samples,
            outputs=validation_results
        )
        
        chat_btn.click(
            fn=generate_response,
            inputs=[chat_input, temp_slider, top_p_slider],
            outputs=chat_output
        )
        
        # Manual refresh button for models
        refresh_models_btn = gr.Button("🔄 Refresh Model List")
        refresh_models_btn.click(
            fn=get_available_models,
            outputs=model_dropdown
        )
    
    return demo

if __name__ == "__main__":
    demo = create_assessment_gui()
    demo.launch(
        share=True,
        debug=True,
        server_port=7861,  # Different port from training GUI
        show_api=False
    )