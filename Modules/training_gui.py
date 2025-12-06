# /content/drive/MyDrive/AIAA3102/Final_Project/Modules/training_gui.py
import gradio as gr
import subprocess
import sys
import time
import os
from pathlib import Path
import json

# Define paths
BASE_DIR = "/content/drive/MyDrive/AIAA3102/Final_Project"
SCRIPTS_DIR = f"{BASE_DIR}/Scripts"
CONFIGS_DIR = f"{BASE_DIR}/Configs"
DATA_DIR = f"{BASE_DIR}/Data"
MODELS_DIR = f"{BASE_DIR}/Models"
RESULTS_DIR = f"{BASE_DIR}/Results"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training process
training_process = None
log_file = f"{RESULTS_DIR}/training_log.txt"

def run_training(
    learning_rate, 
    num_epochs, 
    train_batch_size, 
    grad_accum_steps,
    use_lora, 
    lora_rank, 
    lora_alpha, 
    lora_dropout,
    use_qlora,
    max_train_samples,
    max_eval_samples,
    eval_batch_size,
    metric_for_best_model
):
    global training_process
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"{MODELS_DIR}/finetuned_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        f"{SCRIPTS_DIR}/train_base_wyy.py",
        "--config_dir", CONFIGS_DIR,
        "--train_file", f"{DATA_DIR}/con_train_3000.jsonl",
        "--valid_file", f"{DATA_DIR}/con_valid_3000.jsonl",
        "--output_dir", output_dir,
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_epochs),
        "--per_device_train_batch_size", str(train_batch_size),
        "--per_device_eval_batch_size", str(eval_batch_size),
        "--gradient_accumulation_steps", str(grad_accum_steps),
        "--max_eval_samples", str(max_eval_samples),
        "--metric_for_best_model", metric_for_best_model
    ]
    
    # Add LoRA/QLoRA params
    if use_lora:
        cmd.append("--use_lora")
        cmd.extend([
            "--lora_r", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--lora_dropout", str(lora_dropout)
        ])
    if use_qlora:
        cmd.append("--use_qlora")
    if max_train_samples > 0:
        cmd.extend(["--max_train_samples", str(max_train_samples)])
    
    # Start training
    with open(log_file, "w") as f:
        training_process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    # Wait for process to start
    time.sleep(2)
    
    status_html = f"""
    <div style="display: flex; align-items: center; padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">
        <span style="display: inline-block; width: 12px; height: 12px; background: #4caf50; border-radius: 50%; margin-right: 10px;"></span>
        <div>
            <strong>Training Started!</strong><br>
            <span style="color: #666; font-size: 0.9em;">Output directory: {output_dir}</span><br>
            <span style="color: #1976d2; font-size: 0.9em;">Click 'Refresh Logs' below to monitor progress</span>
        </div>
    </div>
    """
    
    return status_html, gr.update(interactive=False), gr.update(interactive=True)

def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        training_process.wait()
        status_html = """
        <div style="display: flex; align-items: center; padding: 15px; background: #ffebee; border-radius: 8px; border-left: 4px solid #f44336;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #f44336; border-radius: 50%; margin-right: 10px;"></span>
            <div>
                <strong>Training Stopped!</strong><br>
                <span style="color: #666; font-size: 0.9em;">Model saved to last checkpoint</span>
            </div>
        </div>
        """
        return status_html, gr.update(interactive=True), gr.update(interactive=False)
    status_html = """
    <div style="display: flex; align-items: center; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #9e9e9e;">
        <span style="display: inline-block; width: 12px; height: 12px; background: #9e9e9e; border-radius: 50%; margin-right: 10px;"></span>
        <div>
            <strong>No Active Training</strong><br>
            <span style="color: #666; font-size: 0.9em;">Click 'Start Training' to begin</span>
        </div>
    </div>
    """
    return status_html, gr.update(interactive=True), gr.update(interactive=False)

# 替换为美化前的简单日志读取函数
def read_logs():
    if not os.path.exists(log_file):
        return "No logs available yet. Training may not have started."
    try:
        with open(log_file, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading logs: {str(e)}"

def create_gui():
    with gr.Blocks(title="AIAA 3102 Final Project") as demo:  # 修改了浏览器标签标题
        # Header - 修改了主标题
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; font-family: 'Arial', sans-serif; margin-bottom: 10px;">🚀 AIAA 3102 Final Project</h1>
            <p style="color: rgba(255,255,255,0.9); max-width: 800px; margin: 0 auto; line-height: 1.6;">
                Fine-tuning TinyLlama with LoRA/QLoRA for specialized domain adaptation
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                # Training Configuration
                gr.Markdown("### ⚙️ Training Configuration")
                
                with gr.Accordion("📊 Training Hyperparameters", open=True):
                    lr = gr.Number(value=5e-5, label="Learning Rate", minimum=1e-6, maximum=1e-3, step=1e-6)
                    epochs = gr.Number(value=2, label="Number of Epochs", minimum=1, maximum=100, step=1)
                    train_bs = gr.Number(value=2, label="Train Batch Size", minimum=1, maximum=32, step=1)
                    grad_accum = gr.Number(value=1, label="Gradient Accumulation Steps", minimum=1, maximum=16, step=1)
                
                with gr.Accordion("🔍 Evaluation Settings", open=True):
                    max_train_samples = gr.Number(value=0, label="Max Training Samples (0=all)", minimum=0, step=100)
                    max_eval_samples = gr.Number(value=50, label="Max Evaluation Samples", minimum=10, maximum=1000, step=10)
                    eval_bs = gr.Number(value=1, label="Eval Batch Size", minimum=1, maximum=16, step=1)
                    metric = gr.Dropdown(
                        ["eval_loss", "eval_token_loss"],
                        value="eval_loss",
                        label="Best Model Metric"
                    )
                
                with gr.Accordion("🧩 LoRA/QLoRA Configuration", open=True):
                    use_lora = gr.Checkbox(label="Enable LoRA", value=True)
                    with gr.Group(visible=True) as lora_group:
                        lora_rank = gr.Number(value=8, label="LoRA Rank (r)", minimum=1, maximum=128, step=1)
                        lora_alpha = gr.Number(value=32, label="LoRA Alpha", minimum=1, maximum=256, step=1)
                        lora_dropout = gr.Slider(value=0.1, label="LoRA Dropout", minimum=0, maximum=0.5, step=0.01)
                    use_qlora = gr.Checkbox(label="Enable QLoRA (4-bit)", value=True)
                    
                    # Toggle LoRA section visibility
                    use_lora.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=use_lora,
                        outputs=lora_group
                    )
                
                # Control buttons
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("🛑 Stop Training", variant="stop", interactive=False)
                
                # Status display
                status_display = gr.HTML("""
                <div style="display: flex; align-items: center; padding: 15px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #4caf50;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #8bc34a; border-radius: 50%; margin-right: 10px;"></span>
                    <div>
                        <strong>Ready to Train</strong><br>
                        <span style="color: #666; font-size: 0.9em;">Configure parameters and click 'Start Training'</span>
                    </div>
                </div>
                """)
            
            with gr.Column():
                # Logs and Progress
                gr.Markdown("### 📋 Training Logs")
                
                # 使用 Textbox 替代 HTML 组件以显示纯文本日志
                log_output = gr.Textbox(
                    value="Training logs will appear here...",
                    lines=25,
                    max_lines=50,
                    interactive=False,
                    show_copy_button=True
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")
                
                gr.Markdown("""
                <div style="margin-top: 15px; padding: 12px; background: #e3f2fd; border-radius: 8px; border-left: 3px solid #2196f3;">
                    <strong>💡 Tips:</strong>
                    <ul style="margin-top: 8px; color: #1976d2; margin-left: 20px;">
                        <li>Click <strong>Refresh Logs</strong> to update training progress</li>
                        <li>Reduce batch size if you encounter memory issues</li>
                        <li>Enable QLoRA for 4-bit quantization on limited GPU memory</li>
                        <li>Monitor loss values to track convergence</li>
                    </ul>
                </div>
                """)
        
        # Event handlers
        start_btn.click(
            fn=run_training,
            inputs=[
                lr, epochs, train_bs, grad_accum,
                use_lora, lora_rank, lora_alpha, lora_dropout,
                use_qlora,
                max_train_samples, max_eval_samples, eval_bs, metric
            ],
            outputs=[status_display, start_btn, stop_btn]
        )
        
        stop_btn.click(
            fn=stop_training,
            outputs=[status_display, start_btn, stop_btn]
        )
        
        refresh_btn.click(
            fn=read_logs,
            outputs=log_output
        )
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_gui()
    demo.launch(
        share=True, 
        debug=True,
        server_port=7860,
        show_api=False
    )