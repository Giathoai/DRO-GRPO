import sys
import os
import random
import json # THÊM THƯ VIỆN JSON
from unittest.mock import MagicMock

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Nạp các module tùy chỉnh của bạn
from src.trainer import RobustGRPOTrainer
from src.rewards import math_binary_reward, strict_format_reward, dynamic_length_penalty

# =====================================================================
# CALLBACK 1: LƯU TRẠNG THÁI REWARD ĐỂ VẼ BIỂU ĐỒ SAU NÀY
# =====================================================================
class MetricsLoggerCallback(TrainerCallback):
    """
    Callback tự động trích xuất các thông số loss, epoch và các hàm reward
    để lưu vào file JSONL. Rất tiện để dùng Pandas load lên vẽ biểu đồ.
    """
    def __init__(self, log_file="outputs/DRO_GRPO_Metrics.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Nếu muốn ghi đè file cũ mỗi lần chạy lại, bỏ comment dòng dưới:
        # if os.path.exists(self.log_file): os.remove(self.log_file)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # Tạo một dictionary chứa step hiện tại
        metrics_to_save = {"step": state.global_step}
        
        # Lọc ra những tham số liên quan đến reward, loss và epoch
        for key, value in logs.items():
            if "reward" in key.lower() or key in ["loss", "epoch"]:
                metrics_to_save[key] = value
                
        # Nếu có dữ liệu để lưu
        if len(metrics_to_save) > 1:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_to_save) + "\n")

# =====================================================================
# CALLBACK 2: IN LOG SUY LUẬN BẰNG TEXT (Như cũ)
# =====================================================================
class GRPOVisualizerCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, sample_every=10, log_file="outputs/DRO_GRPO_Reasoning_Logs.txt"):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.sample_every = sample_every
        self.log_file = log_file
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== NHẬT KÝ SUY LUẬN MÔ HÌNH DRO-GRPO ===\n\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            
            prompt = item["prompt"]
            ground_truth = item["answer"]
            problem = item.get("problem", "N/A")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                completion_ids = generated_ids[0][len(inputs.input_ids[0]):]
                output_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            log_content = (
                f"{'='*60}\n"
                f"🚀 [DRO-GRPO DEBUG] STEP: {state.global_step}\n"
                f"❓ [QUESTION]:\n{problem}\n"
                f"{'-'*60}\n"
                f"✅ [GROUND TRUTH ANSWER]: {ground_truth}\n"
                f"{'-'*60}\n"
                f"🤖 [MODEL REASONING & OUTPUT]:\n{output_text}\n"
                f"{'='*60}\n\n"
            )

            print(log_content)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
            
            model.train()

def format_vietnamese_math(examples):
    prompts = []
    for question in examples["problem"]:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful math assistant. Please reason step by step and "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)
    
    return {
        "prompt": prompts,
        "answer": examples["solution"], 
        "level": examples["level"],
        "problem": examples["problem"]
    }

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_adapter_path = "outputs/Qwen-1.5B-SFT-Adapter"
    output_dir = "outputs/Qwen-1.5B-DRO-GRPO"

    print(f"⏳ Đang nạp Base Model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if os.path.exists(sft_adapter_path):
        print(f"🔄 Đang gộp SFT Adapter (Reasoning Warm-up)...")
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()
        print("✅ Đã chuẩn bị xong mô hình nền tảng có tư duy CoT.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    configs = [
        'algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    print(f"⏳ Đang tải và gộp dữ liệu từ Vietnamese-MATH...")
    
    all_splits = []
    for config_name in configs:
        ds = load_dataset("ura-hcmut/Vietnamese-MATH", config_name, split="train")
        all_splits.append(ds)
    
    dataset = concatenate_datasets(all_splits)
    dataset = dataset.map(format_vietnamese_math, batched=True, remove_columns=dataset.column_names)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=500, 
        save_steps=50,               
        save_total_limit=2, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_generations=4, 
        max_completion_length=1024,
        use_vllm=False, 
        beta=0.04, 
        bf16=True,
        report_to="none",
    )

    # Khởi tạo CẢ 2 Callback
    visualizer = GRPOVisualizerCallback(tokenizer, dataset, sample_every=10, log_file="outputs/DRO_GRPO_Reasoning_Logs.txt")
    metrics_logger = MetricsLoggerCallback(log_file="outputs/DRO_GRPO_Metrics.jsonl")

    trainer = RobustGRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, strict_format_reward, dynamic_length_penalty],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dr_temp_start=100.0, 
        dr_temp_end=0.8,     
        callbacks=[visualizer, metrics_logger] # ĐƯA CALLBACK VÀO ĐÂY
    )

    print("🚀 Bắt đầu huấn luyện DRO-GRPO (Proposed Method)...")
    trainer.train()

    trainer.save_model(f"{output_dir}-Final")
    tokenizer.save_pretrained(f"{output_dir}-Final")
    print(f"💾 Hệ thống DRO-GRPO đã được lưu tại: {output_dir}-Final")

if __name__ == "__main__":
    main()