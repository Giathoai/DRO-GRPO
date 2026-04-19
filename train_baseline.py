import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel

# Nạp các hàm Reward truyền thống
from src.rewards import math_binary_reward, standard_format_reward

def format_english_math(examples):
    prompts = []
    for question in examples["problem"]:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful math assistant. Please reason step by step and put your final answer within <answer> and </answer> tags.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)
    
    return {
        "prompt": prompts,
        "answer": examples["solution"]
    }

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_adapter_path = "outputs/Qwen-1.5B-SFT-Adapter"
    output_dir = "outputs/Qwen-1.5B-Standard-GRPO"

    # 1. Nạp Model gốc và gộp (Merge) Adapter từ bước SFT
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
        print(f"🔄 Đang gộp SFT Adapter từ {sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload() 
        print("✅ Đã gộp SFT weights thành công.")

    # 2. Cấu hình LoRA mới cho giai đoạn GRPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 3. Chuẩn bị Dataset
    print("⏳ Đang tải dataset Tiếng Anh (MATH)...")
    dataset = load_dataset("hendrycks/competition_math", split="train")
    dataset = dataset.map(format_english_math, batched=True, remove_columns=dataset.column_names)

    # 4. Cấu hình GRPOConfig (Standard Baseline - NO vLLM)
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        learning_rate=1e-5,                  
        optim="adamw_8bit",                   
        lr_scheduler_type="cosine",
        
        logging_steps=1,           
        max_steps=2, # Giữ nguyên 2 step để test cho nhanh hehe
        save_steps=10,               
        save_total_limit=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,       
        gradient_checkpointing=True, 
        
        num_generations=4,                    
        temperature=1.0,                      
        max_completion_length=1024, # Giảm nhẹ để tiết kiệm VRAM khi không có vLLM          
        
        # QUAN TRỌNG: Tắt vLLM để tránh lỗi vllm_ascend
        use_vllm=False,
        
        epsilon=0.2,                        
        epsilon_high=0.28,                   
        beta=0.04,                            
        
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none"             
    )

    # 5. Khởi tạo GRPOTrainer Tiêu chuẩn
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, standard_format_reward], 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🚀 Bắt đầu huấn luyện GRPO Truyền thống (Native HF Rollouts)...")
    trainer.train()

    # 6. Lưu mô hình cuối cùng
    trainer.save_model(f"{output_dir}-Final")
    tokenizer.save_pretrained(f"{output_dir}-Final")
    print(f"💾 Đã lưu Baseline model tại: {output_dir}-Final")

if __name__ == "__main__":
    main()