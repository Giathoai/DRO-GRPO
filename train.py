import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Nạp các module tùy chỉnh của bạn
from src.trainer import RobustGRPOTrainer
from src.rewards import math_binary_reward, dynamic_format_reward

def format_english_math(examples):
    """
    Chuẩn bị dữ liệu Tiếng Anh (MATH):
    Bọc prompt chuẩn Qwen và giữ lại cột 'level' cho Dynamic Reward.
    """
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
        "answer": examples["solution"], 
        "level": examples["level"]
    }

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_adapter_path = "outputs/Qwen-1.5B-SFT-Adapter"
    output_dir = "outputs/Qwen-1.5B-DRO-GRPO"

    # 1. Nạp Model gốc và gộp (Merge) tri thức từ SFT Warm-up
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

    # 2. Thiết lập LoRA mới cho pha Robust Optimization (DRO)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 3. Xử lý dữ liệu English MATH
    print("⏳ Đang chuẩn bị dữ liệu MATH (English)...")
    dataset = load_dataset("hendrycks/competition_math", split="train")
    dataset = dataset.map(format_english_math, batched=True, remove_columns=dataset.column_names)

    # 4. Cấu hình GRPO với vLLM Acceleration
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1.5e-5,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_generations=4, 
        max_completion_length=1024,
        max_steps=500, #hehehee
        save_steps=10,               
        save_total_limit=2, 
        gradient_checkpointing=True, 
        beta=0.001, 
        logging_steps=1,
        use_vllm=True, 
        temperature=1.0, 
        bf16=True,
        report_to="none",
    )

    # 5. Khởi tạo Robust Trainer (Vũ khí chính: DRO + Annealing)
    trainer = RobustGRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, dynamic_format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dr_temp_start=100.0, # Bắt đầu bằng việc khám phá rộng (GRPO-like)
        dr_temp_end=0.8,     # Kết thúc bằng việc tập trung vào Worst-case (DRO)
    )

    print("🚀 Bắt đầu huấn luyện DRO-GRPO (Proposed Method)...")
    trainer.train()

    # 6. Lưu kết quả cuối cùng
    trainer.save_model(f"{output_dir}-Final")
    tokenizer.save_pretrained(f"{output_dir}-Final")
    print(f"💾 Hệ thống DRO-GRPO đã được lưu tại: {output_dir}-Final")

if __name__ == "__main__":
    main()