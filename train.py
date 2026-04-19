import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig
from src.trainer import RobustGRPOTrainer
from src.rewards import math_binary_reward, dynamic_format_reward

def format_english_math(examples):
    """
    Chuẩn bị dữ liệu Tiếng Anh: 
    Bọc câu hỏi vào Chat Template và trích xuất cột đáp án/độ khó
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
    
    # Đổi tên cột 'solution' thành 'answer' để khớp với tham số truyền vào hàm math_binary_reward
    return {
        "prompt": prompts,
        "answer": examples["solution"], 
        "level": examples["level"]
    }

def main():
    # 1. Cấu hình Model với Unsloth
    max_seq_length = 2048 
    print("⏳ Nạp mô hình (Sử dụng bản đã SFT Warm-up tiếng Anh)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Lấy mô hình đã được SFT 200 steps bằng tập OpenR1-Math làm khởi điểm
        model_name="outputs/Qwen-1.5B-SFT-Adapter", 
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )

    # Thêm LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 2. Tải Dataset Tiếng Anh (hendrycks/competition_math)
    print("⏳ Đang tải và xử lý dataset Tiếng Anh (MATH)...")
    dataset = load_dataset("hendrycks/competition_math", split="train")
    
    # Loại bỏ các cột thừa để tránh lỗi khi nạp vào Trainer
    dataset = dataset.map(format_english_math, batched=True, remove_columns=dataset.column_names)

    # 3. Cấu hình Huấn luyện DRO-GRPO
    training_args = GRPOConfig(
        output_dir="outputs/Qwen-1.5B-English-DRO",
        learning_rate=1.5e-5,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        num_generations=4, 
        max_completion_length=1024,
        beta=0.001, 
        logging_steps=1,
        use_vllm=True, 
    )

    # 4. Khởi tạo Custom Trainer (Có Annealing Temperature)
    trainer = RobustGRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, dynamic_format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dr_temp_start=100.0, # Bắt đầu cào bằng (khám phá)
        dr_temp_end=0.8,     # Kết thúc siết chặt (khai thác câu khó)
    )

    print("🚀 Bắt đầu huấn luyện DRO-GRPO (English MATH)...")
    trainer.train()

    # 5. Lưu Model
    print("💾 Đang lưu mô hình...")
    model.save_pretrained("outputs/Qwen-1.5B-English-DRO-Final")
    tokenizer.save_pretrained("outputs/Qwen-1.5B-English-DRO-Final")

if __name__ == "__main__":
    main()