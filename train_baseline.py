import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from unsloth.chat_templates import get_chat_template

# Nạp các hàm Reward dùng chung (Để đảm bảo so sánh công bằng với DRO-GRPO)
from src.rewards import math_binary_reward, dynamic_format_reward

def format_vietnamese_math(examples):
    """
    Hàm bọc câu hỏi toán vào chuẩn Chat Template của Qwen2.5
    Ép mô hình phải dùng thẻ <think> và <answer>
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
def main():
    # 1. Cấu hình Model với Unsloth (Nạp bản đã Warm-up SFT)
    max_seq_length = 2048 
    print("⏳ Nạp mô hình SFT Baseline...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        # LẤY ĐẦU RA CỦA BƯỚC SFT LÀM ĐẦU VÀO CHO BƯỚC NÀY
        model_name="outputs/Qwen-1.5B-SFT-Adapter", 
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )

    # 2. Thêm LoRA Adapters mới cho pha RL
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Chuẩn bị Dataset Vietnamese-MATH
    print("⏳ Đang tải và xử lý dataset Vietnamese-MATH...")
    dataset = load_dataset("ura-hcmut/Vietnamese-MATH", split="train")
    
    # Chỉ lấy các cột cần thiết: problem, answer, level
    dataset = dataset.map(format_vietnamese_math, batched=True)

    # 4. Cấu hình Huấn luyện GRPO Truyền thống
    training_args = GRPOConfig(
        output_dir="outputs/Qwen-1.5B-Standard-GRPO", # LƯU VÀO THƯ MỤC BASELINE
        learning_rate=1.5e-5,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        num_generations=4, 
        max_completion_length=1024,
        beta=0.001, 
        logging_steps=1,
        use_vllm=True, 
        # Tắt các cơ chế chống cào bằng của TRL (để nó chạy đúng bản chất GRPO gốc)
        temperature=1.0, 
    )

    # 5. Khởi tạo GRPOTrainer TIÊU CHUẨN CỦA HUGGING FACE
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, dynamic_format_reward], # Dùng chung hàm reward
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🚀 Bắt đầu huấn luyện GRPO Truyền thống (Baseline)...")
    trainer.train()

    # 6. Lưu Model
    print("💾 Đang lưu mô hình Baseline...")
    model.save_pretrained("outputs/Qwen-1.5B-Standard-GRPO-Final")
    tokenizer.save_pretrained("outputs/Qwen-1.5B-Standard-GRPO-Final")

if __name__ == "__main__":
    main()