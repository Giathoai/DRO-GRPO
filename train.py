# train.py
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig
from src.trainer import RobustGRPOTrainer
from src.rewards import math_binary_reward, dynamic_format_reward

# 1. Cấu hình Model với Unsloth (Tối ưu cho GPU VRAM thấp)
max_seq_length = 2048 # Đủ để chứa prompt và suy luận
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/Qwen-1.5B-SFT-Adapter",
    max_seq_length=max_seq_length,
    load_in_4bit=True, # Bắt buộc để chạy trên GPU 12GB
    fast_inference=True,
    gpu_memory_utilization=0.6, # Chừa RAM cho vLLM sinh text
)

# Thêm LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# 2. Chuẩn bị Dataset Vietnamese-MATH
# Dataset cần có các cột: 'prompt', 'answer', 'level'
dataset = load_dataset("ura-hcmut/Vietnamese-MATH", split="train")
# (Bạn cần viết thêm hàm map để bọc câu hỏi vào template của Qwen ở đây)

# 3. Cấu hình Huấn luyện
training_args = GRPOConfig(
    output_dir="outputs/Qwen-1.5B-DRO",
    learning_rate=1.5e-5,
    per_device_train_batch_size=1, # Rollout 1 câu hỏi
    gradient_accumulation_steps=4, # Tích lũy để tạo Batch = 4
    num_generations=8, # G=4 phản hồi mỗi câu hỏi
    max_completion_length=2048,
    beta=0.001, # KL Penalty rất nhỏ cho phép tự do suy luận
    logging_steps=1,
    use_vllm=True, # Tăng tốc sinh text
)

# 4. Khởi tạo Custom Trainer và Train
trainer = RobustGRPOTrainer(
    model=model,
    reward_funcs=[math_binary_reward, dynamic_format_reward],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    dr_temperature=0.8, # NHIỆT ĐỘ DRO CỦA BẠN NẰM Ở ĐÂY
)

print("🚀 Bắt đầu huấn luyện hệ thống DRO-GRPO...")
trainer.train()

# 5. Lưu Model
model.save_pretrained("outputs/Qwen-1.5B-DRO-Final")
tokenizer.save_pretrained("outputs/Qwen-1.5B-DRO-Final")