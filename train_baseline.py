import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# NẠP HÀM REWARD TRUYỀN THỐNG (Bỏ dynamic_format_reward)
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
    
    # Chỉ cần prompt và answer, không cần level vì Reward truyền thống không dùng tới
    return {
        "prompt": prompts,
        "answer": examples["solution"]
    }

def main():
    max_seq_length = 2048 
    print("⏳ Nạp mô hình SFT Baseline...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/Qwen-1.5B-SFT-Adapter", 
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("⏳ Đang tải dataset Tiếng Anh (MATH)...")
    dataset = load_dataset("hendrycks/competition_math", split="train")
    dataset = dataset.map(format_english_math, batched=True, remove_columns=dataset.column_names)

    training_args = GRPOConfig(
        output_dir="outputs/Qwen-1.5B-Standard-GRPO",
        learning_rate=1.5e-5,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_generations=4, 
        max_completion_length=1024,
        beta=0.001, 
        logging_steps=1,
        use_vllm=True, 
        temperature=1.0, # GRPO gốc cào bằng mọi thứ
    )

    trainer = GRPOTrainer(
        model=model,
        # SỬ DỤNG HÀM REWARD TRUYỀN THỐNG VÀ TOÁN HỌC TRUYỀN THỐNG
        reward_funcs=[math_binary_reward, standard_format_reward], 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🚀 Bắt đầu huấn luyện GRPO Truyền thống (Baseline)...")
    trainer.train()

    print("💾 Đang lưu mô hình Baseline...")
    model.save_pretrained("outputs/Qwen-1.5B-Standard-GRPO-Final")
    tokenizer.save_pretrained("outputs/Qwen-1.5B-Standard-GRPO-Final")

if __name__ == "__main__":
    main()