import torch
from trl import GRPOConfig
from unsloth import FastLanguageModel

from src.dataset import prepare_math_dataset
from src.rewards import accuracy_reward, format_reward, length_penalty_reward
from src.trainer import RobustGRPOTrainer

def main():
    print("🚀 KHỞI ĐỘNG PIPELINE DR. GRPO VỚI UNSLOTH...")
    
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None, 
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = prepare_math_dataset(max_samples=500)

    training_args = GRPOConfig(
        output_dir="./math_grpo_outputs",
        learning_rate=2e-5,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        num_generations=4,             
        max_prompt_length=256,
        max_completion_length=768,     
        beta=0.04,                     
        logging_steps=5,
        save_steps=100,
        max_steps=500,                 
        report_to="wandb"              
    )

    trainer = RobustGRPOTrainer(
        dr_temperature=0.5,
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[accuracy_reward, format_reward, length_penalty_reward]
    )

    print("🔥 BẮT ĐẦU QUÁ TRÌNH TIẾN HÓA (TRAINING)...")
    trainer.train()
    
    print("💾 Đang lưu trọng số LoRA...")
    model.save_pretrained("dr_grpo_math_lora_model")
    tokenizer.save_pretrained("dr_grpo_math_lora_model")
    print("🎉 HOÀN TẤT!")

if __name__ == "__main__":
    main()