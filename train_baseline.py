import sys
import os
import transformers
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel

# Importing your reward functions
from src.rewards import math_binary_reward, standard_format_reward

class GRPOVisualizerCallback(TrainerCallback):
    """
    Callback để in ra quá trình suy luận của mô hình trong lúc train GRPO.
    """
    def __init__(self, tokenizer, dataset, sample_every=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.sample_every = sample_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Chỉ in log theo chu kỳ sample_every
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            
            # 1. Lấy mẫu ngẫu nhiên
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            
            prompt = item["prompt"]
            ground_truth = item["answer"]
            
            # 2. Sinh văn bản (Inference)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Cắt bỏ phần prompt để chỉ lấy phần trả lời
                completion_ids = generated_ids[0][len(inputs.input_ids[0]):]
                output_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            print("\n" + "🔍" * 20)
            print(f"📊 [GRPO DEBUG] STEP: {state.global_step}")
            print(f"❓ [QUESTION]:\n{item.get('problem', 'N/A')}")
            print(f"✅ [GROUND TRUTH ANSWER]: {ground_truth}")
            print(f"🤖 [MODEL REASONING & OUTPUT]:\n{output_text}")
            print("🔍" * 20 + "\n")
            
            model.train()

def format_vietnamese_math(examples):
    prompts = []
    for question in examples["problem"]:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful math assistant. Please reason step by step and "
            "put your final answer within <answer> and </answer> tags.\n"
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
        "problem": examples["problem"] # Giữ lại để in log
    }

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_adapter_path = "outputs/Qwen-1.5B-SFT-Adapter"
    output_dir = "outputs/Qwen-1.5B-Standard-GRPO"

    print(f"⏳ Loading Base Model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if os.path.exists(sft_adapter_path):
        print(f"🔄 Merging SFT Adapter...")
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()
        print("✅ SFT merged.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load Dataset
    configs = ['algebra', 'geometry', 'number_theory'] # Chọn vài tập để test nhanh
    all_splits = []
    for cfg in configs:
        all_splits.append(load_dataset("ura-hcmut/Vietnamese-MATH", cfg, split="train"))
    dataset = concatenate_datasets(all_splits)
    dataset = dataset.map(format_vietnamese_math, batched=True, remove_columns=dataset.column_names)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        optim="adamw_8bit",
        logging_steps=1,
        max_steps=10, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=512,
        use_vllm=False,
        bf16=True,
        report_to="none"
    )

    # KHỞI TẠO CALLBACK
    visualizer = GRPOVisualizerCallback(tokenizer, dataset, sample_every=1)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[math_binary_reward, standard_format_reward], 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[visualizer] # THÊM VÀO ĐÂY
    )

    print("🚀 Starting Baseline Training with Debug Logs...")
    trainer.train()

    trainer.save_model(f"{output_dir}-Final")
    print(f"💾 Saved at: {output_dir}-Final")

if __name__ == "__main__":
    main()