import torch
import sys
import os
import random
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

class SFTVisualizerCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, sample_every=50):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.sample_every = sample_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            
            messages = item["messages"]
            # Ground truth lúc này sẽ chứa nội dung từ cột 'generations'
            ground_truth = messages[1]["content"]
            
            input_messages = messages[:1] 
            text = self.tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]

            print("\n" + "=" * 50)
            print(f"📊 [SFT DEBUG] STEP: {state.global_step}")
            print(f"✅ [GROUND TRUTH (from generations)]:\n{ground_truth[:300]}...")
            print(f"🤖 [MODEL OUTPUT]:\n{output_text}")
            print("=" * 50 + "\n")
            
            model.train()

def train_sft_warmup(model_id: str, output_dir: str):
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Model (BF16, No Quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    # 3. Cấu hình LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 4. Dataset Processing - CẬP NHẬT CỘT 'generations'
    print("⏳ Loading dataset OpenR1-Math-220k...")
    raw_dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    
    def formatting_prompts_func(example):
        # Lưu ý: Cột 'generations' thường là một danh sách (list), 
        # ta lấy phần tử đầu tiên là lời giải mẫu chuẩn.
        generation_content = example["generations"]
        if isinstance(generation_content, list):
            generation_content = generation_content[0]

        return {
            "messages": [
                {"role": "user", "content": example["problem"]},
                {"role": "assistant", "content": generation_content}
            ]
        }

    # Chọn 5000 mẫu để warm-up nhanh
    sft_dataset = raw_dataset.shuffle(seed=42).select(range(5000)).map(formatting_prompts_func)

    # 5. SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages", 
        max_length=2048,
        learning_rate=2e-5,          
        lr_scheduler_type="cosine",
        logging_steps=10,           
        max_steps=200, 
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True, 
        bf16=True,                                   
        remove_unused_columns=False, 
        save_steps=100,
        report_to="none",
    )

    visualizer = SFTVisualizerCallback(tokenizer, sft_dataset, sample_every=50)

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[visualizer], 
    )

    print("🚀 Bắt đầu Warm-up SFT với cột Generations (Full Precision)...")
    trainer.train()
    
    # 7. Lưu kết quả
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 Đã lưu Adapter tại: {output_dir}")

if __name__ == "__main__":
    train_sft_warmup(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="outputs/Qwen-1.5B-SFT-Adapter"
    )