import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# 1. Cấu hình Model với Unsloth (Tối ưu cho GPU VRAM thấp)
max_seq_length = 2048 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True, 
    fast_inference=True,
)

# 2. Thêm LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Áp dụng chuẩn Chat Template của Qwen
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

def format_dataset(examples):
    """
    Chuyển đổi problem và solution từ OpenR1 thành định dạng hội thoại messages.
    """
    messages = []
    for problem, solution in zip(examples["problem"], examples["solution"]):
        messages.append([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ])
    return {"messages": messages}

# 4. Tải và xử lý dataset
print("Tải dataset OpenR1-Math-220k...")
# Tải dataset, có thể tải một phần để tiết kiệm thời gian mapping
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="default")

# Xào trộn (shuffle) và áp dụng format
dataset = dataset.shuffle(seed=42).map(format_dataset, batched=True, remove_columns=dataset.column_names)

# Sử dụng hàm apply_chat_template tích hợp sẵn của Unsloth để biến dictionary thành chuỗi text
def apply_template(examples):
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
    return {"text": texts}

dataset = dataset.map(apply_template, batched=True)

# 5. Cấu hình SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=200,            # CHỈ TRAIN 200 STEPS THEO YÊU CẦU
        learning_rate=2e-5,       # LR nhỏ để model không quên kiến thức gốc
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/SFT-checkpoint",
    ),
)

# 6. Bắt đầu huấn luyện SFT
print("🚀 Bắt đầu Warm-up SFT...")
trainer.train()

# 7. Lưu Adapter SFT
print("💾 Lưu LoRA Adapter...")
model.save_pretrained("outputs/Qwen-1.5B-SFT-Adapter")
tokenizer.save_pretrained("outputs/Qwen-1.5B-SFT-Adapter")