from datasets import load_dataset
from src.utils import SYSTEM_PROMPT, extract_boxed_answer

def prepare_math_dataset(max_samples=3000):
    """Tải, lọc Level và định dạng dataset ura-hcmut/Vietnamese-MATH"""
    print("⏳ Đang tải bộ dữ liệu Vietnamese-MATH từ Hugging Face...")
    dataset = load_dataset("ura-hcmut/Vietnamese-MATH", split="train")
    
    allowed_levels = ["Level 1", "Level 2", "Level 3"]
    dataset = dataset.filter(lambda x: x.get('level') in allowed_levels)
    
    def format_example(example):
        ground_truth = extract_boxed_answer(example['solution'])
        return {
            "prompt": f"{SYSTEM_PROMPT}\n\nBài toán: {example['problem']}",
            "ground_truth": ground_truth,
            "level": example['level']
        }
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    dataset = dataset.map(format_example)
    
    print(f"✅ Đã chuẩn bị xong {len(dataset)} câu hỏi Toán học.")
    return dataset