# eval.py
import re
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def extract_answer(text):
    """
    Trích xuất đáp án từ thẻ <answer> hoặc \boxed{}.
    Phù hợp cho cả định dạng CoT và đáp án LaTeX của Vietnamese-MATH.
    """
    # 1. Tìm trong thẻ <answer> trước
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        ans = match.group(1).strip()
    else:
        # 2. Nếu không có, tìm trong \boxed{} (chuẩn LaTeX của tập MATH)
        match = re.search(r"\\boxed\{(.*?)\}", text)
        ans = match.group(1).strip() if match else text.strip()
    
    # Làm sạch: bỏ dấu phẩy, khoảng trắng dư thừa
    return ans.replace(",", "").replace("$", "").strip()

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Định nghĩa các mô hình
    models_to_eval = [
        {"name": "3. SFT + Standard GRPO (Baseline)", "lora_path": "outputs/Qwen-1.5B-Standard-GRPO-Final"},
        {"name": "4. SFT + DRO-GRPO (Proposed)", "lora_path": "outputs/Qwen-1.5B-DRO-GRPO-Final"}
    ]

    # 2. Tải tập test Vietnamese-MATH
    print("⏳ Đang tải tập dữ liệu ura-hcmut/Vietnamese-MATH (Test split)...")
    # Lưu ý: Đảm bảo bạn đã có quyền truy cập hoặc dataset public
    dataset = load_dataset("ura-hcmut/Vietnamese-MATH", split="test")
    
    prompts = []
    ground_truths = []
    levels = []
    
    for item in dataset:
        question = item["problem"]
        gt = item["answer"].replace(",", "").replace("$", "").strip()
        # Trích xuất số Level (VD: "Level 1" -> 1)
        lvl_match = re.search(r"\d+", str(item["level"]))
        lvl = int(lvl_match.group()) if lvl_match else 0
        
        prompt = (
            "<|im_start|>system\n"
            "Bạn là một trợ lý toán học thông minh. Hãy suy luận từng bước trong thẻ <think> "
            "và đặt đáp án cuối cùng vào trong thẻ <answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)
        ground_truths.append(gt)
        levels.append(lvl)

    # 3. Khởi tạo vLLM
    print(f"🚀 Khởi tạo vLLM Engine...")
    llm = LLM(
        model=base_model_id, 
        enable_lora=True, 
        max_lora_rank=16, 
        max_model_len=2048
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    # 4. Loop qua từng model
    all_results = {}

    for idx, model_info in enumerate(models_to_eval):
        name = model_info["name"]
        lora_path = model_info["lora_path"]
        
        print(f"\n👉 Đang đánh giá: {name}")
        
        # Khởi tạo bảng thống kê cho model này
        # level_stats[lvl] = {correct: 0, total: 0, total_tokens: 0}
        level_stats = {l: {"correct": 0, "total": 0, "tokens": 0} for l in range(1, 6)}
        
        try:
            lora_request = LoRARequest(str(idx), idx + 1, lora_path)
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            
            for i, output in enumerate(outputs):
                pred_text = output.outputs[0].text
                pred_ans = extract_answer(pred_text)
                lvl = levels[i]
                
                # Tính số token sinh ra (chỉ phần assistant trả lời)
                num_tokens = len(output.outputs[0].token_ids)
                
                level_stats[lvl]["total"] += 1
                if pred_ans == ground_truths[i]:
                    level_stats[lvl]["correct"] += 1
                    level_stats[lvl]["tokens"] += num_tokens
            
            all_results[name] = level_stats
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")

    # 5. In báo cáo chi tiết (Dùng cho khóa luận)
    print("\n" + "="*85)
    print(f"{'LEVEL':<10} | {'ACCURACY (%)':<20} | {'AVG TOKENS (Correct)':<25}")
    print("="*85)

    for model_name, stats in all_results.items():
        print(f"\n--- MODEL: {model_name} ---")
        total_correct = 0
        total_samples = 0
        
        for lvl in range(1, 6):
            s = stats[lvl]
            acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
            avg_t = (s["tokens"] / s["correct"]) if s["correct"] > 0 else 0
            
            print(f"Level {lvl:<4} | {acc:>12.2f}% | {avg_t:>18.1f} tokens")
            
            total_correct += s["correct"]
            total_samples += s["total"]
        
        total_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
        print(f"{'OVERALL':<10} | {total_acc:>12.2f}% | (Trung bình trên {total_samples} câu)")

    print("="*85)

if __name__ == "__main__":
    main()