import re
import torch
import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import random
# Thư viện chấm điểm tương đương toán học của Qwen
from math_verify import parse, verify

# ==========================================
# ⚙️ CẤU HÌNH ĐÁNH GIÁ (TÙY CHỈNH TẠI ĐÂY)
# ==========================================
MAX_SAMPLES = 1  # Đổi thành None nếu bạn muốn chạy toàn bộ 5000 câu
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MODELS_TO_EVAL = [
    {
        "display_name": "3. SFT + Standard GRPO (Baseline)", 
        "adapter_name": "baseline_grpo",
        "lora_path": "outputs/Qwen-1.5B-Standard-GRPO-Final"
    },
    {
        "display_name": "4. SFT + DRO-GRPO (Proposed)", 
        "adapter_name": "proposed_dro_grpo",
        "lora_path": "outputs/Qwen-1.5B-DRO-GRPO-Final"
    }
]
# ==========================================

def extract_answer(text):
    """Trích xuất đáp án từ thẻ <answer> hoặc \boxed{}"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        ans = match.group(1).strip()
    else:
        match = re.search(r"\\boxed\{(.*?)\}", text)
        ans = match.group(1).strip() if match else text.strip()
    
    return ans.strip()

def check_correctness(pred, gt):
    """Chấm điểm kết hợp Exact Match và Tương đương Toán học"""
    pred_clean = pred.replace(" ", "")
    gt_clean = gt.replace(" ", "")
    
    # 1. Khớp chuỗi tuyệt đối
    if pred_clean == gt_clean:
        return True
        
    # 2. Tương đương toán học bằng math_verify
    try:
        parsed_pred = parse(pred)
        parsed_gt = parse(gt)
        return verify(parsed_pred, parsed_gt)
    except Exception:
        return False

def main():
    print(f"⏳ Đang nạp Base Model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    configs = [
        'algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    print("⏳ Đang tải và gộp các tập test từ Vietnamese-MATH...")
    test_splits = []
    for cfg in configs:
        try:
            ds = load_dataset("ura-hcmut/Vietnamese-MATH", cfg, split="test")
            test_splits.append(ds)
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không tìm thấy tập test cho {cfg}")
    
    if not test_splits:
        print("❌ Lỗi: Không thể tải bất kỳ tập dữ liệu test nào!")
        return

    dataset = concatenate_datasets(test_splits)
    
    # Cắt giảm số lượng mẫu nếu có cấu hình MAX_SAMPLES
    if MAX_SAMPLES is not None:
        num_samples = min(MAX_SAMPLES, len(dataset))
        dataset = dataset.select(range(num_samples))
        print(f"✅ Đã chuẩn bị xong tập test: GIỚI HẠN CHẠY {num_samples} MẪU.")
    else:
        print(f"✅ Đã chuẩn bị xong tập test: CHẠY TOÀN BỘ {len(dataset)} MẪU.")
    
    all_results = {}
    eval_model = None

    for model_info in MODELS_TO_EVAL:
        display_name = model_info["display_name"]
        adapter_name = model_info["adapter_name"]
        lora_path = model_info["lora_path"]
        
        # Tạo tên file log
        log_filename = f"eval_log_{adapter_name}.txt"
        # Xóa file log cũ nếu đã tồn tại để ghi mới từ đầu
        if os.path.exists(log_filename):
            os.remove(log_filename)
        
        print(f"\n🚀 Đang nạp Adapter cho: {display_name}")
        
        try:
            if eval_model is None:
                eval_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=adapter_name)
            else:
                eval_model.load_adapter(lora_path, adapter_name=adapter_name)
            
            eval_model.set_adapter(adapter_name)
            eval_model.eval()

            level_stats = {l: {"correct": 0, "total": 0, "tokens": 0} for l in range(1, 6)}
            
            for i, item in enumerate(tqdm(dataset, desc=f"Đánh giá {display_name}")):
                try:
                    question = item.get("problem", "")
                    
                    raw_gt = item.get("solution", item.get("answer", ""))
                    if "\\boxed{" in raw_gt:
                        gt = raw_gt.split("\\boxed{")[-1].split("}")[0]
                    else:
                        gt = raw_gt
                    gt = gt.strip()
                    
                    lvl_match = re.search(r"\d+", str(item.get("level", "1")))
                    lvl = int(lvl_match.group()) if lvl_match else 1
                    if lvl not in level_stats: lvl = 1

                    prompt = (
                        f"<|im_start|>system\nYou are a smart mathematical assistant. "
                        f"Please reason step-by-step inside <think> tags and put your final answer inside <answer> tags.<|im_end|>\n"
                        f"<|im_start|>user\n{question}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )

                    inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)
                    input_length = inputs.input_ids.shape[1]

                    with torch.no_grad():
                        output_ids = eval_model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )

                    generated_ids = output_ids[0][input_length:]
                    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    num_tokens = len(generated_ids)
                    pred_ans = extract_answer(pred_text)
                    
                    level_stats[lvl]["total"] += 1
                    
                    is_correct = check_correctness(pred_ans, gt)
                    
                    if is_correct:
                        level_stats[lvl]["correct"] += 1
                        level_stats[lvl]["tokens"] += num_tokens

                    # ==========================================
                    # GHI LOG TỪNG CÂU RA FILE TEXT
                    # ==========================================
                    with open(log_filename, "a", encoding="utf-8") as f:
                        f.write(f"--- [MẪU {i+1} | LEVEL {lvl}] ---\n")
                        f.write(f"📌 Câu hỏi: {question}\n")
                        f.write(f"🎯 Ground Truth: '{gt}'\n")
                        f.write(f"🤖 Model Trích xuất: '{pred_ans}'\n")
                        f.write(f"✅ Kết quả chấm: {'ĐÚNG' if is_correct else 'SAI'}\n")
                        f.write(f"📝 Raw Output:\n{pred_text}\n")
                        f.write("="*80 + "\n\n")

                    del inputs, output_ids
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️ Bỏ qua mẫu thứ {i+1} do lỗi: {e}")
                    with open(log_filename, "a", encoding="utf-8") as f:
                        f.write(f"--- [MẪU {i+1} | LEVEL {lvl}] - BỊ LỖI ---\nLỗi: {e}\n" + "="*80 + "\n\n")
                    continue
            
            all_results[display_name] = level_stats
            print(f"✅ Đã đánh giá xong {display_name}. Log được lưu tại: {log_filename}")
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {display_name}: {e}")

    # ==========================================
    # IN BẢNG BÁO CÁO TỔNG KẾT
    # ==========================================
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
        print(f"{'OVERALL':<10} | {total_acc:>12.2f}% | (Tổng cộng {total_samples} mẫu)")

    print("="*85)

if __name__ == "__main__":
    main()