# eval.py
import re
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def extract_ground_truth(answer_str):
    if "####" in answer_str:
        return answer_str.split("####")[1].strip().replace(",", "")
    return None

def extract_prediction(pred_str):
    match = re.search(r"<answer>(.*?)</answer>", pred_str, re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")
    
    numbers = re.findall(r"-?\d+(?:\.\d+)?", pred_str.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def main():
    # 1. Base Model
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    models_to_eval = [
        {"name": "3. SFT + Standard GRPO (Baseline)", "lora_path": "outputs/Qwen-1.5B-Standard-GRPO"},
        {"name": "4. SFT + DRO-GRPO (Proposed)", "lora_path": "outputs/Qwen-1.5B-DRO-GRPO"}
    ]

    print("⏳ Đang tải tập dữ liệu GSM8K (Test split)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    prompts = []
    ground_truths = []
    
    for item in dataset:
        question = item["question"]
        gt = extract_ground_truth(item["answer"])
        
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
        ground_truths.append(gt)

    print(f"🚀 Khởi tạo vLLM Engine với model gốc: {base_model_id}")
    llm = LLM(
        model=base_model_id, 
        enable_lora=True, 
        max_lora_rank=16, 
        max_model_len=2048,
        tensor_parallel_size=1
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    results_summary = {}

    for idx, model_info in enumerate(models_to_eval):
        name = model_info["name"]
        lora_path = model_info["lora_path"]
        
        print(f"\n==================================================")
        print(f" Đang đánh giá: {name}")
        print(f"==================================================")
        
        try:
            if lora_path is None:
                outputs = llm.generate(prompts, sampling_params)
            else:
                lora_request = LoRARequest(str(idx), idx + 1, lora_path)
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            
            correct = 0
            total = len(outputs)
            
            for i, output in enumerate(outputs):
                pred_str = output.outputs[0].text
                pred_ans = extract_prediction(pred_str)
                
                if pred_ans == ground_truths[i]:
                    correct += 1
                    
            accuracy = (correct / total) * 100
            results_summary[name] = accuracy
            print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {name}: Thư mục LoRA '{lora_path}' chưa tồn tại. Bỏ qua.")
            print(f"Chi tiết lỗi: {e}")

    print("\n" + "="*50)
    print(" 📊 TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ GSM8K ")
    print("="*50)
    for name, acc in results_summary.items():
        print(f"{name.ljust(40)}: {acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()