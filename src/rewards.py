import re

def extract_math_answer(text: str) -> str:
    """Hàm trích xuất đáp án nằm trong thẻ \boxed{} của LaTeX"""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    return match.group(1).strip() if match else None

def math_binary_reward(prompts, completions, answer, **kwargs):
    """Phần thưởng thưa thớt: Đúng +1, Sai -1 (Ép DRO hoạt động)"""
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        pred = extract_math_answer(completion[0]['content'])
        if pred == ground_truth:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

def dynamic_format_reward(prompts, completions, level, **kwargs):
    """Phần thưởng định dạng kết hợp Adaptive Compute Penalty"""
    rewards = []
    for completion, lvl in zip(completions, level):
        content = completion[0]['content']
        score = 0.0
        
        # Mồi nhử (Reward Shaping) cho SLM 1.5B
        if "<think>" in content and "</think>" in content:
            score += 0.5
            
        # Áp dụng Dynamic Penalty (Phạt Overthinking)
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            think_length = len(think_match.group(1))
            
            # Map hệ số phạt theo Level (Độ khó tĩnh)
            if "Level 1" in lvl:
                lambda_val = 0.010  # Dễ -> Phạt nặng, ép nghĩ nhanh
            elif "Level 2" in lvl:
                lambda_val = 0.005  # Vừa -> Phạt trung bình
            elif "Level 3" in lvl:
                lambda_val = 0.001  # Khó -> Phạt rất nhẹ, cho phép nghĩ sâu
            else:
                lambda_val = 0.005
                
            score -= (lambda_val * think_length)
            
        rewards.append(score)
    return rewards

def standard_format_reward(prompts, completions, **kwargs):
    """
    Phần thưởng định dạng tĩnh (Truyền thống).
    Chỉ kiểm tra mô hình có tuân thủ format không, không phạt độ dài (Length Bias).
    """
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        score = 0.0
        
        # Thưởng 1.0 điểm nếu có đầy đủ cấu trúc suy luận và đáp án
        if "<think>" in content and "</think>" in content and "<answer>" in content and "</answer>" in content:
            score += 1.0
            
        rewards.append(score)
    return rewards