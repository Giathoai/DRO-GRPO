import re
from collections import defaultdict

def extract_math_answer(text):
    """Trích xuất đáp án của MÔ HÌNH nằm trong thẻ <answer>...</answer>"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        ans = match.group(1).strip()
        ans = ans.replace(",", "").replace("$", "").replace(" ", "")
        return ans
    return None

def extract_ground_truth(text):
    """Trích xuất đáp án CHUẨN từ dataset (nằm trong \boxed{...})"""
    text = str(text)
    # Ưu tiên 1: Lấy trong thẻ \boxed{}
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        ans = match.group(1).strip()
    else:
        # Fallback: Nếu không có \boxed, lấy khối chữ cuối cùng
        ans = text.split()[-1] if text.split() else text
    
    return ans.replace(",", "").replace("$", "").replace(" ", "").strip()

def math_binary_reward(prompts, completions, answer, **kwargs):
    """Reward 1: Kiểm tra tính đúng đắn của đáp án (Binary 0/1)"""
    rewards = []
    for completion, correct_ans in zip(completions, answer):
        content = completion[0]['content'] if isinstance(completion, list) else completion
        pred = extract_math_answer(content)
        clean_gt = extract_ground_truth(correct_ans)
        
        if pred is not None and pred == clean_gt:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def standard_format_reward(prompts, completions, **kwargs):
    """Reward 2: Kiểm tra định dạng cơ bản"""
    rewards = []
    for completion in completions:
        content = completion[0]['content'] if isinstance(completion, list) else completion
        
        score = 0.0
        if "<think>" in content and "</think>" in content:
            score += 0.5
        if "<answer>" in content and "</answer>" in content:
            score += 0.5
            
        rewards.append(score)
    return rewards

def dynamic_format_reward(prompts, completions, **kwargs):
    """
    Reward 3: In-batch Adaptive Length Penalty (DRO-GRPO).
    Mô hình tự cạnh tranh: Phạt những câu trả lời dài hơn mức trung bình 
    của các câu đúng định dạng trong cùng một nhóm sinh (batch).
    """
    # Khởi tạo mảng rewards mặc định
    rewards = [0.0] * len(completions)
    
    # BƯỚC 1: Nhóm các câu trả lời theo từng câu hỏi (Prompt)
    # GRPOTrainer trả về 1 mảng phẳng, ta cần nhóm num_generations câu lại với nhau
    groups = defaultdict(list)
    for i, prompt in enumerate(prompts):
        p_str = str(prompt)
        groups[p_str].append(i)
        
    # BƯỚC 2: Xử lý chấm điểm và phạt cho từng câu hỏi
    for p_str, indices in groups.items():
        group_contents = []
        group_format_scores = []
        
        # 2.1 Đánh giá định dạng của từng câu trong nhóm
        for idx in indices:
            completion = completions[idx]
            content = completion[0]['content'] if isinstance(completion, list) else completion
            group_contents.append(content)
            
            format_score = 0.0
            if "<think>" in content and "</think>" in content and "<answer>" in content and "</answer>" in content:
                format_score = 1.0
            group_format_scores.append(format_score)
            
        # 2.2 Tính "Chuẩn mực độ dài" (Baseline) của nhóm
        # Lấy chiều dài của những câu trả lời CÓ ĐỊNH DẠNG ĐÚNG
        valid_lengths = [len(content) for content, score in zip(group_contents, group_format_scores) if score > 0]
        
        # Chuẩn mực = Trung bình cộng chiều dài của các câu đúng.
        # (Bạn cũng có thể dùng min(valid_lengths) để ép nó học theo câu ngắn nhất)
        baseline_len = sum(valid_lengths) / len(valid_lengths) if valid_lengths else 0
        
        # 2.3 Áp dụng điểm thưởng và phạt
        for i, idx in enumerate(indices):
            content = group_contents[i]
            format_score = group_format_scores[i]
            
            penalty = 0.0
            # Chỉ phạt khi câu này đúng format VÀ nhóm có tiêu chuẩn (baseline_len > 0)
            if format_score > 0 and baseline_len > 0:
                actual_len = len(content)
                if actual_len > baseline_len:
                    # Công thức: Trừ 0.01 điểm cho mỗi 100 ký tự dài hơn mức trung bình của nhóm
                    excess = actual_len - baseline_len
                    penalty = min(0.5, (excess / 100.0) * 0.01)
            
            final_score = max(0.0, format_score - penalty)
            rewards[idx] = final_score
            
    return rewards