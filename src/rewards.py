import re
import statistics
from collections import defaultdict
from math_verify import parse, verify
# =====================================================================
# 1. PRE-COMPILED REGEX
# =====================================================================
MATH_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GT_BOXED_PATTERN = re.compile(r'\\boxed\{(.*?)\}')

# Pattern chặt chẽ: Ép mở/đóng thẻ đúng thứ tự, không có rác ở ngoài
STRICT_FORMAT_PATTERN = re.compile(r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$", re.DOTALL)

# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def clean_math_str(text):
    if not text:
        return ""
    return str(text).replace(",", "").replace("$", "").replace(" ", "").strip()

def get_content(completion):
    return completion[0]['content'] if isinstance(completion, list) else completion

# =====================================================================
# 3. EXTRACTION FUNCTIONS
# =====================================================================
def extract_math_answer(text):
    match = MATH_ANSWER_PATTERN.search(text)
    return clean_math_str(match.group(1)) if match else None

def extract_ground_truth(text):
    text_str = str(text)
    match = GT_BOXED_PATTERN.search(text_str)
    
    if match:
        raw_ans = match.group(1)
    else:
        parts = text_str.split()
        raw_ans = parts[-1] if parts else text_str
        
    return clean_math_str(raw_ans)

# =====================================================================
# 4. REWARD FUNCTIONS ĐÃ ĐƯỢC PHÂN TÁCH TRÁCH NHIỆM
# =====================================================================

def math_binary_reward(prompts, completions, answer, **kwargs):
    """
    Reward 1: Điểm Toán học (1.0 hoặc 0.0)
    Đã nâng cấp: Sử dụng math_verify để kiểm tra tính tương đương toán học.
    """
    rewards = []
    for comp, ans in zip(completions, answer):
        content = get_content(comp)
        pred = extract_math_answer(content)
        gt = extract_ground_truth(ans)
        
        # Nếu mô hình không sinh ra thẻ <answer> hoặc không có nội dung
        if pred is None:
            rewards.append(0.0)
            continue
            
        # Thử khớp chuỗi tuyệt đối trước (tối ưu tốc độ cho các trường hợp dễ)
        if pred == gt:
            rewards.append(1.0)
            continue
            
        # Nếu không khớp chuỗi, dùng thư viện để chấm điểm tương đương
        try:
            parsed_pred = parse(pred)
            parsed_gt = parse(gt)
            
            # verify() trả về True nếu hai biểu thức tương đương toán học
            if verify(parsed_pred, parsed_gt):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # Bắt lỗi nếu mô hình sinh ra chuỗi kỳ lạ khiến thư viện không thể parse
            # (VD: text thường, ký tự rác, cú pháp LaTeX bị hỏng nặng)
            rewards.append(0.0)
            
    return rewards

def strict_format_reward(prompts, completions, **kwargs):
    """
    Reward 2: Điểm Định dạng (1.0 hoặc Soft Reward).
    Thay thế cho standard_format_reward lỏng lẻo cũ.
    """
    rewards = []
    for comp in completions:
        content = get_content(comp)
        
        # Chấm 1.0 nếu đúng form chuẩn
        if STRICT_FORMAT_PATTERN.search(content):
            rewards.append(1.0)
        else:
            # Chấm điểm khuyến khích (Soft reward) nếu có thẻ nhưng sai thứ tự/có rác
            score = 0.0
            if "<think>" in content and "</think>" in content:
                score += 0.25
            if "<answer>" in content and "</answer>" in content:
                score += 0.25
            rewards.append(score)
            
    return rewards

def dynamic_length_penalty(prompts, completions, **kwargs):
    """
    Reward 3: Điểm Phạt Độ Dài (0.0 đến -0.5).
    Chỉ đóng vai trò TRỪ ĐIỂM đối với những câu trả lời đúng format nhưng dài dòng.
    """
    rewards = [0.0] * len(completions)
    
    groups = defaultdict(list)
    for i, prompt in enumerate(prompts):
        groups[str(prompt)].append(i)
        
    for p_str, indices in groups.items():
        group_think_lengths = [] 
        group_is_valid = []
        
        # 1. Trích xuất độ dài phần <think>
        for idx in indices:
            content = get_content(completions[idx])
            match = STRICT_FORMAT_PATTERN.search(content)
            
            if match:
                group_think_lengths.append(len(match.group(1))) # Chỉ đếm nội dung trong <think>
                group_is_valid.append(True)
            else:
                group_think_lengths.append(0)
                group_is_valid.append(False)
            
        # 2. Tính chuẩn mực (Median) dựa trên các mẫu hợp lệ
        valid_lengths = [
            length for is_valid, length in zip(group_is_valid, group_think_lengths) 
            if is_valid
        ]
        baseline_len = statistics.median(valid_lengths) if valid_lengths else 0
            
        # 3. Tính điểm phạt (Trả về số ÂM)
        for local_i, idx in enumerate(indices):
            actual_len = group_think_lengths[local_i]
            is_valid = group_is_valid[local_i]
            
            penalty = 0.0
            if is_valid and baseline_len > 0 and actual_len > baseline_len:
                ratio = actual_len / baseline_len
                # Phạt tối đa -0.5
                penalty = min(0.5, (ratio - 1.0) * 0.2)
                
            rewards[idx] = -penalty  # CHÚ Ý: Dấu ÂM để trừ vào tổng điểm
            
    return rewards