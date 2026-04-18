import re

def accuracy_reward(completions, ground_truth, **kwargs):
    """Thưởng +1.0 nếu giải đúng Toán"""
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if match:
            pred = match.group(1).strip()
            # Làm sạch chuỗi trước khi so sánh
            pred_clean = pred.replace(" ", "").lower()
            truth_clean = truth.replace(" ", "").lower()
            
            if truth_clean != "" and pred_clean == truth_clean:
                rewards.append(1.0)
            elif truth_clean != "" and truth_clean in pred_clean:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Thưởng +0.5 nếu mở đóng thẻ đúng chuẩn"""
    rewards = []
    for completion in completions:
        if "<think>" in completion and "</think>" in completion and \
           "<answer>" in completion and "</answer>" in completion:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def length_penalty_reward(completions, level, **kwargs):

    rewards = []
    
    for i, completion in enumerate(completions):
        current_level = level[i] if isinstance(level, list) else level
        
        if "Level 1" in current_level:
            lambda_penalty = 0.01   
        elif "Level 2" in current_level:
            lambda_penalty = 0.005  
        else:
            lambda_penalty = 0.001  

        match = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
        if match:
            think_length = len(match.group(1).split())
            rewards.append(-lambda_penalty * think_length)
        else:
            rewards.append(0.0)
            
    return rewards