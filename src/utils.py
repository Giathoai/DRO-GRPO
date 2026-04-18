import re

SYSTEM_PROMPT = """Bạn là một chuyên gia toán học xuất chúng. Bạn PHẢI giải bài toán theo định dạng nghiêm ngặt sau đây:
Bước 1: Suy luận và nháp bằng tiếng Việt bên trong thẻ <think> và </think>.
Bước 2: Chỉ ghi kết quả con số hoặc công thức cuối cùng bên trong thẻ <answer> và </answer>.
Tuyệt đối không vi phạm định dạng này."""

def extract_boxed_answer(text):
    """Hàm phụ trợ móc đáp án nằm trong \boxed{...} của bộ MATH"""
    match = re.search(r'\\boxed{(.*?)}', str(text))
    if match:
        return match.group(1)
    return str(text).strip()