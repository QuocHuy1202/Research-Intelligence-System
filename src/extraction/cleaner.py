def clean_entity_text(word: str) -> str:
    """Làm sạch các sub-word token của BERT (như ##)"""
    return word.replace('##', '').strip()

def is_valid_entity(word: str, score: float, threshold: float = 0.5) -> bool:
    """Kiểm tra xem thực thể có hợp lệ không"""
    if len(word) < 2:
        return False
    if score <= threshold:
        return False
    return True