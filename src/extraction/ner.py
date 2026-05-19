import os
from transformers import pipeline
from .cleaner import clean_entity_text, is_valid_entity

class NERExtractor:
    def __init__(self, model_path: str, device: int = -1, threshold: float = 0.5):
        self.threshold = threshold
        # Lấy token từ biến môi trường
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"Downloading/Loading NER Model: {model_path}...")
        self.pipe = pipeline(
            'token-classification',
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy='simple',
            device=device,
            token=hf_token
        )

    def extract(self, text: str):
        text = text[:512]
        raw_ents = self.pipe(text)
        results = []
        seen = set()
        
        for e in raw_ents:
            word = clean_entity_text(e['word'])
            if not is_valid_entity(word, e['score'], self.threshold) or word in seen:
                continue
                
            results.append({
                'entity': e['entity_group'],
                'text': word,
                'score': round(e['score'], 3)
            })
            seen.add(word)
            
        return results