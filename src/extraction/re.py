import os
from transformers import pipeline, AutoTokenizer

class RelationExtractor:
    def __init__(self, model_path: str, device: int = -1, threshold: float = 0.6):
        self.threshold = threshold
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"📥 Downloading/Loading RE Model: {model_path}...")
        
        # 1. Tải tokenizer gốc của SciBERT (Sạch và không bị lỗi JSON)
        # Nếu lúc train bạn dùng bản cased thì đổi thành 'allenai/scibert_scivocab_cased'
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased", 
            use_fast=True 
        )
        
        # 2. Bổ sung lại các token dùng để đánh dấu thực thể vào từ điển
        tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])
        
        # 3. Đưa tokenizer "đã được chữa bệnh" này vào Pipeline
        self.pipe = pipeline(
            'text-classification',
            model=model_path,
            tokenizer=tokenizer,
            device=device,
            token=hf_token
        )

    def extract(self, ents: list, context_text: str):
        relations = []
        context = context_text[:300]
        
        for i in range(len(ents)):
            for j in range(i+1, min(i+4, len(ents))):
                inp = f"[E1] {ents[i]['text']} [/E1] and [E2] {ents[j]['text']} [/E2] in: {context}"
                result = self.pipe(inp[:512])[0]
                
                if result['label'] != 'NO_RELATION' and result['score'] > self.threshold:
                    relations.append({
                        'head': ents[i]['text'],
                        'head_type': ents[i]['entity'],
                        'relation': result['label'],
                        'tail': ents[j]['text'],
                        'tail_type': ents[j]['entity'],
                        'score': round(result['score'], 3)
                    })
        return relations