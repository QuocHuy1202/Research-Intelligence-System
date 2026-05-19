import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

VERIFY_SYSTEM_PROMPT = """You are a scientific claim verifier. Given a claim and evidence from papers,
classify the claim as exactly one of:
- SUPPORTED: evidence clearly supports the claim
- REFUTED: evidence contradicts the claim
- NEI: Not Enough Information

Respond in JSON format: {"verdict": "SUPPORTED|REFUTED|NEI", "reasoning": "<1-2 sentences>", "evidence_used": ["<sentence snippet>"]}"""

class ClaimVerifier:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()

    def _build_prompt(self, claim: str, evidence_texts: list) -> str:
        evidence_str = '\n'.join([f'[{i+1}] {e[:300]}' for i, e in enumerate(evidence_texts[:5])])
        return f"Claim: {claim}\n\nEvidence:\n{evidence_str}\n\nVerdict:"

    @torch.inference_mode()
    def verify(self, claim: str, evidence_texts: list, max_new_tokens: int = 256):
        user_content = self._build_prompt(claim, evidence_texts)
        messages = [
            {'role': 'system', 'content': VERIFY_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Sửa cảnh báo 1: Thêm truncation=True và max_length=2048
        inputs = self.tokenizer([text], return_tensors='pt', truncation=True, max_length=2048).to(self.device)
        
        # Sửa cảnh báo 2: Ép top_p và top_k về None để đồng bộ với do_sample=False
        output = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            temperature=None, 
            top_p=None, 
            top_k=None,
            repetition_penalty=1.1
        )
        
        generated = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse JSON output
        try:
            match = re.search(r'\{.*\}', generated, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
            
        # Fallback mechanism
        for v in ['SUPPORTED', 'REFUTED', 'NEI']:
            if v in generated.upper():
                return {'verdict': v, 'reasoning': generated[:200], 'evidence_used': []}
                
        return {'verdict': 'NEI', 'reasoning': generated[:200], 'evidence_used': []}