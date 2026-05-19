import yaml
import torch
from typing import Dict, Any

from src.extraction.ner import NERExtractor
from src.extraction.re import RelationExtractor
from src.search.semantic_scholar import SemanticScholarAgent
from src.graph.builder import build_knowledge_graph
from src.verification.verifier import ClaimVerifier

class ResearchIntelligenceAgent:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        device = 0 if torch.cuda.is_available() else -1
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize modules
        self.search_agent = SemanticScholarAgent()
        self.ner_extractor = NERExtractor(self.config['models']['ner_model_path'], device=device)
        self.re_extractor = RelationExtractor(self.config['models']['re_model_path'], device=device)
        self.verifier = ClaimVerifier(self.config['models']['qwen_model_name'], device=device_str)

    def _process_paper(self, paper: dict):
        text = paper.get('abstract') or paper.get('title') or ''
        if not text.strip():
            return paper | {'entities': [], 'relations': []}
            
        ents = self.ner_extractor.extract(text)
        rels = self.re_extractor.extract(ents, text)
        return paper | {'entities': ents, 'relations': rels}

    def run(self, claim: str, top_k: int = None) -> Dict[str, Any]:
        top_k = top_k or self.config['search']['top_k']
        
        # 1. Search
        papers = self.search_agent.search_and_extract(claim, top_k=top_k)
        
        # 2. Extract NER/RE
        processed_papers = [self._process_paper(p) for p in papers]
        
        # 3. Graph
        G = build_knowledge_graph(processed_papers)
        
        # 4. Verify
        evidence_texts = [p['abstract'] for p in papers if p.get('abstract')]
        verification_result = self.verifier.verify(claim, evidence_texts)
        
        # Format response
        citations = [{
            'title': p['title'], 'year': p['year'], 'authors': p['authors']
        } for p in papers]

        return {
            'claim': claim,
            'verdict': verification_result.get('verdict', 'NEI'),
            'reasoning': verification_result.get('reasoning', ''),
            'citations': citations,
            'graph_nodes': G.number_of_nodes()
        }