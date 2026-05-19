import os
import requests

class SemanticScholarAgent:
    BASE_URL = 'https://api.semanticscholar.org/graph/v1'
    FIELDS = 'title,abstract,year,authors,citationCount,externalIds,tldr'

    def __init__(self):
        self.api_key = os.getenv("S2_API_KEY", "")
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    def search_and_extract(self, query: str, top_k: int = 5):
        url = f'{self.BASE_URL}/paper/search'
        params = {'query': query, 'fields': self.FIELDS, 'limit': top_k}
        
        r = requests.get(url, params=params, headers=self.headers, timeout=15)
        r.raise_for_status()
        papers = r.json().get('data', [])
        
        results = []
        for p in papers:
            abstract = p.get('abstract') or ''
            if not abstract and p.get('tldr'):
                abstract = p['tldr'].get('text', '')
                
            results.append({
                'paper_id': p.get('paperId', ''),
                'title': p.get('title', ''),
                'abstract': abstract,
                'year': p.get('year'),
                'citations': p.get('citationCount', 0),
                'authors': [a['name'] for a in p.get('authors', [])[:3]],
            })
        return results