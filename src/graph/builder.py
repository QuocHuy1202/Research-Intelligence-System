import networkx as nx

ENTITY_COLOR = {
    'Method': '#4E79A7', 'Task': '#F28E2B', 'Metric': '#E15759',
    'Material': '#76B7B2', 'Generic': '#59A14F', 'OtherScientificTerm': '#EDC948',
    'UNKNOWN': '#B07AA1'
}

def build_knowledge_graph(processed_papers: list) -> nx.DiGraph:
    """Xây dựng NetworkX Directed Graph từ các thực thể và quan hệ"""
    G = nx.DiGraph()
    
    for paper in processed_papers:
        paper_node = f"PAPER::{paper['title'][:40]}"
        G.add_node(paper_node, type='paper', color='#FF9DA7', size=20)
        
        # Thêm các Node thực thể (Entities)
        for ent in paper.get('entities', []):
            ent_node = ent['text'].lower()
            if ent_node not in G:
                G.add_node(ent_node, type=ent['entity'],
                           color=ENTITY_COLOR.get(ent['entity'], '#B07AA1'), size=10)
            G.add_edge(paper_node, ent_node, relation='MENTIONS', weight=1)
            
        # Thêm các Cạnh quan hệ (Relations)
        for rel in paper.get('relations', []):
            h = rel['head'].lower()
            t = rel['tail'].lower()
            if h not in G:
                G.add_node(h, type=rel['head_type'],
                           color=ENTITY_COLOR.get(rel['head_type'], '#B07AA1'), size=10)
            if t not in G:
                G.add_node(t, type=rel['tail_type'],
                           color=ENTITY_COLOR.get(rel['tail_type'], '#B07AA1'), size=10)
            G.add_edge(h, t, relation=rel['relation'], weight=rel['score'])
            
    return G