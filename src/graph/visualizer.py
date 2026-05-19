import networkx as nx
from pyvis.network import Network

def visualize_graph(G: nx.DiGraph, output_file: str = 'kg_visualization.html'):
    """Xuất đồ thị NetworkX ra file HTML tương tác bằng PyVis"""
    net = Network(height='600px', width='100%', directed=True, cdn_resources='in_line')
    
    for node, attrs in G.nodes(data=True):
        net.add_node(node, label=node[:25],
                     color=attrs.get('color', '#97C2FC'),
                     size=attrs.get('size', 10),
                     title=f"Type: {attrs.get('type','?')}\nNode: {node}")
                     
    for src, dst, attrs in G.edges(data=True):
        net.add_edge(src, dst, label=attrs.get('relation',''),
                     title=attrs.get('relation',''), width=1.5)
                     
    net.set_options("""
    var options = {
        "physics": {"enabled": true, "stabilization": {"iterations": 100}},
        "edges": {"arrows": {"to": {"enabled": true}}},
        "interaction": {"hover": true, "tooltipDelay": 100}
    }""")
    
    net.save_graph(output_file)
    return output_file