import argparse
from pprint import pprint
from src.agent import ResearchIntelligenceAgent
from src.graph.visualizer import visualize_graph
from dotenv import load_dotenv
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Research Intelligence Agent CLI")
    parser.add_argument("claim", type=str, help="The scientific claim to verify")
    parser.add_argument("--top_k", type=int, default=5, help="Number of papers to fetch")
    parser.add_argument("--graph_out", type=str, default="output_graph.html", help="Path to save the Knowledge Graph HTML")
    
    args = parser.parse_args()

    print(f"Analyzing Claim: '{args.claim}'...")
    agent = ResearchIntelligenceAgent()
    
    # Chạy Agent
    result = agent.run(claim=args.claim, top_k=args.top_k)
    
    print("\n" + "="*50)
    print(f"VERDICT: {result['verdict']}")
    print("="*50)
    print(f"Reasoning:\n{result['reasoning']}\n")
    
    print("Citations:")
    for i, citation in enumerate(result['citations'], 1):
        print(f"  [{i}] {citation['title']} ({citation['year']})")
        
    #print(f"\nKnowledge Graph saved to: {args.graph_out}")

if __name__ == "__main__":
    main()