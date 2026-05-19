# Research Intelligence System: Agentic RAG for Scientific Claim Verification

An end-to-end Agentic Retrieval-Augmented Generation (RAG) pipeline designed to automatically search, extract, and verify complex scientific claims using Large/Small Language Models and Knowledge Graphs.

## Key Features

* Agentic Reasoning (LLM): Utilizes a local `Qwen2.5-1.5B-Instruct` model to evaluate scientific claims and output structured verdicts (SUPPORTED, REFUTED, or NEI - Not Enough Information) along with logical reasoning.
* Domain-Specific Information Extraction: Integrates a custom fine-tuned `SciBERT` model on the SciERC dataset to perform joint Named Entity Recognition (NER) and Relation Extraction (RE) on academic texts.
* Interactive Knowledge Graphs: Automatically constructs dynamic, visual relationship networks between scientific concepts (Methods, Metrics, Materials) using `NetworkX` and `PyVis`.
* Real-time Data Retrieval: Connects to the Semantic Scholar API to dynamically fetch real-world peer-reviewed papers as evidence.
* Full-Stack Architecture: Ready-to-use via Command Line (CLI), REST API (`FastAPI`), and a web-based dashboard (`Streamlit`).

---

## Architecture Workflow

1. Input: User provides a scientific claim (e.g., "Low-Rank Adaptation significantly reduces trainable parameters").
2. Retrieve: Agent queries the Semantic Scholar API to gather top-k relevant academic abstracts.
3. Extract: Fine-tuned SciBERT scans abstracts to identify entities and their relations.
4. Visualize: Extracted nodes and edges are compiled into an interactive HTML Knowledge Graph.
5. Verify: Qwen2.5 reads the raw abstracts and performs zero-shot reasoning to classify the claim and provide evidence-backed arguments.

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/QuocHuy1202/Research-Intelligence-System.git

cd research-intelligence-system
```

**2. Install dependencies**
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
**3. Environment Variables**
Create a .env file in the root directory and add your API keys:

*Required for real-time paper retrieval:* **S2_API_KEY**

*Required for downloading models from Hugging Face:* **HF_TOKEN**

## Usage Guide
**Method 1: Command Line Interface (CLI)**
Run the agent directly from your terminal for quick tests.

```bash
python -m cli.run "Transformer architectures cannot be parallelized during training compared to RNNs." --top_k 5
```

**Method 2: Web Interface (Streamlit)**

Launch the interactive dashboard. Note: You must start the Backend API first.

*Terminal 1 (Start Backend API):*

```bash
uvicorn api.main:app --reload
```
*Terminal 2 (Start Frontend UI):*

```bash
streamlit run frontend/app.py
```
*Open your browser at http://localhost:8501 to use the system.*