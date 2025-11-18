# 🚀 LocalRAG — RAG Pipeline for Customer Service on local Hardware

Welcome to _LocalRAG_ — a focused, fast, and modular Retrieval-Augmented Generation (RAG) pipeline built around manual documentation and a curated expert dataset. This repository contains everything you need to preprocess manuals, build datasets (including synthetic LLM-generated examples), experiment with embeddings and rerankers, evaluate RAG systems, and run a local prototype chat interface.

✨ Why this repo is exciting
- ⚡ Fast experiments: small, reproducible preprocessing and dataset generation pipelines.
- 🔬 Research-ready: tools for embedding model comparison, reranking, and end-to-end RAG evaluation.
- 🧪 Prototype-ready: a local chatbot prototype (Docker-friendly) to demo the system and run user studies.

⚙️ Repository layout
- `1_preproc/` — Preprocessing: HTML/manual parsing and chunking for retrieval.
- `2_datasets/` — Datasets: expert dataset, synthetic LLM datasets (train/test), and generation scripts.
- `3_retrieval/` — Retrieval experiments: embedding comparisons, lightweight fine-tuning, and advanced retrieval techniques.
- `4_RAG/` — RAG evaluation: system evaluation scripts and experiment notebooks using the expert dataset.
- `5_prototype/` — Prototype: Dockerfile, `LLM.py`, and a Streamlit demo (`streamlit.py`) to run a local chatbot interface.

🛠️ Project pipeline (high level)
1. 🧩 Convert manual HTML pages into retrieval-friendly text chunks (`1_preproc/`).
2. 🧠 Use those chunks to create an LLM dataset (including synthetic examples) for fine-tuning and evaluation (`2_datasets/`).
3. 🔎 Train and compare custom embedding models, add a reranker to boost top-result quality (`3_retrieval/`).
4. 📊 Evaluate RAG systems on the expert dataset and compare different configurations (`4_RAG/`).
5. 💬 Build a local prototype with a simple chat interface and test it on a workstation or inside Docker (`5_prototype/`).

💡 Notes & practical tips
- The `multilingual-e5-small` embedding model is fast and resource-efficient, but not always the best for top-tier accuracy — on stronger hardware consider models like `bge-m3`.
- 🔁 Reranking improves result quality significantly; for low-resource setups, compact cross-encoder models are a pragmatic choice.
- 🚧 Production readiness is not complete: improvements such as robust prompt engineering, query normalization, and deployment hardening are left as next steps.

📚 Key learnings
- 🧠 Choose chunk sizes thoughtfully — dynamic chunking helps keep related passages together.
- 🌐 Lightweight multilingual embedding models are effective for retrieval baselines.
- 🔧 Fine-tuning (including with synthetic data) can meaningfully boost retrieval performance.
- 🔄 Cross-encoder rerankers help reorder candidates for better downstream LLM responses.
- ⚖️ LLM-based components are constrained by available local hardware; cloud or stronger GPUs enable higher performance.

▶️ Quick start (Windows PowerShell)
1) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install requirements (prototype folder contains the app requirements):

```powershell
cd 5_prototype
pip install -r requirements.txt
```

3) Run the local prototype (Streamlit):

```powershell
cd 5_prototype
streamlit run streamlit.py
```

Enjoy exploring LocalRAG — build fast RAG experiments and ship prototypes quickly!