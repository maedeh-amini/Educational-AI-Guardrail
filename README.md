# Educational-AI-Guardrail: Safe RAG Pipelines for University Regulations

A Python-based framework and evaluation suite designed for building, comparing, and statistically analyzing Semantic and Knowledge Graph Retrieval-Augmented Generation (RAG) pipelines. This repository evaluates how different retrieval mechanisms handle complex reasoning and messy, real-world documents to mitigate LLM hallucinations.

# 🚀 Features

  * **Dual-Pipeline Architecture**: Implements dense vector retrieval (FAISS) alongside graph-based retrieval utilizing the `SimpleKGPipeline` from `neo4j_graphrag` for Cypher query traversals.
  * **Domain-Specific Evaluation**: Benchmarked on a standard dataset (HotpotQA split of RAGbench) as well as a custom dataset of 50 Q/A pairs derived directly from official Osnabrück University study regulations.
  * **Rigorous LLM-as-a-Judge Testing**: Integrated with DeepEval to assess retrieval (Contextual Precision, Contextual Recall) and generation (Faithfulness, Answer Relevancy, BERTScore).
  * **Advanced Statistical Analysis**: Employs an Aligned Rank Transform (ART) ANOVA via R's `ARTool` to compare the 2x2 mixed factorial design and validate the significance of system performances.
  * **Modern Tooling**: Uses `uv` for lightning-fast, deterministic dependency management.

# 📂 Project Structure

```plaintext
├── .deepeval/             # DeepEval configuration and test logs for RAG assessment
├── data/
│   ├── datasets/          # HotpotQA split and custom university Q/A pairs
│   └── vector_store/      # Local storage for embedded document chunks
├── faiss_store/           # FAISS index files (Add to .gitignore or Git LFS)
├── prompts/               # Centralized system and user prompt templates
├── src/                   
│   ├── vectorstore.py     # Semantic embeddings and FAISS management
│   ├── graphstore.py      # Neo4j and SimpleKGPipeline configuration
│   └── chatbot.py         # Application entry point for the UI evaluation
├── tests/
│   ├── test_cases.py      # DeepEval metric execution scripts
│   └── art_anova.R        # R script for ART ANOVA statistical evaluation
├── .env                   # Environment variables (API keys, DB endpoints)
├── .gitignore             # Git ignore rules
├── .python-version        # Specified Python runtime environment
├── main.py                # Main execution script to run both RAG pipelines
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Standard pip dependency list (exported from uv)
└── uv.lock                # UV lockfile for deterministic builds
