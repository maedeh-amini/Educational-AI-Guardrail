# Educational-AI-Guardrail: Safe RAG Pipelines for University Regulations

A Python-based framework and evaluation suite designed for building, comparing, and statistically analyzing Semantic and Knowledge Graph Retrieval-Augmented Generation (RAG) pipelines. This repository evaluates how different retrieval mechanisms handle complex reasoning and messy, real-world documents to mitigate LLM hallucinations.

# 🚀 Features

  * **Dual-Pipeline Architecture**: Implements dense vector retrieval (FAISS) alongside graph-based retrieval utilizing the `SimpleKGPipeline` from `neo4j_graphrag` for Cypher query traversals.
  * **Domain-Specific Evaluation**: Benchmarked on a standard dataset (HotpotQA split of RAGbench) as well as a custom dataset of 50 Q/A pairs derived directly from official Osnabrück University study regulations.
  * **Rigorous LLM-as-a-Judge Testing**: Integrated with DeepEval to assess retrieval (Contextual Precision, Contextual Recall) and generation (Faithfulness, Answer Relevancy, BERTScore).
  * **Advanced Statistical Analysis**: Employs an Aligned Rank Transform (ART) ANOVA via R's `ARTool` to compare the 2x2 mixed factorial design and validate the significance of system performances.
  * **Modern Tooling**: Uses `uv` for lightning-fast, deterministic dependency management.

# 📂 Project Structure

Educational-AI-Guardrail/
├── docs/
│   └── Project Report_ Maedeh Amini.pdf                # Detailed project documentation and report
├── plots/                                              # Generated metric visualization plots
│   ├── answer-relevancy/
│   ├── bertscore/
│   ├── contextual_precision/
│   ├── contextual-recall/
│   └── faithfulness/
├── src/
│   ├── guardrails/                                     # Knowledge Graph RAG implementation
│   │   ├── .env                                        # Environment variables (KG pipeline)
│   │   ├── .python-version                             # Python version specification
│   │   ├── 1_Knowledge_Graph_Pipeline.py               # KG construction and ingestion
│   │   ├── 2_graph_query.py                            # Cypher graph traversal logic
│   │   ├── 3_graph_utils.py                            # Helper utilities for Neo4j
│   │   ├── 4_graph_bertscore_evaluation_benchmark_DATA.py 
│   │   ├── 4b_graph_bertscore_evaluation_academic_DATA.py  
│   │   ├── 5_graph_DeepEval_evaluation.py              # DeepEval suite for KG pipeline
│   │   ├── pyproject.toml                              # Subpackage configuration
│   │   └── uv.lock                                     # Lockfile for environment reproducibility
│   └── Semantic_Pipeline/                              # Semantic dense vector RAG implementation
│       ├── faiss_vector_store/                         # Local FAISS index storage
│       ├── .env                                        # Environment variables (Semantic pipeline)
│       ├── .python-version                             # Python version specification
│       ├── 1_data_loader.py                            # Document ingestion and chunking logic
│       ├── 2_embedding.py                              # Embedding generation pipeline
│       ├── 3_vectorstore.py                            # FAISS vector store management
│       ├── 4_search.py                                 # Semantic similarity retrieval execution
│       ├── 5_bertscore_evaluation.py                   # BERTScore evaluation for semantic pipeline
│       ├── 6_DeepEval_evaluation.py                    # DeepEval suite for semantic pipeline
│       ├── pyproject.toml                              # Subpackage configuration
│       └── uv.lock                                     # Lockfile for environment reproducibility
├── tests/
│   ├── test_cases/                                     # R scripts for statistical hypothesis testing
│   │   ├── art_mixed_anova_answer_relevancy.R          
│   │   ├── art_mixed_anova_bertscore.R                 
│   │   ├── art_mixed_anova_contextual_precision.R      
│   │   ├── art_mixed_anova_contextual_recall.R         
│   │   └── art_mixed_anova_faithfulness.R              
│   └── test_results/                                   # Statistical outputs and test result logs
│       ├── answer_relevancy/
│       ├── bertscore/
│       ├── contextual_precision/
│       ├── contextual_recall/
│       └── faithfulness/
├── .gitignore                                          # Git ignore rules
├── bertscore_data.csv                                  # Compiled BERTScore evaluation data
├── DESIGN.md                                           # Architectural decisions and design rationale
├── LICENSE                                             # Repository open-source license
├── README.md                                           # Main repository overview and setup guide
└── requirements.txt                                    # Standalone Python dependencies list
