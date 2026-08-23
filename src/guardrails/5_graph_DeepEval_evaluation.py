########################################################################################
####       Graph RAG DeepEval Evaluation with Checkpointing & Preview         ####
########################################################################################
import os
import json
import random
import time
import sys
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# 1. Load configuration
load_dotenv(override=True)

# --- DeepEval Global Rate-Limit Safety Settings ---
os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "YES"
os.environ["CONFIDENT_OPEN_BROWSER"] = "0"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_ASYNC_MODE"] = "False"
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

# =========================================================
# 2. Custom Embedding Provider
# =========================================================
class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
        kwargs.pop("encoding_format", None)
        kwargs.pop("chunk_size", None)
        responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
        return [d.embedding for d in responses.data]

# =========================================================
# 3. Component Initialization
# =========================================================
embeddings = GatewayCompatibleEmbeddings(
    api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
    base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
    model=os.environ["EMBEDDING_MODEL"]
)

llm = ChatOpenAI(
    api_key=os.environ["UOS_API_KEY"],
    base_url=os.environ["UOS_API_BASE"],
    model=os.environ["UOS_Model"],
    temperature=0
)

# =========================================================
# 4. Graph Retrieval Strategy
# =========================================================
retrieval_query = """
OPTIONAL MATCH (node)-[rel]->(neighbor)
WHERE NOT type(rel) IN ['NEXT_CHUNK', 'PREVIOUS_CHUNK', 'PARENT', 'PART_OF_DOCUMENT', 'SOURCE']
WITH node, score, 
     collect(DISTINCT type(rel) + ' -> ' + coalesce(neighbor.name, neighbor.id, neighbor.title, '')) AS raw_relations
WITH node, score,
     [r IN raw_relations WHERE NOT r ENDS WITH ' -> '] AS relations
RETURN 
    coalesce(node.text, '') + 
    CASE WHEN size(relations) > 0 THEN '\n\nGRAPH RELATIONSHIPS:\n' + apoc.text.join(relations, ' | ') ELSE '' END AS text,
    score, 
    {source: coalesce(node.source, node.hf_id, 'Unknown')} AS metadata
"""

vector_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
    index_name="chunk_vector_index", 
    search_type="vector",  
    retrieval_query=retrieval_query
)

# =========================================================
# 5. RAG Chain Construction (LCEL)
# =========================================================
template = """Answer the question based only on the provided context, 
which includes text chunks and graph relationships from the knowledge graph:

{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No relevant information found in the database."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

retriever = vector_db.as_retriever(search_kwargs={'k': 5})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# 6. Data Handling & Checkpointing
# =========================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "graph_benchmark_deepeval_generation_checkpoint.json")

def load_test_dataset(filepath):
    print(f"[INFO] Loading dataset from local file: {filepath}")
    df = pd.read_json(filepath, lines=True)
    data = []
    for idx, row in df.iterrows():
        data.append({
            "id": str(row.get("id", row.get("Question_ID", f"q_{idx}"))),
            "question": row.get("question", row.get("Question", "")),
            "reference": row.get("response", row.get("Reference", row.get("reference", ""))),
            "documents": row.get("documents", [])
        })
    return data

def build_test_cases(dataset):
    test_cases_data = []
    if os.path.exists(GENERATION_CHECKPOINT):
        try:
            with open(GENERATION_CHECKPOINT, "r", encoding="utf-8") as f:
                test_cases_data = json.load(f)
        except: pass
    
    start_idx = len(test_cases_data)
    for row in tqdm(dataset[start_idx:], desc="Generating Graph RAG Test Cases"):
        try:
            docs = retriever.invoke(row["question"])
            retrieved_texts = [doc.page_content for doc in docs if doc.page_content]
            answer = rag_chain.invoke(row["question"])
            
            test_cases_data.append({
                "id": row["id"], "input": row["question"], "expected_output": row["reference"],
                "actual_output": answer, "retrieval_context": retrieved_texts,
                "ground_truth_context": row.get("documents", [])
            })
            with open(GENERATION_CHECKPOINT, "w", encoding="utf-8") as f:
                json.dump(test_cases_data, f, indent=2, ensure_ascii=False)
            time.sleep(1)
        except Exception as e:
            print(f"\n[SKIP] ID {row['id']} failed: {e}")
            continue
    return test_cases_data

# =========================================================
# 7. GPT-OSS Judge Model Wrapper
# =========================================================
class GPTOSSJudge(DeepEvalBaseLLM):
    def __init__(self, pipeline): self.pipeline = pipeline
    def load_model(self): return self.pipeline
    def get_model_name(self): return os.getenv("UOS_Model", "RedHatAI/gemma-4-31B-it-FP8-Dynamic")
    def generate(self, prompt: str) -> str:
        return self.pipeline.invoke([HumanMessage(content=prompt)]).content
    async def a_generate(self, prompt: str) -> str:
        return (await self.pipeline.ainvoke([HumanMessage(content=prompt)])).content

# =========================================================
# 8. Evaluation Runner Function
# =========================================================
def run_evaluation(raw_cases, prefix="AnswerRelevancy_graph_bechmark"):
    judge_pipeline = llm # Using the same initialized LLM
    judge = GPTOSSJudge(judge_pipeline)
    threshold = 0.7
    metric = AnswerRelevancyMetric(model=judge, threshold=threshold, include_reason=True)
    
    all_test_results = []
    for i, tc in tqdm(enumerate(raw_cases), desc="Evaluating Graph RAG Cases", total=len(raw_cases)):
        test_case = LLMTestCase(
            input=tc["input"], expected_output=tc["expected_output"],
            actual_output=tc["actual_output"], retrieval_context=tc["retrieval_context"],
            context=tc.get("ground_truth_context", [])
        )
        try:
            res = deepeval.evaluate(test_cases=[test_case], metrics=[metric])
            all_test_results.append(res.test_results[0])
            time.sleep(2)
        except Exception as e:
            print(f"\n[ERROR] Failed at item {i}: {e}")
            all_test_results.append(None)
            time.sleep(2)

    ordered_results = []
    metric_scores_sum = 0.0
    metric_counts = 0
    total_passed_cases = 0

    for idx, result in enumerate(all_test_results):
        score = 0.0
        success = False
        if result:
            for m in result.metrics_data:
                if m.name in ["Answer Relevancy", "AnswerRelevancy"]:
                    score = m.score
                    break
            success = score >= threshold
            if success: total_passed_cases += 1
            metric_scores_sum += score
            metric_counts += 1
        
        ordered_results.append({
            "Question_ID": raw_cases[idx]["id"], "Question": raw_cases[idx]["input"],
            "Reference": raw_cases[idx]["expected_output"], "Prediction": raw_cases[idx]["actual_output"],
            "AnswerRelevancy_Score": score, "Success": success,
            "Retrieval_Context": raw_cases[idx]["retrieval_context"]
        })

    final_results_json = os.path.join(RESULTS_DIR, f"{prefix}_results.json")
    final_results_csv = os.path.join(RESULTS_DIR, f"{prefix}_results.csv")
    summary_file = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")

    with open(final_results_json, "w", encoding="utf-8") as f: json.dump(ordered_results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(ordered_results).to_csv(final_results_csv, index=False)

    total_test_count = len(all_test_results)
    summary_data = {
        "summary_banner": f"Overall Graph AnswerRelevancy Evaluation",
        "hyperparameters": {
            "model": os.getenv("UOS_Model"), "temperature": 0, "retrieval_k": 5,
            "pipeline": "Graph RAG", "prompt_template": template
        },
        "overall_metrics": {
            "overall_pass_rate": f"{(total_passed_cases / total_test_count) * 100:.2f}%" if total_test_count > 0 else "0.00%",
            "total_test_cases": total_test_count,
            "passed_test_cases": total_passed_cases,
            "failed_test_cases": total_test_count - total_passed_cases,
            "Average_AnswerRelevancy": (metric_scores_sum / metric_counts) if metric_counts > 0 else 0.0
        },
        "detailed_results_json": final_results_json,
        "detailed_results_csv": final_results_csv
    }
    with open(summary_file, "w", encoding="utf-8") as f: json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Run complete. Summary: {summary_file}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Benchmark_Data_DeepEval.jsonl")

    test_data = load_test_dataset(file_path) 
    raw_test_cases = build_test_cases(test_data)
    
    if len(raw_test_cases) > 3:
        run_evaluation(raw_test_cases[:3], prefix="AnswerRelevancy_graph_benchmark_PREVIEW")
        if input("\nProceed with full evaluation? (y/n): ").lower() != 'y': 
            exit()
            
    run_evaluation(raw_test_cases, prefix="AnswerRelevancy_graph_benchmark_results")




