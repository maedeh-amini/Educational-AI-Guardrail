########################################################################################
####          Semantic RAG DeepEval Evaluation with Checkpointing & Preview         ####
########################################################################################
import os
import json
import random
import time
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# 1. Load configuration
load_dotenv(override=True)

# --- DeepEval Global Safety Settings (Disables Confident AI & Browser Popups) ---
os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "YES"
os.environ["CONFIDENT_OPEN_BROWSER"] = "0"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_ASYNC_MODE"] = "False"
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM

# Importing search components
from search import RAGSearch, LLMPipeline
from langchain_core.messages import HumanMessage

# -----------------------------
# Folder for results & Checkpoints
# -----------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "deepeval_generation_checkpoint_benchmark_semantic.json")
EVALUATION_CHECKPOINT = os.path.join(RESULTS_DIR, "ContextualRecall_evaluation_checkpoint.json")

PROMPT_TEMPLATE = "Answer the question based only on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:"

# --------------------------------------------------
# 1. Load Local JSONL Dataset
# --------------------------------------------------
def load_test_dataset(filepath="ragbench_hotpotqa_50_sampled.jsonl"):
    print(f"[INFO] Loading dataset from local file: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_json(filepath, lines=True)

    data = []
    for _, row in df.iterrows():
        data.append({
            "id": row.get("id", row.get("Question_ID", "unknown_id")),
            "question": row["question"],
            "reference": row["response"] if "response" in row else row["Reference"],
            "documents": row.get("documents", [])
        })

    print(f"[INFO] Loaded {len(data)} examples from {filepath}")
    return data

# --------------------------------------------------
# 2. Build DeepEval test cases with Checkpointing
# --------------------------------------------------
def build_test_cases(rag_search: RAGSearch, dataset):
    test_cases_data = []

    if os.path.exists(GENERATION_CHECKPOINT):
        try:
            with open(GENERATION_CHECKPOINT, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            for idx, item in enumerate(loaded_data):
                if "ground_truth_context" not in item or not item["ground_truth_context"]:
                    if idx < len(dataset):
                        item["ground_truth_context"] = dataset[idx].get("documents", [])
                test_cases_data.append(item)

            print(f"[INFO] Resuming test case generation. Loaded {len(test_cases_data)} pre-built cases.")
        except Exception as e:
            print(f"[WARNING] Failed to load generation checkpoint: {e}. Rebuilding...")

    start_idx = len(test_cases_data)

    for i, ex in enumerate(tqdm(dataset[start_idx:], desc="Building Semantic RAG test cases", initial=start_idx, total=len(dataset)), start=start_idx):
        try:
            rag_output_raw = rag_search.query_rag(ex["question"], top_k=5)
            rag_output = json.loads(rag_output_raw) if isinstance(rag_output_raw, str) else rag_output_raw

            retrieved_docs = rag_output.get("retrieved_docs", [])
            retrieved_texts = [
                doc.get("page_content", "")
                for doc in retrieved_docs
                if doc.get("page_content")
            ]
            answer = rag_output.get("answer", "")

            test_cases_data.append({
                "id": ex["id"],
                "input": ex["question"],
                "expected_output": ex["reference"],
                "actual_output": answer,
                "retrieval_context": retrieved_texts,
                "ground_truth_context": ex.get("documents", [])
            })

            with open(GENERATION_CHECKPOINT, "w", encoding="utf-8") as f:
                json.dump(test_cases_data, f, indent=2, ensure_ascii=False)

            time.sleep(1)

        except Exception as e:
            print(f"\n[SKIP] Row failed: {e}")
            time.sleep(2)
            continue

    return test_cases_data

# --------------------------------------------------
# 3. GPT-OSS Judge Model Wrapper
# --------------------------------------------------
class GPTOSSJudge(DeepEvalBaseLLM):
    def __init__(self, pipeline, model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic"):
        self.pipeline = pipeline
        self.model_name = model_name

    def load_model(self): return self.pipeline
    def get_model_name(self): return self.model_name
    def generate(self, prompt: str) -> str:
        response = self.pipeline.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# --------------------------------------------------
# 4. Evaluation Runner Function
# --------------------------------------------------
def run_evaluation(raw_cases, prefix="ContextualRecall"):
    test_cases = [
        LLMTestCase(
            input=tc["input"],
            expected_output=tc["expected_output"],
            actual_output=tc["actual_output"],
            retrieval_context=tc["retrieval_context"],
            context=tc.get("ground_truth_context", tc.get("context", []))
        ) for tc in raw_cases
    ]

    if not test_cases:
        print("[ERROR] No cases to evaluate.")
        return

    judge_pipeline = LLMPipeline(
        model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic",
        api_key=os.getenv("UOS_API_KEY"),
        api_base=os.getenv("UOS_API_BASE")
    )
    judge = GPTOSSJudge(judge_pipeline)

    # Note: Define threshold explicitly to use in our manual success check
    threshold = 0.7
    metrics = [ContextualRecallMetric(model=judge, threshold=threshold, include_reason=True)]   
    active_metric_names = "ContextualRecall"

    h_params = {
        "model": "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
        "temperature": 0,
        "retrieval_k": 5,
        "pipeline": "Semantic RAG",
        "prompt_template": PROMPT_TEMPLATE
    }

    all_test_results = []
    batch_size = 1

    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i : i + batch_size]
        print(f"\n>>> Processing Eval Batch {(i // batch_size) + 1}...")

        try:
            results_container = deepeval.evaluate(test_cases=batch, metrics=metrics, hyperparameters=h_params)
            current_batch_results = results_container.test_results if hasattr(results_container, 'test_results') else results_container
            all_test_results.extend(current_batch_results)
            time.sleep(2)
        except Exception as e:
            print(f"[BATCH ERROR] Failed at block {i}: {e}")
            time.sleep(5)
            # Append None to maintain index alignment
            all_test_results.append(None) 
            continue

    final_results_json = os.path.join(RESULTS_DIR, f"{prefix}_results.json")
    final_results_csv = os.path.join(RESULTS_DIR, f"{prefix}_results.csv")
    summary_file = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")

    ordered_results = []
    metric_scores_sum = 0.0
    metric_counts = 0
    total_passed_cases = 0

    for idx, result in enumerate(all_test_results):
        score = 0.0
        success = False
        
        if result: 
            # Extract score manually to handle potential naming variances
            for m_data in result.metrics_data:
                if m_data.name in ["Contextual Recall", "ContextualRecall"]:
                    score = getattr(m_data, "score", 0.0)
                    break
            
            # EXPLICIT SUCCESS CALCULATION
            success = score >= threshold
            
            if success:
                total_passed_cases += 1
                
            metric_scores_sum += score
            metric_counts += 1

        q_id = raw_cases[idx]["id"]
        
        ordered_results.append({
            "Question_ID": q_id,
            "Question": raw_cases[idx]["input"],
            "Reference": raw_cases[idx]["expected_output"],
            "Prediction": raw_cases[idx]["actual_output"],
            "ContextualRecall_Score": score,
            "Success": success,
            "Retrieval_Context": raw_cases[idx]["retrieval_context"]
        })

    with open(final_results_json, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)

    results_df = pd.DataFrame(ordered_results)
    results_df.to_csv(final_results_csv, index=False)
    print(f"[INFO] Saved individual results to {final_results_json} and {final_results_csv}")

    total_test_count = len(all_test_results)
    avg_ContextualRecall_score = (metric_scores_sum / metric_counts) if metric_counts > 0 else 0.0

    overall_metrics = {
        "overall_pass_rate": f"{(total_passed_cases / total_test_count) * 100:.2f}%" if total_test_count > 0 else "0.00%",
        "total_test_cases": total_test_count,
        "passed_test_cases": total_passed_cases,
        "failed_test_cases": total_test_count - total_passed_cases,
        "Average_ContextualRecall": avg_ContextualRecall_score
    }

    summary_data = {
        "summary_banner": f"Overall ContextualRecall Evaluation - {active_metric_names}",
        "hyperparameters": h_params,
        "overall_metrics": overall_metrics,
        "detailed_results_json": final_results_json,
        "detailed_results_csv": final_results_csv
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Run complete. Summary saved to: {summary_file}")
    print(f"\n" + "="*45)
    print("FINAL STATS: SEMANTIC RAG ContextualRecall")
    print("="*45)
    print(f"Average ContextualRecall Score: {avg_ContextualRecall_score:.4f}")
    print(f"Pass Rate: {overall_metrics['overall_pass_rate']}")
    print("="*45 + "\n")

# --------------------------------------------------
# 5. Main Execution Block with Preview Mechanism
# --------------------------------------------------
if __name__ == "__main__":
    rag_search = RAGSearch()

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(BASE_DIR, "src", "ragbench_hotpotqa_50_sampled.jsonl")
    if not os.path.exists(file_path):
        file_path = "ragbench_hotpotqa_50_sampled.jsonl"

    test_data = load_test_dataset(file_path) 
    raw_test_cases = build_test_cases(rag_search, test_data)

    # --- PREVIEW LOGIC ---
    preview_size = 3
    if len(raw_test_cases) > preview_size:
        sample_cases = raw_test_cases[:preview_size]
        print(f"\n[PREVIEW] Running initial sample of {preview_size} examples...")

        run_evaluation(sample_cases, prefix="ContextualRecall_PREVIEW")

        proceed = input(f"\n[PROMPT] Review the preview in 'results/ContextualRecall_PREVIEW_results.csv'. Proceed with the full {len(raw_test_cases)} examples? (y/n): ")
        if proceed.strip().lower() not in ['y', 'yes']:
            print("[INFO] Evaluation aborted by user. Exiting...")
            sys.exit(0)

    print(f"\n[INFO] Starting full evaluation for all {len(raw_test_cases)} rows...")
    run_evaluation(raw_test_cases, prefix="ContextualRecall_results")

    # Clean up checkpoints on successful full run
    for cf in [GENERATION_CHECKPOINT, EVALUATION_CHECKPOINT]:
        if os.path.exists(cf): 
            os.remove(cf)





