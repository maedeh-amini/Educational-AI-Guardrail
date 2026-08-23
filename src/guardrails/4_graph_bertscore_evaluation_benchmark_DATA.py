import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import BERTScorer
from neo4j import GraphDatabase 

# Import your refined query components
from src.graph_query_2 import rag_chain

# =========================================================
# 1. Configuration & Setup
# =========================================================
load_dotenv()
# Make sure this matches your actual Neo4j DB name (default is usually "neo4j")
DB_NAME = os.environ.get("NEO4J_DATABASE", "graphdb") 

def get_ids_from_graph():
    """
    Fetches the chunk IDs that actually exist in the Neo4j database.
    """
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
    
    with driver.session(database=DB_NAME) as session:
        result = session.run(query)
        # Convert to strings to ensure matching works properly later
        ids = list(set([str(record["id"]) for record in result]))
    
    driver.close()
    print(f"[INFO] Found {len(ids)} unique document IDs in the '{DB_NAME}' database.")
    return ids

def load_test_dataset(filepath="ragbench_hotpotqa_50_sampled.jsonl"):
    """
    Load the downsampled dataset from the local JSONL file,
    and filter it to ONLY include questions present in the Graph DB.
    """
    # 1. Get the IDs that actually exist in Neo4j
    graph_ids = get_ids_from_graph()
    if not graph_ids:
        print("[WARNING] Graph is empty or no chunk IDs found! All questions will fail.")

    # 2. Read the JSONL file
    df = pd.read_json(filepath, lines=True)
    
    test_examples = []
    for _, row in df.iterrows():
        # 3. ONLY add the question if we have the data for it in Neo4j
        if str(row["id"]) in graph_ids:
            test_examples.append({
                "id": row["id"],
                "question": row["question"],
                "reference": row["response"]
            })
        
    print(f"[INFO] Kept {len(test_examples)} out of {len(df)} test examples (Present in Graph DB)")
    return test_examples

# =========================================================
# 2. BERTScore Evaluation Logic
# =========================================================
def run_evaluation(test_examples, output_prefix="bertscore_graph_BenchmarkData"):
    """
    Runs Graph RAG on test examples and computes BERTScore.
    Saves individual scores to CSV and JSONL, and overall averages to a summary JSON.
    """
    if not test_examples:
        print("[SKIP] No examples to evaluate. Check your Neo4j database ingestion.")
        return

    ids = []
    questions = []
    references = []
    predictions = []

    print(f"\n[START] Evaluating Graph RAG | Samples: {len(test_examples)}")
    
    for example in tqdm(test_examples, desc="Processing Graph queries"):
        q_id = example["id"]
        question = example["question"]
        reference = example["reference"]

        # Graph RAG query
        try:
            # --- FIX 2: Handle LangChain dictionary inputs ---
            # Different chains expect different input keys. We try the most common ones.
            try:
                raw_ans = rag_chain.invoke({"query": question})
            except Exception:
                try:
                    raw_ans = rag_chain.invoke({"question": question})
                except Exception:
                    try:
                        raw_ans = rag_chain.invoke({"input": question})
                    except Exception:
                        # Fallback to standard string if dictionaries fail
                        raw_ans = rag_chain.invoke(question)
            
            # Safely handle dictionary outputs from LangChain
            if isinstance(raw_ans, dict):
                ans = raw_ans.get("result", raw_ans.get("answer", raw_ans.get("output", str(raw_ans))))
            else:
                ans = str(raw_ans)
                
        except Exception as e:
            print(f"\n[ERROR] Row {q_id} failed: {e}")
            ans = "" 

        ids.append(q_id)
        questions.append(question)
        references.append(reference)
        predictions.append(ans)

    # Compute BERTScore
    print("\n[INFO] Computing BERTScore...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(model_type="distilroberta-base", lang="en", device=device)
    
    P, R, F1 = scorer.score(predictions, references)

    # Convert PyTorch tensors to standard Python lists
    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()

    # Create a DataFrame for individual scores
    results_df = pd.DataFrame({
        "Question_ID": ids,
        "Question": questions,
        "Reference": references,
        "Prediction": predictions,
        "BERTScore_Precision": p_list,
        "BERTScore_Recall": r_list,
        "BERTScore_F1": f1_list  # This is the primary "BERTScore"
    })
    
    # Save individual scores to BOTH CSV and JSONL
    csv_filename = f"{output_prefix}.csv"
    jsonl_filename = f"{output_prefix}.jsonl"
    
    results_df.to_csv(csv_filename, index=False)
    results_df.to_json(jsonl_filename, orient="records", lines=True)
    print(f"[INFO] Saved individual scores to {csv_filename} and {jsonl_filename}")

    # Calculate overall averages
    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()

    # Save the final overall scores to a separate summary file
    summary_data = {
        "Overall_Precision": avg_p,
        "Overall_Recall": avg_r,
        "Overall_F1": avg_f1
    }
    summary_filename = f"{output_prefix}_summary.json"
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[INFO] Saved overall summary scores to {summary_filename}")

    print("\n" + "="*45)
    print("FINAL STATS: GRAPH RAG EVALUATION")
    print("="*45)
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall:    {avg_r:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print("="*45 + "\n")

    return avg_p, avg_r, avg_f1

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(BASE_DIR, "src", "ragbench_hotpotqa_50_sampled.jsonl")
        
        print(f"[INFO] Loading data from: {file_path}")
        test_data = load_test_dataset(file_path)

        # --- PREVIEW LOGIC ---
        preview_size = 3
        
        if len(test_data) > preview_size:
            # 1. Grab the first 3 examples
            sample_data = test_data[:preview_size]
            
            print(f"\n[PREVIEW] Running initial sample of {preview_size} examples...")
            
            # Save the preview to a temporary file prefix
            run_evaluation(
                sample_data, 
                output_prefix="bertscore_graph_PREVIEW"
            )
            
            # 2. Pause and wait for user input
            proceed = input(f"\n[PROMPT] Review the preview in 'bertscore_graph_PREVIEW.csv'. Proceed with the full {len(test_data)} examples? (y/n): ")
            
            if proceed.strip().lower() not in ['y', 'yes']:
                print("[INFO] Evaluation aborted by user. Exiting...")
                sys.exit(0) # Stops the script here
                
        # 3. If the user said "yes" (or if the dataset has 3 or fewer items anyway), run the full set
        print(f"\n[INFO] Starting full evaluation for all {len(test_data)} rows...")
        run_evaluation(
            test_data, 
            output_prefix="bertscore_graph_BenchmarkData"
        )

    except Exception as e:
        print(f"[CRITICAL FAILURE]: {e}")


