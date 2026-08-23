import json
import pandas as pd
from tqdm import tqdm
from bert_score import score
from search import RAGSearch

def load_test_dataset(filepath="ragbench_hotpotqa_50_sampled.jsonl"):
    """
    Load the downsampled dataset from the local JSONL file.
    Returns a list of dictionaries: {"id": ..., "question": ..., "reference": ...}
    """
    df = pd.read_json(filepath, lines=True)
    
    test_examples = []
    for _, row in df.iterrows():
        test_examples.append({
            "id": row["id"],
            "question": row["question"],
            "reference": row["response"]
        })
        
    print(f"[INFO] Loaded {len(test_examples)} test examples from {filepath}")
    return test_examples


def evaluate_rag_with_bertscore(rag_search: RAGSearch, test_examples, output_prefix="bertscore_semantic_BenchmarkData"):
    """
    Runs RAG on test examples and computes BERTScore.
    Saves individual scores to CSV and JSONL for ANOVA, and saves overall averages to a summary JSON.
    """
    ids = []
    questions = []
    references = []
    predictions = []

    for example in tqdm(test_examples, desc="Evaluating RAG on test set"):
        q_id = example["id"]
        question = example["question"]
        reference = example["reference"]

        # RAG query
        rag_output = rag_search.query_rag(question, top_k=5)
        
        # Parse the JSON response
        try:
            rag_output_json = json.loads(rag_output)
            predicted_answer = rag_output_json.get("answer", "")
        except json.JSONDecodeError:
            predicted_answer = rag_output

        ids.append(q_id)
        questions.append(question)
        references.append(reference)
        predictions.append(predicted_answer)

    # Compute BERTScore
    print("\n[INFO] Computing BERTScore...")
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        model_type="distilroberta-base",
        batch_size=32, 
        rescale_with_baseline=False
    )
    
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
    
    # 1 & 2. Save individual scores to BOTH CSV and JSONL
    csv_filename = f"{output_prefix}.csv"
    jsonl_filename = f"{output_prefix}.jsonl"
    
    results_df.to_csv(csv_filename, index=False)
    results_df.to_json(jsonl_filename, orient="records", lines=True)
    print(f"[INFO] Saved individual scores to {csv_filename} and {jsonl_filename}")

    # Calculate overall averages
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    # 3. Save the final overall scores to a separate summary file
    summary_data = {
        "Overall_Precision": avg_precision,
        "Overall_Recall": avg_recall,
        "Overall_F1": avg_f1
    }
    summary_filename = f"{output_prefix}_summary.json"
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[INFO] Saved overall summary scores to {summary_filename}")

    print(f"\n[RESULT] OVERALL BERTScore - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_search = RAGSearch()

    # Load test data from the JSONL file generated earlier
    test_data = load_test_dataset("ragbench_hotpotqa_50_sampled.jsonl")

    # Evaluate and save results
    evaluate_rag_with_bertscore(
        rag_search, 
        test_data, 
        output_prefix="bertscore_semantic_BenchmarkData"
    )



