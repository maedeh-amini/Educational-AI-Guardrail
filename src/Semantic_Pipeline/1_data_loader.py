import random
from tqdm import tqdm
from datasets import load_dataset

def load_source_data(subset_fraction: float = 1.0):
    """
    Loads data from HuggingFace and returns a list of processed dictionaries.
    """
    # 1. Load the dataset
    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    
    # 2. Calculate subset size
    subset_size = max(1, int(len(dataset) * subset_fraction))
    indices = random.sample(range(len(dataset)), subset_size)
    
    processed_data = []
    
    # 3. Process rows
    for idx in tqdm(indices, desc="Fetching HF Data"):
        row = dataset[idx]
        
        # Capture the ID
        hf_id = str(row.get("id", idx)) 
        
        # Capture and clean the documents text
        docs = row.get("documents", [])
        content = " ".join(docs) if isinstance(docs, list) else str(docs)
        
        processed_data.append({
            "hf_id": hf_id,
            "text": content
        })
    
    return processed_data

# Example usage
if __name__ == "__main__":
    # Pass a float for the fraction of data you want (e.g., 0.05 for 5%)
    docs = load_source_data(subset_fraction=1.0)
    
    print(f"\n[INFO] Loaded {len(docs)} documents.")
    if docs:
        print("Example document ID:", docs[0]["hf_id"])
        print("Example text snippet:", docs[0]["text"][:100], "...") 



