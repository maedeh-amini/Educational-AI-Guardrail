import os
import faiss
import numpy as np
import pickle
from typing import List, Any
# Remove SentenceTransformer import
from embedding import EmbeddingPipeline # Assuming neighbors in /src

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "test/bge-m3"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        # 1. Initialize our new API-based Pipeline
        self.embedding_model = embedding_model
        self.emb_pipe = EmbeddingPipeline(model_name=self.embedding_model)
        
        print(f"[INFO] VectorStore initialized with API model: {embedding_model}")

    def build_from_documents(self, documents: List[dict]):
        """
        Builds the store using block chunks (no overlap).
        """
        print(f"[INFO] Building vector store from {len(documents)} raw items...")

        # 2. Use the new block-based chunking (one row = one block)
        chunks = self.emb_pipe.create_block_chunks(documents)

        # 3. Generate embeddings via API
        embeddings = self.emb_pipe.embed_chunks(chunks)

        # Store metadata (keeping original text and any existing IDs if present)
        metadatas = []
        for i, chunk in enumerate(chunks):
            # We try to preserve the hf_id from the data_loader if available
            hf_id = documents[i].get("hf_id", "N/A") if i < len(documents) else "N/A"
            metadatas.append({"text": chunk, "hf_id": hf_id})

        # 4. Add to FAISS (Ensuring float32 for compatibility)
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]

        if self.index is None:
            # FlatL2 is standard, but BGE-M3 works great with Inner Product (Cosine) too
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors (Dim: {dim}) to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
        else:
            print("[WARNING] No existing index found to load.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index is empty. Please load or build the index first.")

        D, I = self.index.search(query_embedding, top_k)
        results = []

        for idx, dist in zip(I[0], D[0]):
            if idx == -1: continue # Faiss returns -1 if not enough results
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": idx,
                "distance": float(dist),
                "metadata": meta
            })
        return results

    def query(self, query_text: str, top_k: int = 5):
        """
        Processes a string query using the API embedder.
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        
        # 5. Use the API embedder instead of the old local model
        # We wrap in a list because the embedder expects a list of strings
        query_emb = self.emb_pipe.embed_chunks([query_text]).astype("float32")

        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from data_loader import load_source_data

    # Load full dataset or subset
    docs = load_source_data(subset_fraction=1.0)

    # Initialize store (This will handle the API embedding internally)
    store = FaissVectorStore(persist_dir="faiss_vector_store", embedding_model="test/bge-m3")

    # Build and save
    store.build_from_documents(docs)

    # Test Query
    sample_query = "Who is John Rankin Rogers?"
    results = store.query(sample_query, top_k=3)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Dist: {res['distance']:.4f}):")
        print(f"Text: {res['metadata']['text'][:200]}...")



