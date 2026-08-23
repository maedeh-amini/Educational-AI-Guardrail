
# =========================================================
# Graph Builder
# =========================================================

import os
import random
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from neo4j import GraphDatabase
from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
from neo4j_graphrag.indexes import create_vector_index

# =========================================================
# 1. Setup & Environment
# =========================================================
load_dotenv()

UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]
EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]
DB_NAME = os.environ.get("NEO4J_DATABASE", "graphdb")

class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text):
        return self.client.embeddings.create(
            input=text, 
            model=self.model,
            encoding_format=None
        ).data[0].embedding

    def embed_nodes(self, texts):
        response = self.client.embeddings.create(
            input=texts, 
            model=self.model,
            encoding_format=None
        )
        return [d.embedding for d in response.data]

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# =========================================================
# 2. Knowledge Graph Orchestrator
# =========================================================
class KnowledgeGraphOrchestrator:
    def __init__(self, driver, subset_fraction=1.0):
        self.driver = driver
        self.subset_fraction = subset_fraction
        self.processed_data = [] # List of dicts: {'id': ..., 'text': ...}

        self.llm = OpenAILLM(
            model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic", 
            api_key=UOS_API_KEY,
            base_url=UOS_API_BASE,
            model_params={"temperature": 0}
        )

        self.embedder = GatewayCompatibleEmbeddings(model="bge-m3")
        self.embedder.client.api_key = EMB_KEY
        self.embedder.client.base_url = EMB_BASE

    def load_source_data(self):
        dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
        subset_size = max(1, int(len(dataset) * self.subset_fraction))
        indices = random.sample(range(len(dataset)), subset_size)
        
        for idx in tqdm(indices, desc="Fetching HF Data"):
            row = dataset[idx]
            
            # CAPTURE THE ID HERE
            hf_id = str(row["id"]) 
            
            # Capture and clean the documents text
            docs = row["documents"]
            content = " ".join(docs) if isinstance(docs, list) else str(docs)
            
            self.processed_data.append({
                "hf_id": hf_id,
                "text": content
            })
        print(f"[INFO] Loaded {len(self.processed_data)} items with original IDs.")

    async def execute_pipeline(self):
        # 1. Verification of Index
        try:
            create_vector_index(
                self.driver,
                name="entity_vector_index",
                label="__Entity__",
                embedding_property="embedding",
                dimensions=1024, 
                similarity_fn="cosine"
            )
        except Exception:
            pass

        # 2. Initialize Pipeline
        kg_pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            neo4j_database=DB_NAME,
            embedder=self.embedder,
            from_pdf=False,
            schema="FREE", 
            on_error=OnError.IGNORE
        )

        print(f"[INFO] Building Graph & Injecting IDs into {DB_NAME}...")
        
        # We process one-by-one to maintain the ID mapping
        for item in tqdm(self.processed_data, desc="Processing Rows"):
            try:
                # A. Run Extraction
                await kg_pipeline.run_async(text=item['text'])
                
                # B. Immediate ID Injection
                # We find the node by text and set the hf_id property
                injection_query = """
                MATCH (c:Chunk) 
                WHERE c.text = $text AND c.hf_id IS NULL
                SET c.hf_id = $hf_id
                """
                
                with self.driver.session(database=DB_NAME) as session:
                    session.run(injection_query, text=item['text'], hf_id=item['hf_id'])
                
                # Gateway cooldown
                await asyncio.sleep(1.5) 
                
            except Exception as e:
                print(f"[ERROR] Row {item['hf_id']} failed: {e}")

# =========================================================
# 3. Execution
# =========================================================
if __name__ == "__main__":
    # IMPORTANT: Clear database before re-running to avoid ID mismatches
    # Command: MATCH (n) DETACH DELETE n
    
    orchestrator = KnowledgeGraphOrchestrator(neo4j_driver, subset_fraction=1.0
    )
    orchestrator.load_source_data()
    asyncio.run(orchestrator.execute_pipeline())
    print("[SUCCESS] Builder Pipeline Finished with ID Persistence.")








