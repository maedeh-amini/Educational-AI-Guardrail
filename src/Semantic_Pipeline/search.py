import os
import json
from typing import List
from dotenv import load_dotenv

# Standard LangChain integrations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Project internal imports
from data_loader import load_source_data
from vectorstore import FaissVectorStore

load_dotenv(override=True)

# -----------------------------
# 1. Custom GPT-OSS wrapper
# -----------------------------
class LLMPipeline:
    """
    Wrapper for self-hosted GPT-OSS using LangChain's ChatOpenAI 
    to handle headers and authentication more robustly.
    """
    def __init__(self, model_name: str, api_key: str, api_base: str, temperature: float = 0.0):
        # We use ChatOpenAI which is more robust against 403 errors than raw requests
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base.rstrip('/') + "/", # Ensures standard pathing
            temperature=temperature
        )

    def generate(self, messages: List[HumanMessage]):
        """
        Accepts a list of LangChain messages and returns a mock Res object 
        to maintain compatibility with your original search logic.
        """
        response = self.llm.invoke(messages)
        assistant_text = response.content

        # Keeping your existing class-based return structure to avoid breaking search_and_summarize
        class Gen:
            def __init__(self, text): self.text = text
        class Res:
            def __init__(self, text): self.generations = [[Gen(text)]]
        
        return Res(assistant_text)

# -----------------------------
# 2. RAG Search class
# -----------------------------
class RAGSearch:
    def __init__(self,
                 persist_dir: str = "faiss_vector_store",
                 embedding_model: str = "test/bge-m3",
                 llm_model: str = "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
                 api_key_env: str = "UOS_API_KEY",
                 api_base_env: str = "UOS_API_BASE"):

        # Load pre-built FAISS vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        print(f"[INFO] FAISS vector store loaded from '{persist_dir}' using {embedding_model}")

        self.api_key = os.getenv(api_key_env)
        self.api_base = os.getenv(api_base_env)
        
        if not self.api_key or not self.api_base:
            raise ValueError(f"API key or base URL not found in environment.")

        # Initialize the refactored LLM pipeline
        self.llm = LLMPipeline(
            model_name=llm_model,
            api_key=self.api_key,
            api_base=self.api_base
        )
        print(f"[INFO] LLM initialized using model: {llm_model}")

        # Path logic: prompt is in the folder above 'src'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        prompt_path = os.path.join(root_dir, "prompts", "rag_prompt.txt")
        
        if not os.path.exists(prompt_path):
             raise FileNotFoundError(f"Could not find prompt template at: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print(f"[INFO] Loaded prompt template.")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."

        prompt = self.prompt_template.format(query=query, context=context)
        response = self.llm.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text

    def query_rag(self, question: str, top_k: int = 3) -> str:
        """
        DeepEval-compatible output format.
        """
        results = self.vectorstore.query(question, top_k=top_k)
        retrieved_docs = []
        context_chunks = []

        for r in results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            if text:
                retrieved_docs.append({"page_content": text})
                context_chunks.append(text)

        context = "\n\n".join(context_chunks)
        if not context:
            answer = "No relevant documents found."
        else:
            prompt = self.prompt_template.format(query=question, context=context)
            response = self.llm.generate([HumanMessage(content=prompt)])
            answer = response.generations[0][0].text

        return json.dumps({
            "answer": answer,
            "retrieved_docs": retrieved_docs
        })

# -----------------------------
# Execution
# -----------------------------
if __name__ == "__main__":
    # Point to the directory created by vectorstore.py
    rag_search = RAGSearch(persist_dir="faiss_vector_store")

    query = "Which music group has the most members, DC Talk, or Manchester Orchestra?"

    print("\n--- Running Standard RAG ---")
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)

    print("\n--- Running DeepEval-compatible Query ---")
    ragas_output = rag_search.query_rag(query, top_k=1)
    print("Output:", ragas_output)



