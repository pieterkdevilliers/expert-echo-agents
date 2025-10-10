# tools/rag_tool.py
import os
import json
import requests
import chromadb
from typing import AsyncGenerator, Dict, List, Union
from fastapi.responses import StreamingResponse
from pydantic_ai import Tool
from dotenv import load_dotenv
from openai import OpenAI

from util_agents import rephrase_user_query as rephrase_agent

load_dotenv()


class RAGTool(Tool):
    name = "rag"
    description = "Retrieval-Augmented Generation tool for answering questions from stored knowledge bases."

    def __init__(self):
        super().__init__()
        self.environment = os.getenv("ENVIRONMENT")
        self.chroma_endpoint = os.getenv("CHROMA_ENDPOINT")
        self.headers = {
            "X-Chroma-Token": os.getenv("CHROMA_SERVER_AUTHN_CREDENTIALS"),
            "Content-Type": "application/json",
        }
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # 🧩 Embedding helper
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        response = self.openai_client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]

    # 🧠 RAG core method (asynchronous streaming)
    async def run(
        self,
        query: str,
        account_unique_id: str,
        relevance_score: float = 0.2,
        k_value: int = 7,
        sources_returned: int = 3,
        chat_history=None,
        prompt_text=None,
        temperature=0.2,
        scoreapp_report_text=None,
        user_products_prompt="",
    ) -> AsyncGenerator[Dict, None]:
        """
        Main async generator for RAG query streaming.
        Yields chunks of model output.
        """
        from shared_utils.query_source_data import (
            OpenAIEmbeddingManager,
            ChromaDBManager,
            search_db_advanced,
        )

        manager = ChromaDBManager(
            environment=self.environment,
            chroma_endpoint=self.chroma_endpoint,
            headers=self.headers,
        )

        embedding_manager = OpenAIEmbeddingManager()
        prepared_db = manager.get_or_create_collection(account_unique_id, embedding_manager)

        # Reuse your existing search_db_advanced generator
        async for chunk in search_db_advanced(
            manager=manager,
            db=prepared_db,
            query=query,
            relevance_score=relevance_score,
            k_value=k_value,
            sources_returned=sources_returned,
            account_unique_id=account_unique_id,
            visitor_email="",
            chat_history=chat_history,
            prompt_text=prompt_text,
            temperature=temperature,
            scoreapp_report_text=scoreapp_report_text,
            user_products_prompt=user_products_prompt,
        ):
            yield chunk
