import logfire
import os
import json
import shared_utils.query_source_data as rag_agent
from agents.expert_agent import expert_agent
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, AsyncGenerator
from fastapi.responses import StreamingResponse
from schemas.agent_schemas import Query
from core.config import settings
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
    logfire.instrument_pydantic_ai()
else:
    logfire.instrument_pydantic_ai()

CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
ENVIRONMENT = os.environ.get('ENVIRONMENT')
CHROMA_SERVER_AUTHN_CREDENTIALS = os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS')
headers = {
    'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
    'Content-Type': 'application/json'
}

embedding_manager = rag_agent.OpenAIEmbeddingManager()



# Separate function that actually executes the RAG agent
# This is called by the endpoint after expert_agent makes the decision
async def execute_rag_agent(query: Query) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Creates and runs a dedicated RAG agent with custom prompt.
    This is called by the endpoint when expert_agent selects RAG.
    
    Yields streaming chunks from the RAG agent.
    """
    print("🚀 Executing dedicated RAG agent with custom prompt")
    print(f"📝 Custom prompt preview: {query.prompt[:100]}..." if len(query.prompt) > 100 else query.prompt)
    
    # Initialize ChromaDB manager
    manager = rag_agent.ChromaDBManager(
        environment=ENVIRONMENT,
        chroma_endpoint=CHROMA_ENDPOINT,
        headers=headers,
    )

    # Get or create collection
    prepared_db = manager.get_or_create_collection(
        query.account_unique_id,
        embedding_manager
    )
    
    # Execute search_db_advanced which creates its own RAG agent internally
    # with the custom prompt from query.prompt
    async for chunk in rag_agent.search_db_advanced(
        manager=manager,
        db=prepared_db,
        query=query.query,
        relevance_score=query.relevance_score,
        k_value=query.k_value,
        sources_returned=query.sources_returned,
        account_unique_id=query.account_unique_id,
        visitor_email=query.visitor_email,
        chat_history=query.chat_history,
        prompt_text=query.prompt,  # ⭐ Custom RAG prompt passed here
        temperature=query.temperature,
        scoreapp_report_text=query.scoreapp_report_text,
        user_products_prompt=query.user_products_prompt
    ):
        yield chunk