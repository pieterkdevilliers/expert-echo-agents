import logfire
import os
import shared_utils.query_source_data as rag_agent
from pydantic_ai import Agent
from schemas.agent_schemas import Query
from core.config import settings
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
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


async def query_rag_query_agent(query: Query):
    """
    Run a RAG query on the ChromaDB
    """
    print('received query: ', query)
    manager = rag_agent.ChromaDBManager(
        environment=ENVIRONMENT,
        chroma_endpoint=CHROMA_ENDPOINT,
        headers=headers,
    )

    prepared_db = manager.get_or_create_collection(
        query.account_unique_id,
        embedding_manager
    )
    search_result = await rag_agent.search_db_advanced(
        manager=manager,
        db=prepared_db,
        query=query.query,
        relevance_score=query.relevance_score,
        k_value=query.k_value,
        sources_returned=query.sources_returned,
        account_unique_id=query.account_unique_id,
        visitor_email=query.visitor_email,
        chat_history=query.chat_history,
        prompt_text=query.prompt,
        temperature=query.temperature,
        scoreapp_report_text=query.scoreapp_report_text,
        user_products_prompt=query.user_products_prompt)
    
    return search_result