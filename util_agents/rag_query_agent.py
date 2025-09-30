import logfire
import os
import shared_utils.query_source_data as rag_agent
from pydantic_ai import Agent
from schemas.agent_schemas import Query
from dotenv import load_dotenv

load_dotenv()

logfire.configure()  
logfire.instrument_pydantic_ai()

CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
ENVIRONMENT = os.environ.get('ENVIRONMENT')

embedding_manager = rag_agent.OpenAIEmbeddingManager()


async def query_rag_query_agent(query: Query):
    """
    Run a RAG query on the ChromaDB
    """
    manager = rag_agent.ChromaDBManager(
        environment=ENVIRONMENT,
        chroma_endpoint=CHROMA_ENDPOINT,
        headers={}
    )

    prepared_db = manager.get_or_create_collection(
        query.account_unique_id,
        embedding_manager
    )

    print("prepared_db:", prepared_db)
    return prepared_db