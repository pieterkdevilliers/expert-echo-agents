import logfire
import os
import json
import shared_utils.query_source_data as rag_agent
from pydantic_ai import Agent
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


# New streaming endpoint
async def query_rag_query_agent(query: Query):
    """
    Streaming RAG query endpoint (replacing non-streaming version)
    """
    print('received query inside query_rag_query_agent: ', query)
    
    manager = rag_agent.ChromaDBManager(
        environment=ENVIRONMENT,
        chroma_endpoint=CHROMA_ENDPOINT,
        headers=headers,
    )

    prepared_db = manager.get_or_create_collection(
        query.account_unique_id,
        embedding_manager
    )
    
    async def generate():
        try:
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
                prompt_text=query.prompt,
                temperature=query.temperature,
                scoreapp_report_text=query.scoreapp_report_text,
                user_products_prompt=query.user_products_prompt
            ):
                # Send as Server-Sent Events format
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_chunk = {
                "type": "error",
                "content": f"Streaming error: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
