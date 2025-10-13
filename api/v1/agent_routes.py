from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel
import core.auth as auth
from schemas.agent_schemas import Query, UserQuery, QueryRequest, AgentDeps
import util_agents.rag_query_agent as rag_agent
from shared_utils.query_source_data import search_db_advanced
import util_agents.rephrase_user_query as rephrase_agent
from fastapi.responses import StreamingResponse
from agents.expert_agent import expert_agent
import json


router = APIRouter()

# Import your existing components
from shared_utils.query_source_data import (
    ChromaDBManager,
    embedding_manager,
    CHAT_MODEL_NAME,
    ENVIRONMENT,
    CHROMA_ENDPOINT,
    CHROMA_SERVER_AUTHN_CREDENTIALS,
    headers
)

# Initialize ChromaDB Manager (singleton pattern)
chroma_manager = ChromaDBManager(
    environment=ENVIRONMENT,
    chroma_endpoint=CHROMA_ENDPOINT,
    headers=headers
)


@router.post("/query")
async def rag_query(query: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Accepts a query payload, runs it through the AI agent, and returns the streaming response.
    """
    if not query.query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    
    # Directly return the StreamingResponse
    return await rag_agent.query_rag_query_agent(query=query)


@router.post("/rephrase-user-query")
async def rephrase_user_query(query: UserQuery, authorized: bool = Depends(auth.verify_api_key)):
    """
    Rephrases a user query to be more specific and detailed.
    """
    if not query.query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    
    result = await rephrase_agent.rephrase_user_query(query=query.query)

    return result


# ==============================================================================
# ROUTER ENDPOINT WITH STREAMING SUPPORT
# ==============================================================================



@router.post("/query-agent")
async def query_agent_endpoint(query: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Agentic endpoint that streams the response back to RepoA (or any SSE client)
    Uses expert_agent, which can call RAG as a tool
    """

    # Convert Query to AgentDeps
    deps = AgentDeps(
        query=query.query,
        prompt=query.prompt,
        visitor_email=query.visitor_email,
        visitor_uuid=query.visitor_uuid,
        account_unique_id=query.account_unique_id,
        chat_history=query.chat_history,
        relevance_score=query.relevance_score,
        k_value=query.k_value,
        sources_returned=query.sources_returned,
        temperature=query.temperature,
        chat_session_id=query.chat_session_id,
        scoreapp_report_text=query.scoreapp_report_text,
        user_products_prompt=query.user_products_prompt,
        sources=[]
    )

    async def generate():
        try:
            full_text = ""
            sources = []

            # ✅ Wrap the agent stream fully inside the generator
            async with expert_agent.run_stream(query.query, deps=deps) as result:
                async for chunk in result.stream_text():
                    # chunk is cumulative, yield only new text
                    new_text = chunk[len(full_text):]
                    full_text = chunk

                    if new_text:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': new_text})}\n\n"

                # Optional: send sources if agent stored them
                if hasattr(result, "metadata") and result.metadata.get("sources"):
                    sources = result.metadata["sources"]
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )