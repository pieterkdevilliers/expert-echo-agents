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
async def query_agent(request: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Agentic endpoint: Streams AI responses using expert_agent.
    Uses RAG as a tool internally, plus additional tools.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    async def generate():
        try:
            previous_text = ""
            sources = []

            # Run the agent stream
            async with expert_agent.run_stream(request.query, deps=request) as result:
                async for chunk in result.stream_text():
                    # chunk is cumulative text → extract only new part
                    new_text = chunk[len(previous_text):]
                    previous_text = chunk

                    if new_text:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': new_text})}\n\n"

                # Any sources returned by tools (like RAG)
                sources = result.metadata.get("sources", [])
                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            # Completion signal
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