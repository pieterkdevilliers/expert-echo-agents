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


# @router.post("/agent-query")
# # async def query_agent(query: Query):
# async def query_agent(query: str):
#     """
#     Unified endpoint for all agent interactions (RAG, Calendar, etc.)
#     The agent decides which tool to use.
#     """
#     result = await expert_agent.run(deps=query)

#     return result


# ==============================================================================
# ROUTER ENDPOINT WITH STREAMING SUPPORT
# ==============================================================================


@router.post("/query-agent")
async def query_agent(request: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Agentic endpoint - the agent decides whether to use RAG or other tools.
    Streams the response back to the client.
    """
    print(f"Query being sent for streaming: ", request)
    
    # Convert Query to AgentDeps
    deps = AgentDeps(
        query=request.query,
        prompt=request.prompt,
        visitor_email=request.visitor_email,
        visitor_uuid=request.visitor_uuid,
        account_unique_id=request.account_unique_id,
        chat_history=request.chat_history,
        relevance_score=request.relevance_score,
        k_value=request.k_value,
        sources_returned=request.sources_returned,
        temperature=request.temperature,
        chat_session_id=request.chat_session_id,
        scoreapp_report_text=request.scoreapp_report_text,
        user_products_prompt=request.user_products_prompt
    )
    
    async def generate():
        try:
            print("🤖 Starting agent stream...")
            
            # Track sources to send at the end
            sources = []
            
            async with expert_agent.run_stream(request.query, deps=deps) as result:
                print("📡 Agent stream opened")
                
                async for chunk in result.stream_text():
                    # Send text chunks as they arrive
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                
                # After streaming completes, check if we have sources in the result
                # The agent will have called the RAG tool and we can extract sources
                final_result = await result
                print(f"✅ Agent completed. Data: {final_result.data}")
                
                # Try to extract sources if available
                if hasattr(final_result, 'data') and isinstance(final_result.data, dict):
                    sources = final_result.data.get('sources', [])
                
                # Send sources if available
                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            print("🏁 Stream completed successfully")
            
        except Exception as e:
            print(f"❌ Stream error: {str(e)}")
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



# ==============================================================================
# ALTERNATIVE: Direct RAG endpoint (if you want to bypass the agent)
# ==============================================================================

@router.post("/rag-direct")
async def rag_direct(request: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Direct RAG query without agent orchestration.
    Useful for backwards compatibility or specific RAG-only queries.
    """
    deps = AgentDeps(**request.dict())
    
    # Get collection
    db = chroma_manager.get_or_create_collection(
        account_unique_id=deps.account_unique_id,
        embedding_function=embedding_manager
    )
    
    async def generate():
        
        async for chunk in search_db_advanced(
            manager=chroma_manager,
            db=db,
            query=request.query,
            relevance_score=deps.relevance_score,
            k_value=deps.k_value,
            sources_returned=deps.sources_returned,
            account_unique_id=deps.account_unique_id,
            visitor_email=deps.visitor_email,
            chat_history=deps.chat_history,
            prompt_text=deps.prompt_text,
            temperature=deps.temperature,
            scoreapp_report_text=deps.scoreapp_report_text,
            user_products_prompt=deps.user_products_prompt
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==============================================================================
# DEBUGGING ENDPOINT
# ==============================================================================

@router.post("/test-agent")
async def test_agent(request: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Test endpoint to verify agent is working without streaming complexity.
    """
    deps = AgentDeps(
        query=request.query,
        prompt=request.prompt,
        visitor_email=request.visitor_email,
        visitor_uuid=request.visitor_uuid,
        account_unique_id=request.account_unique_id,
        chat_history=request.chat_history,
        relevance_score=request.relevance_score,
        k_value=request.k_value,
        sources_returned=request.sources_returned,
        temperature=request.temperature,
        chat_session_id=request.chat_session_id,
        scoreapp_report_text=request.scoreapp_report_text,
        user_products_prompt=request.user_products_prompt
    )
    
    try:
        print("🧪 Testing agent...")
        result = await expert_agent.run(request.query, deps=deps)
        print(f"✅ Agent result: {result.data}")
        
        return {
            "success": True,
            "response": result.data,
            "all_messages": [
                {"role": msg.kind, "content": str(msg.content)} 
                for msg in result.all_messages()
            ]
        }
    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }