from fastapi import APIRouter, Depends, HTTPException
import core.auth as auth
from schemas.agent_schemas import Query, UserQuery
import util_agents.rag_query_agent as rag_agent
import util_agents.rephrase_user_query as rephrase_agent


router = APIRouter()


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
    
    result = await rephrase_agent.query_task_list_agent(query=query.query)

    return result

