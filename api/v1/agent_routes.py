from fastapi import APIRouter, Depends, HTTPException
import core.auth as auth
from schemas.agent_schemas import Query
import util_agents.rag_query_agent as rag_agent


router = APIRouter()


@router.post("/query")
async def rag_query(query: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Accepts a query payload, runs it through the AI agent, and returns the response.
    """
    print("Auth Status: ", authorized)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    response = await rag_agent.query_rag_query_agent(query=query)
    # result = await agent.run(query)
    return {"response": response}
