import logfire
import os
import json
from pydantic_ai import Agent
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from schemas.agent_schemas import UserQuery
from core.config import settings
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
    logfire.instrument_pydantic_ai()
else:
    logfire.instrument_pydantic_ai()

ENVIRONMENT = os.environ.get('ENVIRONMENT')


class QueryContext(BaseModel):
    """
    Context for user queries.
    """
    query: str


rephrase_user_query_agent = Agent[QueryContext, str](
    'openai:gpt-4o',
    instructions='''
    You are an expert at rephrasing user queries to be more specific and detailed.
    Given a user query, rephrase it to enhance clarity and detail, making it more suitable
    for information retrieval tasks.
    Ensure the rephrased query retains the original intent while improving specificity.'''
)

async def rephrase_user_query(query: str):
    """
    Rephrase the user query for better clarity and detail.
    """
    context = QueryContext(query=query)
    
    result = await rephrase_user_query_agent.run(query, deps=context)
    
    print("Rephrased Query: ", result)
    
    return result.output
    
    