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

CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
ENVIRONMENT = os.environ.get('ENVIRONMENT')
CHROMA_SERVER_AUTHN_CREDENTIALS = os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS')
headers = {
    'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
    'Content-Type': 'application/json'
}


class QueryContext(BaseModel):
    """
    Context for managing a task list.
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

async def query_task_list_agent(query: str):
    """
    Query the task list agent with user-specific tasks.
    """
    context = QueryContext(query=query)
    
    result = await rephrase_user_query_agent.run(query, deps=context)
    
    print("Rephrased Query: ", result)
    
    return result.output
    
    