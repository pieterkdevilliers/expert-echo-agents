import logfire
from pydantic_ai import Agent
from schemas.agent_schemas import Query
from dotenv import load_dotenv

load_dotenv()

logfire.configure()  
logfire.instrument_pydantic_ai()


rag_query_agent = Agent[Query, str](
    'openai:gpt-4o',
    instructions='''
    You just answer questions
    '''
)


async def query_rag_query_agent(query: Query):
    """
    Query the article writer agent with a subject or title.
    """
    result = await rag_query_agent.run(query.query)
    return result
