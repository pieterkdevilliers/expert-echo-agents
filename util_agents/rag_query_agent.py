import logfire
from pydantic_ai import Agent
from schemas.agent_schemas import Query
from dotenv import load_dotenv

load_dotenv()

logfire.configure()  
logfire.instrument_pydantic_ai()


def build_rag_query_agent(prompt_text: str) -> Agent[Query, str]:
    return Agent[Query, str](
        "openai:gpt-4o",
        instructions=prompt_text,
    )

async def query_rag_query_agent(query: Query):
    agent = build_rag_query_agent(query.instructions)
    result = await agent.run(query.query)
    return result
