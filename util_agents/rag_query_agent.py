import logfire
from pydantic_ai import Agent
from schemas.agent_schemas import Query
from dotenv import load_dotenv

load_dotenv()

logfire.configure()  
logfire.instrument_pydantic_ai()


async def query_rag_query_agent(query: Query):
    # Build agent with session-specific instructions
    agent = Agent[str, str](
        "openai:gpt-4o",
        instructions=query.prompt,
    )

    # Flatten history into a string
    history_text = "\n".join(
        f"{msg['sender']}: {msg['message']}"
        for msg in query.chat_history
    )

    # Construct the full input
    full_input = f"""
    Chat history so far:
    {history_text}

    Now the user asks:
    {query.query}
    """

    # Run the agent
    result = await agent.run(full_input)

    return result