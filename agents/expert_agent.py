# agents/expert_agent.py
from pydantic_ai import Agent, RunContext
from schemas.agent_schemas import Query
import os


expert_agent = Agent(
    "openai:gpt-4o",
    deps_type=Query,
    system_prompt=(
        """You are an expert query co-ordinator, with access to multiple tools.
        Your job is to determine which tool to use based on the user's query.
        User's query is available in `deps.query`.
        You have access to the following tools:
        1. RAG Tool — for answering knowledge-based queries.
        2. Calendar Tool — for scheduling and managing appointments.
        Use the RAG Tool for information retrieval tasks.
        Use the Calendar Tool for any scheduling-related queries.
        Be sure to call the appropriate tool based on the user's request."""
    ),
)

@expert_agent.tool()
async def rag_tool(ctx: RunContext[str]) -> str:
    print("🧠 RAG Tool invoked with query:", ctx.deps)
    RAG_PROMPT = ctx.deps.prompt
    print("RAG Prompt: ", RAG_PROMPT)
    # Integrate with your RAG agent here
    return f"RAG Tool still under construction - not ready to use. {ctx.deps}"

@expert_agent.tool()
async def calendar_tool(ctx: RunContext[str]) -> str:
    print("📅 Calendar Tool invoked with query:", ctx.deps)
    # return f"Calendar Tool still under construction - not ready to use. {ctx.deps}"
    return {"available_slots": ["2025-10-13 10:00", "2025-10-15 14:00"]}