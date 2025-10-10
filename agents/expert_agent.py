# agents/expert_agent.py
from pydantic_ai import Agent, RunContext
from tools.rag_tool import RAGTool
# from tools.calendar_tool import CalendarTool  # future
import os


expert_agent = Agent(
    "openai:gpt-4o",
    deps_type=str,
    system_prompt=(
        """You are a helpful assistant with two tools:
        1. RAG Tool — for answering knowledge-based queries.
        2. Calendar Tool — for scheduling and managing events.
        Always use the RAG Tool for information retrieval tasks.
        Use the Calendar Tool for any scheduling-related queries.
        Be sure to call the appropriate tool based on the user's request."""
    ),
)

@expert_agent.tool()
async def rag_tool(ctx: RunContext[str]) -> str:
    print("🧠 RAG Tool invoked with query:", ctx.deps)
    return f"RAG Tool still under construction - not ready to use. {ctx.deps}"

@expert_agent.tool()
async def calendar_tool(ctx: RunContext[str]) -> str:
    print("📅 Calendar Tool invoked with query:", ctx.deps)
    # return f"Calendar Tool still under construction - not ready to use. {ctx.deps}"
    return {"available_slots": ["2025-10-13 10:00", "2025-10-15 14:00"]}