# agents/expert_agent.py
import os
from typing import Dict, Any
from pydantic_ai import Agent, RunContext
import shared_utils.agent_query_source_data as rag_agent
from schemas.agent_schemas import Query

# CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
ENVIRONMENT = os.environ.get('ENVIRONMENT')
# CHROMA_SERVER_AUTHN_CREDENTIALS = os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS')

# headers = {
#     'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
#     'Content-Type': 'application/json'
# }

embedding_manager = rag_agent.OpenAIEmbeddingManager()

expert_agent = Agent[Query, Dict[str, Any]](  # Specify deps and result types
    "openai:gpt-4o",
    system_prompt=(
        """You are a routing agent. Your ONLY job is to decide which single tool to call.
        
        Tools available:
        - rag_tool: For questions about information, products, services, pricing, features, or any knowledge
        - calendar_tool: ONLY for scheduling appointments or checking availability
        
        RULES:
        1. Call EXACTLY ONE tool (never both)
        2. Return ONLY the tool's output - do not add any text
        3. Do not explain your choice
        4. Do not answer the question yourself
        
        If the question is about information/knowledge → call rag_tool
        If the question is ONLY about scheduling → call calendar_tool"""
    ),
    retries=0,  # Prevent retry loops
)

@expert_agent.tool()
async def rag_tool(ctx: RunContext[Query]) -> Dict[str, Any]:
    """
    Initiates a RAG agent with custom prompt and executes the search.
    Returns a signal that includes a generator for streaming.
    """
    print("🧠 RAG Tool invoked with query:", ctx.deps.query)
    print(f"🧠 Using custom prompt for RAG agent")
    
    # This is the key: return a dict with metadata that tells the endpoint
    # to execute the RAG pipeline (can't return generator from tool)
    return {
        "tool": "rag",
        "action": "execute_rag_agent",
        "query": ctx.deps.query,
        "account_unique_id": ctx.deps.account_unique_id,
        "message": "RAG agent will be executed with custom prompt"
    }

@expert_agent.tool()
async def calendar_tool(ctx: RunContext[Query]) -> Dict[str, Any]:
    """
    Handles calendar and scheduling queries
    """
    print("📅 Calendar Tool invoked with query:", ctx.deps.query)
    # TODO: Implement actual calendar logic
    return {
        "tool": "calendar",
        "status": "success",
        "available_slots": ["2025-10-13 10:00", "2025-10-15 14:00"],
        "message": "Calendar integration coming soon"
    }