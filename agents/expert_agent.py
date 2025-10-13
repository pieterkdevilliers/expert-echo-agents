# agents/expert_agent.py
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
from schemas.agent_schemas import AgentDeps
import shared_utils.query_source_data as rag_agent
import os

# Import your existing components
from shared_utils.query_source_data import (
    ChromaDBManager,
    embedding_manager,
    CHAT_MODEL_NAME,
    ENVIRONMENT,
    CHROMA_ENDPOINT,
    CHROMA_SERVER_AUTHN_CREDENTIALS,
    headers
)


# Initialize ChromaDB Manager (singleton pattern)
chroma_manager = ChromaDBManager(
    environment=ENVIRONMENT,
    chroma_endpoint=CHROMA_ENDPOINT,
    headers=headers
)


# Create the expert agent
expert_agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDeps,
    system_prompt=(
        """You are an intelligent assistant with access to multiple tools:
        
        1. **RAG Tool** — Search and retrieve information from the knowledge base
        2. **Calendar Tool** — Schedule and manage calendar events
        
        Guidelines:
        - Use the RAG Tool for any informational queries, questions about products, services, or knowledge base content
        - Use the Calendar Tool for scheduling, availability checks, or event management
        - You can use multiple tools in sequence if needed
        - Always provide clear, helpful responses based on tool outputs
        """
    ),
)

@expert_agent.tool()
async def search_knowledge_base(ctx: RunContext[AgentDeps], search_query: str) -> str:
    """
    Wrap the existing streaming RAG logic as a tool.
    """
    print(f"🧠 RAG Tool invoked with query: {search_query}")
    
    deps = ctx.deps
    full_response = ""
    sources = []
    error_msg = None

    # Get the RAG DB for this account
    db = rag_agent.ChromaDBManager(
        environment=deps.env or "prod",
        chroma_endpoint=deps.chroma_endpoint,
        headers=deps.headers
    ).get_or_create_collection(
        account_unique_id=deps.account_unique_id,
        embedding_function=rag_agent.embedding_manager
    )
    
    try:
        async for chunk in rag_agent.search_db_advanced(
            manager=None,  # You can pass your manager if needed
            db=db,
            query=search_query,
            relevance_score=deps.relevance_score,
            k_value=deps.k_value,
            sources_returned=deps.sources_returned,
            account_unique_id=deps.account_unique_id,
            visitor_email=deps.visitor_email,
            chat_history=deps.chat_history,
            prompt_text=deps.prompt,
            temperature=deps.temperature,
            scoreapp_report_text=deps.scoreapp_report_text,
            user_products_prompt=deps.user_products_prompt
        ):
            if chunk["type"] == "chunk":
                full_response += chunk["content"]
            elif chunk["type"] == "sources":
                sources = chunk["content"]
            elif chunk["type"] == "error":
                error_msg = chunk["content"]
                break

        if error_msg:
            return f"Error in RAG tool: {error_msg}"
        
        # Optionally: store sources in ctx if agent wants to use them later
        ctx.metadata["sources"] = sources
        
        return full_response

    except Exception as e:
        print(f"❌ RAG Tool error: {e}")
        return f"Error in RAG tool: {str(e)}"


@expert_agent.tool()
async def calendar_search(ctx: RunContext[AgentDeps], date_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Check calendar availability and manage events.
    
    Args:
        date_range: Optional date range to check (e.g., "next week", "October 15-20")
    
    Returns:
        Dictionary containing available slots
    """
    print(f"📅 Calendar Tool invoked with date_range: {date_range}")
    
    # Placeholder for calendar integration
    return {
        "success": True,
        "available_slots": [
            "2025-10-13 10:00",
            "2025-10-15 14:00",
            "2025-10-16 09:00"
        ],
        "message": "Calendar integration ready for implementation"
    }