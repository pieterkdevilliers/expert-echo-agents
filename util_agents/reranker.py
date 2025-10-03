import os
import logfire
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
    logfire.instrument_pydantic_ai()
else:
    logfire.instrument_pydantic_ai()


class RerankingResult(BaseModel):
    """Structured output for document reranking"""
    ranked_indices: List[int] = Field(
        description="List of document indices ordered from most to least relevant (1-indexed)"
    )


class RerankingContext(BaseModel):
    """Context for the reranking agent"""
    query: str
    documents: List[str]
    metadatas: List[dict]
    top_n: int = 5


# Create the reranking agent with structured output
reranker_agent = Agent[RerankingResult](
    'openai:gpt-4o-mini',
    deps_type=RerankingContext,
    system_prompt="""You are an expert document relevance ranker.
    
        Your task is to analyze a user query and a list of candidate documents, 
        then rank the documents from most to least relevant to the query.

        Consider:
        - Semantic relevance to the query
        - Information completeness
        - Directness of the answer
        - Quality of the content

        You MUST return a structured response with the ranked_indices field containing 
        a list of integers representing document numbers in order of relevance.

        IMPORTANT: Return ONLY the structured data, no explanations or reasoning."""
        )


@reranker_agent.system_prompt
def add_documents_to_prompt(ctx: RunContext[RerankingContext]) -> str:
    """Dynamically add documents to the system prompt"""
    docs_str = "\n".join(
        f"{i+1}. {text[:500]}"  # truncate long docs
        for i, text in enumerate(ctx.deps.documents)
    )
    
    return f"""
Here are the candidate documents to rank:

{docs_str}

User Query: "{ctx.deps.query}"

Rank these documents from most to least relevant to the user's query.
"""


async def rerank_with_gpt(
    query: str, 
    documents: List[str], 
    metadatas: List[dict], 
    model: str = "gpt-4o-mini", 
    top_n: int = 5
) -> tuple[List[str], List[dict]]:
    """
    Use GPT via pydantic-ai agent to rerank candidate documents.
    
    Args:
        query: The user's search query
        documents: List of document texts to rerank
        metadatas: Corresponding metadata for each document
        model: OpenAI model to use (note: agent uses model from initialization)
        top_n: Number of top documents to return
        
    Returns:
        Tuple of (reranked_documents, reranked_metadatas)
    """
    print('******reranking started: ')
    print('******query: ', query)
    print(f'****** documents: {len(documents)} documents')
    print(f'******metadatas: {len(metadatas)} metadatas')
    
    if not documents:
        print("No documents to rerank")
        return [], []
    
    try:
        # Create context for the agent
        context = RerankingContext(
            query=query,
            documents=documents,
            metadatas=metadatas,
            top_n=top_n
        )
        
        # Run the agent with structured output
        result = await reranker_agent.run(
            "Rank all documents by relevance. Return only the ranked_indices list.",
            deps=context,
        )
        
        print('*********************RERANKING Result: ', result)
        
        # Extract ranked indices (convert from 1-indexed to 0-indexed)
        # ranked_ids = [idx - 1 for idx in result.data.ranked_indices if 1 <= idx <= len(documents)]
        # print('*********************RANKED IDS (0-indexed): ', ranked_ids)
        
        # Build reranked lists
        reranked_docs = []
        reranked_metas = []
        # for idx in ranked_ids[:top_n]:
        #     if 0 <= idx < len(documents):
        #         reranked_docs.append(documents[idx])
        #         reranked_metas.append(metadatas[idx])
        
        print('************************************RERANKED***************************************')
        print('Reranked docs count:', len(reranked_docs))
        print('Reranked metas count:', len(reranked_metas))
        
        return reranked_docs, reranked_metas
    
    except Exception as e:
        print(f"❌ Error in reranking: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Return original documents if reranking fails
        return documents[:top_n], metadatas[:top_n]