import os
import logfire
from typing import List
from openai import OpenAI

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
    logfire.instrument_pydantic_ai()
else:
    logfire.instrument_pydantic_ai()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def rerank_with_gpt(query: str, documents: List[str], metadatas: List[dict], model="gpt-4o-mini", top_n=5):
    """
    Use GPT to rerank candidate documents and return top_n docs in best order.
    """
    print('******reranking started: ')
    print('******query: ', query)
    print('****** documents: ', documents)
    print('******metadatas: ', metadatas)
    
    if not documents:
        print("No documents to rerank")
        return [], []
    
    docs_str = "\n".join(
        f"{i+1}. (id={i}) {text[:500]}"  # truncate long docs to keep token count safe
        for i, text in enumerate(documents)
    )

    prompt = f"""
        The user asked: "{query}"

        Here are candidate documents:
        {docs_str}

        Rank the documents from most to least relevant to the query.
        Return only the document numbers in order, comma-separated (e.g., 3,1,2,...).
        """
    print("*********PROMPT FOR RERANKER: ", prompt)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        print('*********************RERANKING Response: ', response)
        
        ranking = response.choices[0].message.content.strip()
        print('*********************RANKING STRING: ', ranking)
        
        ranked_ids = [int(x.strip())-1 for x in ranking.split(",") if x.strip().isdigit()]
        print('*********************RANKED IDS: ', ranked_ids)

        reranked_docs = []
        reranked_metas = []
        for idx in ranked_ids[:top_n]:
            if 0 <= idx < len(documents):
                reranked_docs.append(documents[idx])
                reranked_metas.append(metadatas[idx])
        
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
