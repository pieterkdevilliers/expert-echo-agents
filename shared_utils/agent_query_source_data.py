import os
import requests
from typing import List, Dict, Union
# import chromadb
from util_agents import rephrase_user_query as rephrase_agent
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL_NAME = os.environ.get('OPENAI_CHAT_MODEL')
# CHROMA_PATH = "chroma"
ENVIRONMENT = os.environ.get('ENVIRONMENT')
# CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
# CHROMA_SERVER_AUTHN_CREDENTIALS = os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS')

# headers = {
#     'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
#     'Content-Type': 'application/json'
# }

# Initialize OpenAI client directly
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Direct embedding function (simplified approach)
def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings from OpenAI directly"""
    if not isinstance(texts, list):
        texts = [texts]
    
    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )
    return [embedding.embedding for embedding in response.data]


class OpenAIEmbeddingManager:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._dimension = None

    # ✅ Required interface for batch embeddings
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        Calls this for documents or queries.
        """
        # Ensure all inputs are strings
        input = [str(i) for i in input]

        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        embeddings = [e.embedding for e in response.data]

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    # ✅ Helper for multiple documents
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.__call__(texts)

    # ✅ Helper for single query, compatible signature
    def embed_query(self, input: str, **kwargs) -> List[List[float]]:
        """
        Embed a single query in batch format (expects list of lists)
        """
        return self.__call__([str(input)])  # returns List[List[float]]

    # ✅ Retrieve embedding dimension
    def get_dimension(self) -> int:
        if self._dimension is None:
            dummy = self.__call__(["dummy"])
            self._dimension = len(dummy[0])
        return self._dimension

    # ✅ Name used to avoid embedding function conflicts
    def name(self) -> str:
        return f"openai-{self.model}"


embedding_manager = OpenAIEmbeddingManager()



########################################
# Previously prepare_db
########################################

# class ChromaDBManager:
#     def __init__(self, environment: str, chroma_endpoint: str = None, headers: dict = None):
#         self.environment = environment
#         self.chroma_endpoint = chroma_endpoint
#         self.headers = headers or {}

#     def get_or_create_collection(self, account_unique_id: str, embedding_function=None):
#         """Get or create a collection based on environment"""
#         if self.environment == "development":
#             return self._handle_local_collection(account_unique_id, embedding_function)
#         else:
#             return self._handle_remote_collection(account_unique_id)
    
#     def _handle_local_collection(self, account_unique_id: str, embedding_function):
#         """Handle local ChromaDB collection"""
#         chroma_path = f"./chroma/{account_unique_id}"
#         os.makedirs(chroma_path, exist_ok=True)
        
#         client = chromadb.PersistentClient(path=chroma_path)
#         collection_name = f"collection-{account_unique_id}"
        
#         try:
#             collection = client.get_collection(
#                 name=collection_name,
#                 embedding_function=embedding_function
#             )
#         except Exception:
#             collection = client.create_collection(
#                 name=collection_name,
#                 embedding_function=embedding_function
#             )
#         return collection
    
#     def _handle_remote_collection(self, account_unique_id: str):
#         """Handle remote ChromaDB collection via Render-hosted API"""
#         collection_name = f"collection-{account_unique_id}"

#         # Step 1: Fetch all collections from remote server
#         collections_url = f"{self.chroma_endpoint}/collections"
#         resp = requests.get(collections_url, headers=self.headers)
#         if resp.status_code != 200:
#             raise RuntimeError(f"Error fetching collections: {resp.text}")
#         collections = resp.json()

#         # Step 2: Find the collection ID by name
#         collection_id = None
#         for col in collections:
#             if col["name"] == collection_name:
#                 collection_id = col["id"]
#                 break

#         if not collection_id:
#             # Optionally, create the collection if it doesn't exist
#             create_url = f"{self.chroma_endpoint}/collections"
#             payload = {"name": collection_name}
#             resp = requests.post(create_url, json=payload, headers=self.headers)
#             resp.raise_for_status()
#             collection_id = resp.json()["id"]

#         else:
#             print(f"Found remote collection: {collection_name} (id={collection_id})")

#         # Step 3: Return a dict representing the "collection"
#         return {
#             "type": "remote",
#             "collection_name": collection_name,
#             "collection_id": collection_id,
#             "endpoint": self.chroma_endpoint,
#             "headers": self.headers,
#             "exists": True,
#         }
    
#     def query_remote_collection(self, collection_dict, queries: List[str], n_results=7, include=None):
#         """Query a remote collection using embeddings"""
#         collection_id = collection_dict["collection_id"]
#         url = f"{collection_dict['endpoint']}/collections/{collection_id}/query"
        
#         # Embed the queries
#         embeddings = embedding_manager.embed_query(queries[0]) if len(queries) == 1 else embedding_manager.embed_documents(queries)
        
#         payload = {
#             "query_embeddings": embeddings,
#             "n_results": n_results,
#             "include": include or ["documents", "metadatas", "distances"],
#             "where": {},  # optional filters
#             "where_document": {}  # optional filters
#         }
        
#         resp = requests.post(url, json=payload, headers=collection_dict["headers"])
#         resp.raise_for_status()
#         print('chroma query response: ', resp.json())
#         return resp.json()



###########################################
# Previously Search DB
###########################################
# Modified search_db_advanced to support streaming
async def search_db_advanced(
    manager,
    db: Dict, 
    query: str, 
    relevance_score: float, 
    k_value: int, 
    sources_returned: int, 
    account_unique_id: str, 
    visitor_email: str, 
    chat_history=None, 
    prompt_text=None, 
    temperature=0.2,
    scoreapp_report_text={},
    user_products_prompt=""
):
    """
    Streaming version - yields chunks as they come from AI
    Returns only the top N most relevant sources based on distance scores
    """
    print('query received in search action: ', query)
    print('scoreapp text received in search action: ', scoreapp_report_text)
    print('user_products received in search action: ', user_products_prompt)
    
    # 1️⃣ Validate DB handle (from Pinecone manager)
    if not isinstance(db, dict) or db.get("type") != "remote" or "index" not in db:
        yield {
            "type": "error",
            "content": "Invalid database object provided."
        }
        return

    namespace = db["namespace"]
    index = db["index"]

    # 2️⃣ Embed the query (same as before)
    query_emb = embedding_manager.embed_query(query)[0]  # [0] since batch of 1 → list[float]

    # 3️⃣ Query Pinecone
    try:
        results = index.query(
            vector=query_emb,
            top_k=k_value,
            include_metadata=True,
            include_values=False,  # Save bandwidth – don't need vectors back
            namespace=namespace
        )
    except Exception as e:
        yield {
            "type": "error",
            "content": f"Database query error: {str(e)}"
        }
        return


    # 4️⃣ Extract matches (Pinecone format)
    matches = results.get('matches', [])
    if not matches:
        yield {"type": "error", "content": f"Unable to find matching results for: {query}"}
        return

    # Extract documents, metadatas, distances
    # Assume you stored "text" in metadata during ingestion
    documents = [match['metadata'].get('text', '') for match in matches]
    metadatas = [match['metadata'] for match in matches]
    scores = [match['score'] for match in matches]  # Similarity (higher better)
    distances = [1 - score for score in scores]    # Convert to "distance" (lower better)

    # Optional: Filter by relevance_score (e.g., if converted distance > threshold, skip)
    # Example: filtered = [(doc, meta, dist) for doc, meta, dist in zip(documents, metadatas, distances) if dist <= relevance_score]
    # But original doesn't use it – add if needed

    # 5️⃣ Build context from docs (same)
    context_text = "\n\n---\n\n".join(doc for doc in documents)

    # 6️⃣ Create sorted list for source ranking (same, using converted distances)
    source_ranking = []
    for i, (dist, meta) in enumerate(zip(distances, metadatas)):
        source = meta.get("source", None)
        if source:  # Only include if source exists
            source_ranking.append({
                "distance": dist,
                "source": source,
                "metadata": meta,
                "index": i
            })
    source_ranking.sort(key=lambda x: x["distance"])  # Ascending distance (best first)
    
    # Get top N unique sources (same)
    best_sources = list(set([item["source"] for item in source_ranking[:sources_returned]]))
    print('best_sources: ', best_sources)
    
    print(f"Using {k_value} docs for context, returning top {sources_returned} sources")
    print(f"Distance scores: {[f'{item['distance']:.4f}' for item in source_ranking[:sources_returned]]}")
    print(f"Distance scores - Best Sources: {[f'{item['distance']:.4f}' for item in source_ranking]}")

    # 7️⃣ Build chat history (same)
    history_text = ""
    if chat_history:
        formatted_messages = []
        for msg in chat_history:
            sender = msg.get('sender', 'unknown') if isinstance(msg, dict) else getattr(msg, 'sender', 'unknown')
            message = msg.get('message', '') if isinstance(msg, dict) else getattr(msg, 'message', '')
            role = "User" if sender == "user" else "Assistant"
            formatted_messages.append(f"{role}: {message}")
        history_text = "\n".join(formatted_messages)

    # 8️⃣ Create AI agent
    model = OpenAIChatModel(CHAT_MODEL_NAME)
    
    system_prompt_parts = []
    
    system_prompt_parts.append(prompt_text or """You are an expert analyst for a business, tasked with providing clear, comprehensive, and well-structured answers. Your tone should aim to match the tone of the source material, remaining conversational.

Your primary goal is to synthesize a complete answer from ALL relevant information found in the provided context, including the Chat History. Do not just use the first piece of information you find. If multiple parts of the context are relevant, combine them into a single, coherent response.

Follow these strict formatting rules:
1. Structure your answer in clear, well-written paragraphs. Do not return a single block of text.
2. Ensure the response is easy to read and logically organized.

Critically, you must adhere to these constraints:
- Base your answer ONLY on the information provided below.
- Do not mention the words "context", "information provided", or "source documents".
- If the information is not in the context to answer the question, you must respond with: 
  "I don't have an answer for that right now. Please use the button below to send us an email, and we will get you the information you need."
- Do not make up an answer.
- Keep reference to the chat history, in order to keep the conversation realistic.""")

    if history_text:
        system_prompt_parts.append(f"""
PREVIOUS CONVERSATION:
{history_text}

IMPORTANT: Use the conversation history above to maintain context. If the user asks about something mentioned earlier in the conversation (like their name, previous questions, etc.), refer to the history above.""")

    system_prompt_parts.append(f"""
KNOWLEDGE BASE CONTEXT:
{context_text}""")

    if scoreapp_report_text and scoreapp_report_text.get('scoreapp_report_text'):
        system_prompt_parts.append(f"""
SCOREAPP REPORT:
{scoreapp_report_text.get('scoreapp_report_text')}""")

    if user_products_prompt:
        system_prompt_parts.append(f"""
AVAILABLE PRODUCTS AND SERVICES:
{user_products_prompt}""")

    full_system_prompt = "\n".join(system_prompt_parts)
    print('************************full prompt start: ', full_system_prompt, 'full prompt end*********************')
    
    agent = Agent(
        model=model,
        system_prompt=full_system_prompt
    )

    # 9️⃣ Stream agent response (same)
    try:
        previous_text = ""

        # Start the stream (enter context)
        stream_response = await agent.run_stream(
            query,
            model_settings={"temperature": temperature}
        )

        async for chunk in stream_response.stream_text():
            new_text = chunk[len(previous_text):]
            previous_text = chunk
            if new_text:
                yield {
                    "type": "chunk",
                    "content": new_text
                }

        # Explicitly close / cleanup if needed
        # (most streaming clients auto-close on exhaustion, but explicit is safer)
        await stream_response.aclose()  # or stream_response.close() if sync

        yield {"type": "sources", "content": best_sources}
        yield {"type": "done", "content": None}

    except Exception as e:
        yield {
            "type": "error",
            "content": f"Error generating response: {str(e)}"
        }