import argparse
import os
import requests
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from sqlmodel import select, Session
import chromadb
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL_NAME = os.environ.get('OPENAI_CHAT_MODEL')
CHROMA_PATH = "chroma"
ENVIRONMENT = os.environ.get('ENVIRONMENT')
CHROMA_ENDPOINT = os.environ.get('CHROMA_ENDPOINT')
CHROMA_SERVER_AUTHN_CREDENTIALS = os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS')

headers = {
    'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
    'Content-Type': 'application/json'
}

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

    # ✅ Required Chroma interface for batch embeddings
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        Chroma calls this for documents or queries.
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

    # ✅ Helper for single query, Chroma-compatible signature
    def embed_query(self, input: str, **kwargs) -> List[List[float]]:
        """
        Embed a single query in batch format (Chroma expects list of lists)
        """
        return self.__call__([str(input)])  # returns List[List[float]]

    # ✅ Retrieve embedding dimension
    def get_dimension(self) -> int:
        if self._dimension is None:
            dummy = self.__call__(["dummy"])
            self._dimension = len(dummy[0])
        return self._dimension

    # ✅ Name used by Chroma to avoid embedding function conflicts
    def name(self) -> str:
        return f"openai-{self.model}"


embedding_manager = OpenAIEmbeddingManager()



########################################
# Previously prepare_db
########################################

class ChromaDBManager:
    def __init__(self, environment: str, chroma_endpoint: str = None, headers: dict = None):
        self.environment = environment
        self.chroma_endpoint = chroma_endpoint
        self.headers = headers or {}

    def get_or_create_collection(self, account_unique_id: str, embedding_function=None):
        """Get or create a collection based on environment"""
        if self.environment == "development":
            return self._handle_local_collection(account_unique_id, embedding_function)
        else:
            return self._handle_remote_collection(account_unique_id)
    
    def _handle_local_collection(self, account_unique_id: str, embedding_function):
        """Handle local ChromaDB collection"""
        chroma_path = f"./chroma/{account_unique_id}"
        os.makedirs(chroma_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=chroma_path)
        collection_name = f"collection-{account_unique_id}"
        
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        return collection
    
    def _handle_remote_collection(self, account_unique_id: str):
        """Handle remote ChromaDB collection via Render-hosted API"""
        collection_name = f"collection-{account_unique_id}"

        # Step 1: Fetch all collections from remote server
        collections_url = f"{self.chroma_endpoint}/collections"
        resp = requests.get(collections_url, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Error fetching collections: {resp.text}")
        collections = resp.json()

        # Step 2: Find the collection ID by name
        collection_id = None
        for col in collections:
            if col["name"] == collection_name:
                collection_id = col["id"]
                break

        if not collection_id:
            # Optionally, create the collection if it doesn't exist
            create_url = f"{self.chroma_endpoint}/collections"
            payload = {"name": collection_name}
            resp = requests.post(create_url, json=payload, headers=self.headers)
            resp.raise_for_status()
            collection_id = resp.json()["id"]

        else:
            print(f"Found remote collection: {collection_name} (id={collection_id})")

        # Step 3: Return a dict representing the "collection"
        return {
            "type": "remote",
            "collection_name": collection_name,
            "collection_id": collection_id,
            "endpoint": self.chroma_endpoint,
            "headers": self.headers,
            "exists": True,
        }
    
    def query_remote_collection(self, collection_dict, queries: List[str], n_results=7, include=None):
        """Query a remote collection using embeddings"""
        collection_id = collection_dict["collection_id"]
        url = f"{collection_dict['endpoint']}/collections/{collection_id}/query"
        
        # Embed the queries
        embeddings = embedding_manager.embed_query(queries[0]) if len(queries) == 1 else embedding_manager.embed_documents(queries)
        
        payload = {
            "query_embeddings": embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
            "where": {},  # optional filters
            "where_document": {}  # optional filters
        }
        
        resp = requests.post(url, json=payload, headers=collection_dict["headers"])
        resp.raise_for_status()
        print('chroma query response: ', resp.json())
        return resp.json()



###########################################
# Previously Search DB
###########################################
# Modified search_db_advanced to support streaming
async def search_db_advanced(
    manager,
    db: Union[chromadb.Collection, Dict], 
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
    print('scoreapp text received in search action: ', scoreapp_report_text)
    print('user_products received in search action: ', user_products_prompt)
    
    # 1️⃣ Local environment
    if ENVIRONMENT == 'development' and isinstance(db, chromadb.Collection):
        results = db.query(
            query_texts=[query],
            n_results=k_value,
            include=["metadatas", "documents", "distances"]
        )
        if not results['documents'][0]:
            yield {
                "type": "error",
                "content": f"Unable to find matching results for: {query}"
            }
            return

    # 2️⃣ Remote environment
    elif isinstance(db, dict) and db.get("type") == "remote":
        try:
            results = manager.query_remote_collection(
                db,
                [query],
                n_results=k_value
            )
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Database connection error: {str(e)}"
            }
            return
    else:
        yield {
            "type": "error",
            "content": "Invalid database object provided."
        }
        return

    # 3️⃣ Extract documents, metadata, and distances
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        yield {"type": "error", "content": f"Unable to find matching results for: {query}"}
        return
    if not documents:
        yield {"type": "error", "content": "No documents to rerank."}
        return
    if not query.strip():
        yield {"type": "error", "content": "Query is empty."}
        return


    # 4️⃣ Build context from docs
    context_text = "\n\n---\n\n".join(doc for doc in documents)

    # 5️⃣ Use reranked order as relevance
    best_sources = []
    source_ranking = []

    for rank, meta in enumerate(metadatas[:sources_returned]):
        source = meta.get("source", None)
        
        if source:
            print('source: ', source)
            best_sources.append(source)
            print('best_sources: ', best_sources)
            source_ranking.append({
                "rank": rank + 1,  # 1-indexed rank
                "source": source,
                "metadata": meta
            })
            print('source_ranking: ', source_ranking)

    print('best_sources: ', best_sources)
    print(f"Using {len(documents)} reranked docs for context, returning top {sources_returned} sources")
    print('source_ranking: ', source_ranking)
    print(f"Reranked order: {[item['rank'] for item in source_ranking]}")

    # 6️⃣ Build chat history
    history_text = ""
    if chat_history:
        formatted_messages = []
        for msg in chat_history:
            sender = msg.get('sender', 'unknown') if isinstance(msg, dict) else getattr(msg, 'sender', 'unknown')
            message = msg.get('message', '') if isinstance(msg, dict) else getattr(msg, 'message', '')
            role = "User" if sender == "user" else "Assistant"
            formatted_messages.append(f"{role}: {message}")
        history_text = "\n".join(formatted_messages)

    # 7️⃣ Create AI agent
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

    # 8️⃣ Stream agent response
    try:
        # Track what we've already sent
        previous_text = ""

        async with agent.run_stream(
            query,
            model_settings={"temperature": temperature}
        ) as response:
            async for chunk in response.stream_text():
                # chunk is cumulative text → find only new part
                new_text = chunk[len(previous_text):]
                previous_text = chunk

                if new_text:  # only send if something new appeared
                    yield {
                        "type": "chunk",
                        "content": new_text
                    }
        
        # After streaming completes, send the BEST sources (not just first N)
        yield {
            "type": "sources",
            "content": reranked_metas
        }
        
        # Then signal completion
        yield {
            "type": "done",
            "content": None
        }

    except Exception as e:
        yield {
            "type": "error",
            "content": f"Error generating response: {str(e)}"
        }

# # Define response structure using Pydantic
# class DetailedSearchResponse(BaseModel):
#     query: str
#     response_text: str
#     sources: List[Optional[str]]
#     confidence_score: Optional[float] = None
#     context_used: bool = True

# async def search_db_advanced(
#     manager,
#     db: Union[chromadb.Collection, Dict], 
#     query: str, 
#     relevance_score: float, 
#     k_value: int, 
#     sources_returned: int, 
#     account_unique_id: str, 
#     visitor_email: str, 
#     chat_history=None, 
#     prompt_text=None, 
#     temperature=0.2,
#     scoreapp_report_text={},
#     user_products_prompt=""
# ) -> Union[str, DetailedSearchResponse]:
#     """
#     Advanced search with structured response using Pydantic AI
#     Handles local and remote Chroma collections.
#     """
#     print('scoreapp text received in search action: ', scoreapp_report_text)
#     print('user_products received in search action: ', user_products_prompt)
#     # 1️⃣ Local environment
#     if ENVIRONMENT == 'development' and isinstance(db, chromadb.Collection):
#         results = db.query(
#             query_texts=[query],
#             n_results=k_value,
#             include=["metadatas", "documents", "distances"]
#         )
#         if not results['documents'][0]:
#             return f"Unable to find matching results for: {query}"

#     # 2️⃣ Remote environment (db is a dict)
#     elif isinstance(db, dict) and db.get("type") == "remote":
#         try:
#             results = manager.query_remote_collection(
#                 db,
#                 [query],
#                 n_results=k_value
#             )
#         except Exception as e:
#             return f"Database connection error: {str(e)}"
#     else:
#         return "Invalid database object provided."

#     # 3️⃣ Extract documents and metadata
#     documents = results.get("documents", [[]])[0]
#     metadatas = results.get("metadatas", [[]])[0]

#     if not documents:
#         return f"Unable to find matching results for: {query}"

#     # 4️⃣ Build context and chat history - FIXED HERE
#     context_text = "\n\n---\n\n".join(doc for doc in documents)

#     # Fix: Access the correct keys from your chat history dict
#     history_text = ""
#     if chat_history:
#         formatted_messages = []
#         for msg in chat_history:
#             # Handle both dict and object formats
#             sender = msg.get('sender', 'unknown') if isinstance(msg, dict) else getattr(msg, 'sender', 'unknown')
#             message = msg.get('message', '') if isinstance(msg, dict) else getattr(msg, 'message', '')
            
#             # Format as conversation
#             role = "User" if sender == "user" else "Assistant"
#             formatted_messages.append(f"{role}: {message}")
        
#         history_text = "\n".join(formatted_messages)

#     # 5️⃣ Create AI agent with IMPROVED system prompt
#     model = OpenAIChatModel(CHAT_MODEL_NAME)
    
#     # Build the system prompt with clear sections
#     system_prompt_parts = []
#     print('SCORE_APP_RESULTS: ', scoreapp_report_text)
#     print('PRODUCTS: ', user_products_prompt)
#     # Base instructions
#     system_prompt_parts.append(prompt_text or """You are an expert analyst for a business, tasked with providing clear, comprehensive, and well-structured answers. Your tone should aim to match the tone of the source material, remaining conversational.

# Your primary goal is to synthesize a complete answer from ALL relevant information found in the provided context, including the Chat History. Do not just use the first piece of information you find. If multiple parts of the context are relevant, combine them into a single, coherent response.

# Follow these strict formatting rules:
# 1. Structure your answer in clear, well-written paragraphs. Do not return a single block of text.
# 2. Ensure the response is easy to read and logically organized.

# Critically, you must adhere to these constraints:
# - Base your answer ONLY on the information provided below.
# - Do not mention the words "context", "information provided", or "source documents".
# - If the information is not in the context to answer the question, you must respond with: 
#   "I don't have an answer for that right now. Please use the button below to send us an email, and we will get you the information you need."
# - Do not make up an answer.
# - Keep reference to the chat history, in order to keep the conversation realistic.""")

#     # Add chat history if available - CRITICAL: This comes FIRST
#     if history_text:
#         system_prompt_parts.append(f"""
# PREVIOUS CONVERSATION:
# {history_text}

# IMPORTANT: Use the conversation history above to maintain context. If the user asks about something mentioned earlier in the conversation (like their name, previous questions, etc.), refer to the history above.""")

#     # Add context from knowledge base
#     system_prompt_parts.append(f"""
# KNOWLEDGE BASE CONTEXT:
# {context_text}""")

#     # Add ScoreApp report if available
#     if scoreapp_report_text and scoreapp_report_text.get('scoreapp_report_text'):
#         system_prompt_parts.append(f"""
# SCOREAPP REPORT:
# {scoreapp_report_text.get('scoreapp_report_text')}""")

#     # Add user products if available
#     if user_products_prompt:
#         system_prompt_parts.append(f"""
# AVAILABLE PRODUCTS AND SERVICES:
# {user_products_prompt}""")

#     # Combine all parts
#     full_system_prompt = "\n".join(system_prompt_parts)
#     print('************************full prompt start: ', full_system_prompt, 'full prompt end*********************')
#     agent = Agent(
#         model=model,
#         system_prompt=full_system_prompt
#     )

#     # 6️⃣ Run agent asynchronously
#     try:
#         result = await agent.run(
#             query,
#             model_settings={"temperature": temperature})

#         # Build structured response
#         response = DetailedSearchResponse(
#             query=query,
#             response_text=result.output,
#             sources=[meta.get("source", None) for meta in metadatas[:sources_returned]],
#             confidence_score=None,
#             context_used=True
#         )
#         print('Reponse: ', response)
#         return response
#     except Exception as e:
#         return f"Error generating response: {str(e)}"