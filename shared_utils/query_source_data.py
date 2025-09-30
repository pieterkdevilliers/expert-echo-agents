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
print(f"Using OpenAI chat model: {CHAT_MODEL_NAME}")

CHROMA_PATH = "chroma"
ENVIRONMENT = os.environ.get('ENVIRONMENT')

# Chroma API endpoint and credentials
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
            print(f"Retrieved existing local collection: {collection_name}")
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Created new local collection: {collection_name}")
        print('Collection Returned: ', collection)
        return collection
    
    def _handle_remote_collection(self, account_unique_id: str):
        """Handle remote ChromaDB collection via Render-hosted API"""
        collection_name = f"collection-{account_unique_id}"
        print("collection_name: ", collection_name)

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
            print(f"Created remote collection: {collection_name} (id={collection_id})")
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
    
    def query_remote_collection(self, collection_dict, queries, n_results=7):
        """Query a remote collection"""
        collection_id = collection_dict["collection_id"]
        url = f"{collection_dict['endpoint']}/collections/{collection_id}/query"
        payload = {
            "queries": queries,
            "n_results": n_results
        }
        resp = requests.post(url, json=payload, headers=collection_dict["headers"])
        resp.raise_for_status()
        return resp.json()



###########################################
# Previously Search DB
###########################################


# Define response structure using Pydantic
class DetailedSearchResponse(BaseModel):
    query: str
    response_text: str
    sources: List[Optional[str]]
    confidence_score: Optional[float] = None
    context_used: bool = True

def search_db_advanced(
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
) -> Union[str, DetailedSearchResponse]:
    """
    Advanced search with structured response using Pydantic AI
    Handles local and remote Chroma collections.
    """
    print(f"Relevant score: {relevance_score}")
    print(f"k value: {k_value}")
    print(f"Type of db: {type(db)}")
    print(f"Temperature: {temperature}")

    # 1️⃣ Local environment
    if ENVIRONMENT == 'development' and isinstance(db, chromadb.Collection):
        results = db.query(
            query_texts=[query],
            n_results=k_value,
            include=["metadatas", "documents", "distances"]
        )
        if not results['documents'][0]:
            return f"Unable to find matching results for: {query}"

    # 2️⃣ Remote environment (db is a dict)
    elif isinstance(db, dict) and db.get("type") == "remote":
        try:
            # Use the helper to query remote collection
            results = manager.query_remote_collection(db, [query], n_results=k_value)
        except Exception as e:
            print(f"Error querying remote ChromaDB: {e}")
            return f"Database connection error: {str(e)}"

    else:
        return "Invalid database object provided."

    # 3️⃣ Extract documents and metadata
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return f"Unable to find matching results for: {query}"

    # 4️⃣ Build context and chat history
    context_text = "\n\n---\n\n".join(doc for doc in documents)

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{(msg.sender_type if hasattr(msg, 'sender_type') else msg['sender_type']).capitalize()}: "
            f"{(msg.message_text if hasattr(msg, 'message_text') else msg['message_text'])}"
            for msg in chat_history
        )

    # 5️⃣ Create AI agent
    model = OpenAIChatModel(CHAT_MODEL_NAME, temperature=temperature)
    agent = Agent(
        model=model,
        result_type=DetailedSearchResponse,
        system_prompt=f"""You are a helpful assistant that provides structured responses to user queries.
            Analyze the provided context and respond with:
            1. A helpful answer to the question  
            2. An assessment of how well the context matches the query
            3. Whether you used the provided context in your response

            Chat History:
            {history_text}

            Context from knowledge base:
            {context_text}

            ScoreApp Report:
            {scoreapp_report_text}

            User Products:
            {user_products_prompt}

            Additional Instructions:
            {prompt_text or ""}

            Question: {query}

            Please provide a helpful response based on the context provided above."""
    )

    # 6️⃣ Run agent and attach sources
    try:
        result = agent.run_sync(query)
        response = result.data
        response.sources = [meta.get("source", None) for meta in metadatas[:sources_returned]]
        response.query = query
        return response
    except Exception as e:
        print(f"Error generating structured response: {e}")
        return f"Error generating response: {str(e)}"