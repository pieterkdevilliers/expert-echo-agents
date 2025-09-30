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
        self.headers = headers
        
    def get_or_create_collection(self, account_unique_id: str, embedding_function=None):
        """Get or create a collection based on environment"""
        
        if ENVIRONMENT == 'development':
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
        """Handle remote ChromaDB collection"""
        collection_name = f"collection-{account_unique_id}"
        print('collection_name: ', collection_name)
        print('headers: ', self.headers)
        
        # Check if collection exists
        try:
            response = requests.get(
                f'{self.chroma_endpoint}/collections/{collection_name}',
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'type': 'remote',
                    'collection_name': collection_name,
                    'endpoint': self.chroma_endpoint,
                    'headers': self.headers,
                    'exists': True
                }
            else:
                return {
                    'type': 'remote',
                    'collection_name': collection_name,
                    'endpoint': self.chroma_endpoint,
                    'headers': self.headers,
                    'exists': False
                }
                
        except requests.RequestException as e:
            print(f"Error accessing remote collection: {e}")
            raise



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
    db: Union[chromadb.Collection, Dict], 
    query: str, 
    relevance_score: float, 
    k_value: int, 
    sources_returned: int, 
    account_unique_id: str, 
    visitor_email: str, 
    chat_history=None, 
    prompt_text=None, 
    temperature=0.2
) -> Union[str, DetailedSearchResponse]:
    """
    Advanced search with structured response using Pydantic AI
    """
    print(f"Relevant score: {relevance_score}")
    print(f"k value: {k_value}")
    print(f"Type of db: {type(db)}")
    print(f"Temperature: {temperature}")
    
    # Handle different environments
    if ENVIRONMENT == 'development':
        # db is a ChromaDB Collection object
        results = db.query(
            query_texts=[query],
            n_results=k_value,
            include=["metadatas", "documents", "distances"]
        )
        
        # Check relevance scores (distances in ChromaDB are inverse of similarity)
        if not results['documents'][0] or (results['distances'][0] and results['distances'][0][0] > (1 - relevance_score)):
            return f"Unable to find matching results for: {query}"
            
    else:
        # Remote ChromaDB handling
        try:
            client = chromadb.HttpClient(
                host='https://fastapi-rag-chroma.onrender.com', 
                port=8000, 
                headers=headers
            )
            collection_name = f'collection-{account_unique_id}'
            print(f"Collection name: {collection_name}")
            
            collection = client.get_collection(
                name=collection_name, 
                embedding_function=embedding_manager
            )

            results = collection.query(
                query_texts=[query],
                n_results=k_value,
                include=["metadatas", "documents", "distances"]
            )
            
        except Exception as e:
            print(f"Error querying remote ChromaDB: {e}")
            return f"Database connection error: {str(e)}"

    # Log the results to inspect the structure
    print(f"Query results: {results}")

    # Extract documents and metadata
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not documents:
        return f"Unable to find matching results for: {query}"

    # Create context text from documents
    context_text = "\n\n---\n\n".join(doc for doc in documents)

    # Build chat history text
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{(msg.sender_type if hasattr(msg, 'sender_type') else msg['sender_type']).capitalize()}: "
            f"{(msg.message_text if hasattr(msg, 'message_text') else msg['message_text'])}"
            for msg in chat_history
        )

    # Create Pydantic AI agent
    model = OpenAIChatModel(CHAT_MODEL_NAME, temperature=temperature)
    agent = Agent(
        model=model,
        result_type=DetailedSearchResponse,
        system_prompt=f"""You are a helpful assistant that provides structured responses to user queries.
        Analyze the provided context and respond with:
        1. A helpful answer to the question  
        2. An assessment of how well the context matches the query
        3. Whether you used the provided context in your response
        
        Be honest about the limitations of your knowledge based on the provided context.
        
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

    try:
        result = agent.run_sync(query)
        response = result.data
        
        # Add the sources from our search
        response.sources = [meta.get("source", None) for meta in metadatas[:sources_returned]]
        response.query = query
        
        return response
        
    except Exception as e:
        print(f"Error generating structured response: {e}")
        return f"Error generating response: {str(e)}"