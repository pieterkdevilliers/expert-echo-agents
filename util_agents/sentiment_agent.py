import logfire
import os
from typing import List, Dict, Any
from pydantic_ai import Agent
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    logfire.configure()
    logfire.instrument_pydantic_ai()
else:
    logfire.instrument_pydantic_ai()

ENVIRONMENT = os.environ.get('ENVIRONMENT')


class QueryContext(BaseModel):
    """
    Context for user queries.
    """
    query: str


class Conversation(BaseModel):
    """
    Conversation context.
    """
    history: List[Dict[str, Any]] = []


class QuerySentiment(BaseModel):
    """
    Sentiment analysis result.
    """
    sentiment: str
    explanation: str

class ConversationAnalysisInput(BaseModel):
    """
    Main input for conversation sentiment analysis
    """
    conversation_text: str

initial_user_query_sentiment_agent = Agent[QueryContext, str](
    'openai:gpt-4o',
    output_type=QuerySentiment,
    instructions='''
    You are an expert at analyzing the sentiment of user queries.
    Return a structured sentiment classification with a concise explanation.
    
    - negative    = predominantly negative (frustration, anger, unresolved complaints)
    - neutral  = neutral / mixed / ongoing inquiry without strong emotion
    - positive  = positive overall (satisfaction, resolution, appreciation)
    
    Ensure your analysis is concise and relevant to the query's context.'''
)

async def analyze_initial_user_query_sentiment(query: str):
    """
    Analyze the sentiment of the user query.
    """
    context = QueryContext(query=query)

    result = await initial_user_query_sentiment_agent.run(query, deps=context)

    print("Sentiment Analysis Result: ", result)

    return result.output


def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """
    Format the conversation history into a single string.
    Each message is prefixed by the sender's role (USER or AGENT).
    """
    lines = []
    for msg in history:
        sender = msg.get('sender', 'unknown').upper()
        content = msg.get('message', '').strip()
        lines.append(f"{sender}: {content}")
    print("Formatted Conversation History: ", "\n".join(lines))
    return "\n\n".join(lines)


conversation_sentiment_agent = Agent[ConversationAnalysisInput, QuerySentiment](
    'openai:gpt-4o',
    output_type=QuerySentiment,
    instructions='''
    You are an expert at analyzing the overall sentiment of a customer support conversation.

    - negative    = predominantly negative (frustration, anger, unresolved complaints)
    - neutral  = neutral / mixed / ongoing inquiry without strong emotion
    - positive  = positive overall (satisfaction, resolution, appreciation)

    Analyze the **entire conversation history** provided.
    Consider how the tone evolves.
    Return a structured sentiment classification with a concise explanation.
    '''
)

async def analyze_conversation_sentiment(chat_history: List[Dict[str, Any]]):
    """
    Analyze the sentiment of the conversation context.
    """
    formatted_history = format_conversation_history(chat_history)
    input_model = ConversationAnalysisInput(conversation_text=formatted_history)

    result = await conversation_sentiment_agent.run(
        input_model,
        deps=None)

    print("Sentiment Analysis Result: ", result)

    return result.output