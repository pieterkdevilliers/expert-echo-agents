import logfire
import os
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
    history: dict[str, str] = {}


class QuerySentiment(BaseModel):
    """
    Sentiment analysis result.
    """
    sentiment: str
    explanation: str

initial_user_query_sentiment_agent = Agent[QueryContext, str](
    'openai:gpt-4o',
    output_type=QuerySentiment,
    instructions='''
    You are an expert at analyzing the sentiment of user queries.
    Given a user query, determine its sentiment (e.g., positive, negative, neutral)
    and provide a brief explanation for your assessment.
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


conversation_sentiment_agent = Agent[QueryContext, str](
    'openai:gpt-4o',
    output_type=QuerySentiment,
    instructions='''
    You are an expert at analyzing the sentiment of user queries.
    Given a user query, determine its sentiment (e.g., positive, negative, neutral)
    and provide a brief explanation for your assessment.
    Ensure your analysis is concise and relevant to the query's context.'''
)

async def analyze_conversation_sentiment(history: dict[str, str]):
    """
    Analyze the sentiment of the conversation context.
    """
    context = Conversation(history=history)

    print("Conversation History for Sentiment Analysis: ", history)

    result = await conversation_sentiment_agent.run(history, deps=context)

    print("Sentiment Analysis Result: ", result)

    return result.output