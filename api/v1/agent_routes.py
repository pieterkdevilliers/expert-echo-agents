from fastapi import APIRouter, Depends, HTTPException
import core.auth as auth
from schemas.agent_schemas import Query, UserQuery
import util_agents.rag_query_agent as rag_agent
# from util_agents.agent_rag_query import execute_rag_agent
import util_agents.rephrase_user_query as rephrase_agent
from fastapi.responses import StreamingResponse
from agents.expert_agent import expert_agent
import json


router = APIRouter()


@router.post("/query")
async def rag_query(query: Query, authorized: bool = Depends(auth.verify_api_key)):
    """
    Accepts a query payload, runs it through the AI agent, and returns the streaming response.
    """
    if not query.query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    
    # Directly return the StreamingResponse
    return await rag_agent.query_rag_query_agent(query=query)


@router.post("/rephrase-user-query")
async def rephrase_user_query(query: UserQuery, authorized: bool = Depends(auth.verify_api_key)):
    """
    Rephrases a user query to be more specific and detailed.
    """
    if not query.query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    
    result = await rephrase_agent.rephrase_user_query(query=query.query)

    return result


# @router.post("/agent-query")
# async def query_agent(query: Query):
#     """
#     Unified endpoint for all agent interactions (RAG, Calendar, etc.)
    
#     Flow:
#     1. Rephrase the user's query for better results
#     2. Expert agent decides which tool to use
#     3. Execute the appropriate tool (streaming for RAG, JSON for others)
#     """
#     print('=' * 80)
#     print('🎯 NEW QUERY RECEIVED AT ENDPOINT')
#     print(f'Original Query: {query.query}')
#     print('=' * 80)
    
#     # Step 1: Rephrase query for better RAG results
#     rephrased_query = await rephrase_agent.rephrase_user_query(query=query.query)
#     print(f"📝 Rephrased Query: {rephrased_query}")
#     query.query = rephrased_query or query.query  # Fallback to original if rephrasing fails
    
#     try:
#         # Step 2: Expert agent decides which tool to use
#         print("\n🤔 Expert agent deciding which tool to use...")
#         result = await expert_agent.run(deps=query)
        
#         # Debug: print all messages to see what happened
#         print(f"\n🔍 Agent messages: {len(result.all_messages())}")
#         for msg in result.all_messages():
#             print(f"  - {msg.kind}: {str(msg)[:100]}")
        
#         # Look for tool calls in the messages
#         tool_result = None
#         selected_tool = None

#         for msg in result.all_messages():
#             if hasattr(msg, "parts"):
#                 for part in msg.parts:
#                     # Detect tool calls
#                     if hasattr(part, "tool_name"):
#                         print(f"\n🔧 Found tool call: {part.tool_name}")
#                         selected_tool = part.tool_name

#                     # Detect tool return (this is where the payload actually is)
#                     if part.__class__.__name__ == "ToolReturnPart":
#                         content = getattr(part, "content", None)
#                         if content:
#                             tool_result = (
#                                 json.loads(content)
#                                 if isinstance(content, str)
#                                 else content
#                             )
#                             print(f"📦 Extracted tool result: {tool_result}")
        
#         if not tool_result:
#             print("⚠️ No tool result found, using agent output")
#             tool_result = result.output if isinstance(result.output, dict) else {"tool": "unknown", "message": str(result.output)}
        
#         selected_tool_name = tool_result.get('tool') if isinstance(tool_result, dict) else 'unknown'
#         print(f"✅ Selected tool: {selected_tool_name}")
        
#         # Step 3: Execute based on the decision
#         if isinstance(tool_result, dict) and tool_result.get("tool") == "rag":
#             print("\n🧠 Executing RAG Agent with custom prompt...")
            
#             # Stream the RAG response
#             async def generate():
#                 try:
#                     async for chunk in execute_rag_agent(query):
#                         yield f"data: {json.dumps(chunk)}\n\n"
#                 except Exception as e:
#                     error_chunk = {
#                         "type": "error",
#                         "content": f"RAG agent error: {str(e)}"
#                     }
#                     yield f"data: {json.dumps(error_chunk)}\n\n"
            
#             return StreamingResponse(
#                 generate(),
#                 media_type="text/event-stream",
#                 headers={
#                     "Cache-Control": "no-cache",
#                     "Connection": "keep-alive",
#                     "X-Accel-Buffering": "no",
#                 }
#             )
        
#         elif isinstance(tool_result, dict) and tool_result.get("tool") == "calendar":
#             print("\n📅 Returning calendar response...")
#             # Return JSON for calendar queries
#             return {
#                 "status": "success",
#                 "tool": "calendar",
#                 "data": tool_result,
#                 "messages": [msg.model_dump() for msg in result.all_messages()]
#             }
        
#         else:
#             print(f"\n❌ Unknown tool selected: {tool_result}")
#             return {
#                 "status": "error",
#                 "message": f"Unknown tool selected: {tool_result}",
#                 "data": tool_result
#             }
        
#     except Exception as e:
#         print(f"\n💥 ERROR in query_agent endpoint: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "status": "error",
#             "message": f"Query processing error: {str(e)}"
#         }