import json
import logging
from typing import AsyncIterator, Any
from uuid import UUID

from google.genai.client import AsyncClient
from google.genai import types

from app.dto.chat import GenerateChatRequest
from app.utils.tools.hybrid_search import hybrid_search
from app.repositories.chat import ChatRepository

logger = logging.getLogger(__name__)

class GeminiAgentInteractor:
    """
    Interactor for chat with Gemini using a custom ReAct (Reasoning + Acting) loop.
    Enables iterative search and "thinking process" visualization.
    """

    def __init__(
        self,
        client: AsyncClient,
        chat_repository: ChatRepository,
    ):
        self.client = client
        self.chat_repository = chat_repository
        self.tools = [hybrid_search]

    async def stream(self, request: GenerateChatRequest) -> AsyncIterator[bytes]:
        """
        Stream chat response using a ReAct loop with hybrid_search tool.
        """
        
        # 1. Initialize conversation history
        session_id = request.session_id
        chat_session = None
        
        if session_id == "new":
            chat_session = await self.chat_repository.create_session()
            session_id = str(chat_session.id)
            yield self._format_event("session_info", {"session_id": session_id})
            messages = []
        else:
            try:
                uuid_obj = UUID(session_id)
                chat_session = await self.chat_repository.get_session(uuid_obj)
                if not chat_session:
                    chat_session = await self.chat_repository.create_session()
                    session_id = str(chat_session.id)
                    yield self._format_event("session_info", {"session_id": session_id})
                    messages = []
                else:
                    messages = chat_session.messages
            except ValueError:
                chat_session = await self.chat_repository.create_session()
                session_id = str(chat_session.id)
                yield self._format_event("session_info", {"session_id": session_id})
                messages = []

        # Save user message
        await self.chat_repository.add_message(
            session_id=UUID(session_id),
            role="user",
            content=request.prompt
        )

        # Build context from history
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))
        
        # Add current user message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=request.prompt)]
        ))

        # System prompt for ReAct and Citations
        system_prompt = (
            "You are an advanced document assistant. Your goal is to answer the user's question "
            "by finding and synthesizing information from the available documents.\n\n"
            "TOOLS:\n"
            "You have access to a powerful `hybrid_search` tool. "
            "You MUST use this tool to find information. Do not answer from your own knowledge "
            "unless it's general chit-chat.\n\n"
            "ITERATIVE SEARCH STRATEGY:\n"
            "1. Analyze the user's request and formulate a specific search query.\n"
            "2. Call `hybrid_search` with your query.\n"
            "3. Analyze the results. If they are relevant and sufficient, formulate your answer.\n"
            "4. IF NO RELEVANT RESULTS FOUND:\n"
            "   - Do NOT give up immediately.\n"
            "   - Try relaxing the search parameters (e.g., lower `min_confidence`, remove `document_types`).\n"
            "   - Try different keywords or synonyms.\n"
            "   - Try searching for specific entities mentioned in the request.\n"
            "   - You can make up to 3-4 search attempts with different strategies.\n\n"
            "THINKING PROCESS:\n"
            "You must explain your reasoning before each step. "
            "Explain WHY you are searching for something, or WHY you are changing your strategy.\n\n"
            "CITATIONS:\n"
            "You MUST cite your sources. When you use information from search results, cite using EXACTLY the document title "
            "as it appears in the search results, including any page numbers. For example: `[DocumentName, p.10]` or `[DocumentName, p.5]`. "
            "Use these citations at the end of sentences where you use information from that specific page. "
            "If the same document appears on multiple pages in results, use the specific page citation (e.g., `[Contract, p.10]`). "
            "The format must match EXACTLY what appears in the hybrid_search results."
        )

        model = "gemini-2.5-flash" # Using stable model
        max_steps = 10
        current_step = 0
        
        final_answer_text = ""
        search_results_map = {}  # Track search results: title -> {document_id, filename, page, bbox}
        all_search_results_data = []  # Track all search results for saving to DB
        thinking_process_data = []  # Track thinking process for saving to DB
        
        init_thought_1 = {"step": "init", "thought": "Starting agent..."}
        init_thought_2 = {"step": "init", "thought": "Analyzing request..."}
        thinking_process_data.append(init_thought_1)
        thinking_process_data.append(init_thought_2)
        yield self._format_event("thinking", init_thought_1)
        yield self._format_event("thinking", init_thought_2)

        # Tool Definition
        tools_config = [
            types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                    name="hybrid_search",
                    description="Search for information in documents using hybrid (vector + keyword) search.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "query": types.Schema(type="STRING", description="Search query"),
                            "limit": types.Schema(type="INTEGER", description="Max results"),
                            "min_confidence": types.Schema(type="NUMBER", description="Min confidence (0.0-1.0)"),
                            "document_types": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Filter by doc type"),
                            "text_weight": types.Schema(type="NUMBER", description="Weight for text search"),
                            "vector_weight": types.Schema(type="NUMBER", description="Weight for vector search"),
                        },
                        required=["query"]
                    )
                )
            ])
        ]

        while current_step < max_steps:
            current_step += 1
            logger.info(f"Step {current_step}: Generating content...")

            try:
                # Streaming generation
                response_stream = await self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=tools_config,
                        temperature=0.2,
                    )
                )

                accumulated_text = ""
                tool_calls = []
                
                async for chunk in response_stream:
                    # Robustly handle chunks
                    if chunk.candidates:
                        for candidate in chunk.candidates:
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if part.text:
                                        text_delta = part.text
                                        accumulated_text += text_delta
                                        # Only yield text if we are not in a tool call sequence (heuristically)
                                        # If we have tool calls, the text is likely "thinking"
                                        yield self._format_event("text_delta", {"delta": text_delta})
                                    
                                    if part.function_call:
                                        tool_calls.append(part.function_call)
                    elif chunk.text:
                         # Fallback
                        text_delta = chunk.text
                        accumulated_text += text_delta
                        yield self._format_event("text_delta", {"delta": text_delta})

                # Process turn
                if tool_calls:
                    # It was a tool call turn
                    logger.info(f"Tool calls detected: {len(tool_calls)}")
                    
                    # Add assistant content (thought + tool calls) to history
                    parts = []
                    if accumulated_text:
                        parts.append(types.Part(text=accumulated_text))
                    
                    for tool_call in tool_calls:
                        parts.append(types.Part(function_call=tool_call))
                        
                        # Create thinking step for tool call
                        thinking_data = {
                            "step": current_step,
                            "thought": accumulated_text if accumulated_text else f"Calling {tool_call.name}..."
                        }
                        thinking_process_data.append(thinking_data)
                        
                        # Notify frontend of tool call
                        tool_call_data = {
                            "name": tool_call.name,
                            "args": tool_call.args
                        }
                        thinking_data["tool_call"] = tool_call_data
                        yield self._format_event("tool_call", tool_call_data)

                    contents.append(types.Content(role="model", parts=parts))

                    # Execute tools
                    for tool_call in tool_calls:
                        function_name = tool_call.name
                        function_args = tool_call.args
                        
                        if function_name == "hybrid_search":
                            args_dict = {k: v for k, v in function_args.items()}
                            try:
                                logger.info(f"Executing hybrid_search with args: {args_dict}")
                                result = await hybrid_search(**args_dict)
                                
                                # Track search results for citations
                                if result and isinstance(result, dict) and "documents" in result:
                                    for doc in result["documents"]:
                                        doc_title_base = doc.get("filename", "Unknown Document")
                                        # Remove file extension for citation display
                                        if "." in doc_title_base:
                                            doc_title_base = doc_title_base.rsplit(".", 1)[0]
                                        
                                        # Group matched fields by page
                                        fields_by_page = {}
                                        if doc.get("matched_fields"):
                                            for field in doc["matched_fields"]:
                                                page_num = field.get("page_num")
                                                if page_num:
                                                    try:
                                                        page_num = int(page_num)
                                                    except (ValueError, TypeError):
                                                        continue
                                                    
                                                    if page_num not in fields_by_page:
                                                        fields_by_page[page_num] = []
                                                    fields_by_page[page_num].append(field)
                                        
                                        # Create a citation for each page
                                        for page_num, page_fields in fields_by_page.items():
                                            # Get the best field for this page
                                            best_field = max(page_fields, key=lambda f: f.get("hybrid_score", 0))
                                            
                                            # Create citation key with filename (including extension) and page number
                                            doc_title = f"{doc.get('filename')}, p.{page_num}"
                                            
                                            bbox_data = best_field.get("bbox")
                                            bbox = None
                                            if bbox_data:
                                                if isinstance(bbox_data, dict):
                                                    page_key = str(page_num)
                                                    bbox = bbox_data.get(page_key)
                                                elif isinstance(bbox_data, list):
                                                    bbox = bbox_data
                                            
                                            search_results_map[doc_title] = {
                                                "document_id": doc.get("document_id"),
                                                "filename": doc.get("filename"),
                                                "page_num": page_num,
                                                "bbox": bbox,
                                            }
                                            logger.info(f"Citation created: {doc_title} -> page {page_num}, bbox: {bbox}")
                                
                                # Store search results for DB
                                if function_name == "hybrid_search" and result:
                                    all_search_results_data.append(result)
                                
                                # Add tool result to thinking process
                                tool_result_data = {
                                    "name": function_name,
                                    "result": result
                                }
                                # Find the last thinking step and add tool result to it
                                if thinking_process_data:
                                    last_step = thinking_process_data[-1]
                                    if "tool_call" in last_step and last_step["tool_call"]["name"] == function_name:
                                        last_step["tool_result"] = tool_result_data
                                
                                # Notify frontend of result
                                yield self._format_event("tool_result", tool_result_data)
                                
                                contents.append(types.Content(
                                    role="tool",
                                    parts=[types.Part(
                                        function_response=types.FunctionResponse(
                                            name=function_name,
                                            response={"result": result}
                                        )
                                    )]
                                ))
                            except Exception as e:
                                logger.error(f"Tool execution failed: {e}")
                                yield self._format_event("error", {"error": str(e)})
                                contents.append(types.Content(
                                    role="tool",
                                    parts=[types.Part(
                                        function_response=types.FunctionResponse(
                                            name=function_name,
                                            response={"error": str(e)}
                                        )
                                    )]
                                ))
                    
                    # Continue loop to let model process tool results
                    continue
                
                else:
                    # No tool calls, this is the final answer
                    logger.info("Final answer generated.")
                    final_answer_text = accumulated_text
                    
                    # Filter citations: only include citation keys that actually appear in the response text
                    actual_citations_map = {}
                    if search_results_map and final_answer_text:
                        for citation_key, citation_data in search_results_map.items():
                            # Check if citation key appears in the response text
                            # Citation format: [DocumentName, p.10] or [DocumentName, p.5]
                            citation_pattern = f"[{citation_key}]"
                            if citation_pattern in final_answer_text:
                                actual_citations_map[citation_key] = citation_data
                                logger.info(f"Citation '{citation_key}' found in response text")
                            else:
                                logger.info(f"Citation '{citation_key}' NOT found in response text, excluding from citations")
                    
                    # Emit only actual citations before done event
                    if actual_citations_map:
                        yield self._format_event("citations", actual_citations_map)
                    
                    yield self._format_event("done", {})
                    
                    # Save model response to DB
                    if final_answer_text:
                        # Combine all search results into a single structure
                        combined_search_results = None
                        if all_search_results_data:
                            combined_search_results = {
                                "results": all_search_results_data,
                                "citations_map": actual_citations_map if actual_citations_map else {}
                            }
                        
                        # Use actual_citations_map instead of search_results_map
                        await self.chat_repository.add_message(
                            session_id=UUID(session_id),
                            role="model",
                            content=final_answer_text,
                            citations=actual_citations_map if actual_citations_map else None,
                            search_results=combined_search_results,
                            thinking_process={"steps": thinking_process_data} if thinking_process_data else None
                        )
                    break

            except Exception as e:
                logger.error(f"Gemini interaction failed: {e}")
                yield self._format_event("error", {"error": str(e)})
                return

    def _format_event(self, event_type: str, data: Any) -> bytes:
        """Format event as SSE data"""
        payload = {
            "type": event_type,
            "data": data
        }
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")
