from hashlib import sha1
import json
import copy
import time
from typing import AsyncIterator, Sequence

from agents import Agent, Runner, SQLiteSession

from app.dto.chat import GenerateChatRequest
from app.entities.documents import Document
from app.repositories.documents import DocumentsRepository
from app.utils.tools.hybrid_search import hybrid_search
from app.core.config import settings


class GenerateChatResponseInteractor:

    _session_logs: dict[str, dict] = {}

    def __init__(
        self,
        documents_repository: DocumentsRepository,
    ):
        self.documents_repository = documents_repository
        self.agent: Agent | None = None
        self.agent_model: str | None = None

    def _get_agent(self, model: str) -> Agent:
        if self.agent is None or self.agent_model != model:
            self.agent = Agent(
                name="document_analyst",
                model=model,
                instructions=(
                    "You are an insightful document analyst conversing with a user. "
                    "Use the `hybrid_search` tool to explore relevant document chunks. "
                    "IMPORTANT: If the first search doesn't find the answer, try different query variations: "
                    "- Use synonyms or related terms\n"
                    "- Search for specific entities (names, companies, dates)\n"
                    "- Try broader or narrower queries\n"
                    "- Use document_ids parameter to focus on specific documents mentioned\n"
                    "Call the tool multiple times with different queries until you find the answer or exhaust all possibilities. "
                    "Increase the limit parameter (up to 20) if needed to see more results. "
                    "Base answers only on verified context from the search results. "
                    "If information is truly missing after multiple searches, clearly state that.\n\n"
                    "CRITICAL: Always format your responses in Markdown. Use proper Markdown syntax for:\n"
                    "- Tables: Use pipe-separated format with header row and separator row (| Header | Header |)\n"
                    "- Headers: Use #, ##, ### for different levels\n"
                    "- Lists: Use - or * for unordered, numbers for ordered\n"
                    "- Code: Use ``` for blocks, ` for inline\n"
                    "- Bold/Italic: Use **bold** and *italic*\n"
                    "Ensure all tables are properly formatted with complete header rows and separator rows."
                ),
                tools=[hybrid_search],
            )
            self.agent_model = model
        return self.agent

    def _build_session_id(self, provided_id: str | None, document_ids: Sequence[int]) -> str:
        if provided_id:
            return provided_id

        # Генерируем уникальный ID для нового чата
        # Добавляем timestamp для уникальности, чтобы каждый новый чат имел свой ID
        raw = "|".join(str(doc_id) for doc_id in sorted(document_ids)) if document_ids else "global"
        timestamp = int(time.time() * 1000)  # миллисекунды для большей уникальности
        unique_raw = f"{raw}|{timestamp}"
        digest = sha1(unique_raw.encode("utf-8")).hexdigest()[:16]
        return f"docs-session-{digest}"

    async def _describe_documents(self, document_ids: Sequence[int]) -> str:
        if not document_ids:
            return ""

        documents = await self.documents_repository.get_all(
            where=[Document.id.in_(document_ids)]
        )

        if not documents:
            return ", ".join(str(doc_id) for doc_id in document_ids)

        lines = [f"- ID {doc.id}: {doc.filename}" for doc in documents]
        return "\n".join(lines)

    @staticmethod
    def _extract_text_delta(event_payload: dict) -> str:
        if not isinstance(event_payload, dict):
            return ""

        data = event_payload.get("data")
        if isinstance(data, dict):
            if data.get("type") == "response.output_text.delta":
                delta = data.get("delta")
                if isinstance(delta, str):
                    return delta
                if isinstance(delta, dict):
                    return delta.get("text", "")
                if isinstance(delta, list):
                    return "".join(
                        fragment.get("text", "")
                        for fragment in delta
                        if isinstance(fragment, dict)
                    )
        return ""

    @staticmethod
    def _event_to_dict(event) -> dict:
        """Преобразует событие в словарь"""
        try:
            # Пробуем model_dump если это Pydantic модель
            if hasattr(event, 'model_dump'):
                return event.model_dump()
        except Exception:
            pass

        try:
            # Пробуем dict() если объект поддерживает преобразование
            if hasattr(event, '__dict__'):
                result = {}
                for key, value in event.__dict__.items():
                    # Рекурсивно преобразуем вложенные объекты
                    if hasattr(value, '__dict__'):
                        result[key] = GenerateChatResponseInteractor._event_to_dict(value)
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        result[key] = value
                    elif isinstance(value, (list, tuple)):
                        result[key] = [
                            GenerateChatResponseInteractor._event_to_dict(item) 
                            if hasattr(item, '__dict__') else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        result[key] = {
                            k: GenerateChatResponseInteractor._event_to_dict(v) 
                            if hasattr(v, '__dict__') else v
                            for k, v in value.items()
                        }
                    else:
                        result[key] = str(value)
                return result
        except Exception:
            pass

        # Если ничего не помогло, возвращаем строковое представление
        return {"_raw": str(event), "type": getattr(event, 'type', 'unknown')}

    @staticmethod
    def _is_text_delta_event(event_dict: dict) -> bool:
        """Проверяет, является ли событие событием text_delta"""
        data = event_dict.get("data", {})
        if isinstance(data, dict):
            return data.get("type") == "response.output_text.delta"
        # Также проверяем тип события напрямую
        event_type = event_dict.get("type", "")
        return "output_text.delta" in str(event_type) or "text.delta" in str(event_type)

    @staticmethod
    def _is_technical_event(event_dict: dict, event_type: str) -> bool:
        """Проверяет, является ли событие техническим (не нужно показывать пользователю)"""
        # Проверяем run_item_stream_event - не фильтруем если это tool call/result
        event_name = event_dict.get("name", "")
        if event_name in ("tool_called", "tool_output"):
            return False
        
        item = event_dict.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type in ("tool_call_item", "tool_call_output_item"):
                return False
        
        # Проверяем, является ли это tool call или tool result через response.output_item.added
        # Такие события НЕ должны фильтроваться, даже если они raw_response_event
        data = event_dict.get("data", {})
        if isinstance(data, dict):
            if data.get("type") == "response.output_item.added":
                item = data.get("item", {})
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    # Tool calls и tool results не фильтруем
                    if item_type in ("function_call", "function_call_output", "tool_output", "tool_result"):
                        return False
        
        # Технические события, которые не нужно показывать
        # НО raw_response_event может содержать tool calls/results, поэтому проверяем содержимое
        if event_type == "raw_response_event":
            # Если это raw_response_event, проверяем содержимое
            if isinstance(data, dict):
                data_type = data.get("type", "")
                # Если это tool call/result или другой важный тип, не фильтруем
                if data_type == "response.output_item.added":
                    return False
            # Иначе фильтруем как техническое
            return True
        
        # run_item_stream_event с message_output_created - не техническое, но не tool call/result
        if event_type == "run_item_stream_event":
            # Если это не tool call/result, фильтруем
            if event_name not in ("tool_called", "tool_output"):
                return True
        
        technical_types = [
            "agent_updated_stream_event",
            "response.created",
            "response.in_progress",
        ]
        
        if event_type in technical_types:
            return True
        
        # Проверяем тип в data
        if isinstance(data, dict):
            data_type = data.get("type", "")
            # События аргументов функции - это часть tool call, обрабатываем отдельно
            if "function_call_arguments.delta" in data_type:
                return True
            # Технические события response
            if data_type in ("response.created", "response.in_progress"):
                return True
        
        return False

    @staticmethod
    def _is_tool_call_event(event_dict: dict, event_type: str) -> bool:
        """Проверяет, является ли событие событием tool call"""
        # Проверяем run_item_stream_event с tool_called
        event_name = event_dict.get("name", "")
        if event_name == "tool_called":
            return True
        
        # Проверяем item в событии
        item = event_dict.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type in ("tool_call_item", "function_call"):
                return True
            # Проверяем raw_item
            raw_item = item.get("raw_item", {})
            if isinstance(raw_item, dict) and raw_item.get("type") == "function_call":
                return True
        
        data = event_dict.get("data", {})
        if isinstance(data, dict):
            data_type = data.get("type", "")
            # Событие добавления tool call
            if data_type == "response.output_item.added":
                item = data.get("item", {})
                if isinstance(item, dict):
                    return item.get("type") == "function_call"
            # Прямое событие tool call
            if "function_call" in data_type or "tool_call" in data_type:
                return True
        
        return "tool" in event_type.lower() and ("call" in event_type.lower() or "use" in event_type.lower())

    @staticmethod
    def _is_tool_result_event(event_dict: dict, event_type: str) -> bool:
        """Проверяет, является ли событие результатом tool call"""
        # Проверяем run_item_stream_event с tool_output
        event_name = event_dict.get("name", "")
        if event_name == "tool_output":
            return True
        
        # Проверяем item в событии
        item = event_dict.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type in ("tool_call_output_item", "function_call_output", "tool_output", "tool_result"):
                return True
            # Проверяем raw_item
            raw_item = item.get("raw_item", {})
            if isinstance(raw_item, dict) and raw_item.get("type") == "function_call_output":
                return True
        
        data = event_dict.get("data", {})
        if isinstance(data, dict):
            data_type = data.get("type", "")
            # Событие добавления результата tool call
            if data_type == "response.output_item.added":
                item = data.get("item", {})
                if isinstance(item, dict):
                    return item.get("type") in ("function_call_output", "tool_output", "tool_result")
            # Прямое событие результата tool call
            if "function_call_output" in data_type or "tool_output" in data_type or "tool_result" in data_type:
                return True
        
        return "tool" in event_type.lower() and ("result" in event_type.lower() or "output" in event_type.lower())

    @staticmethod
    def _is_final_response_event(event_dict: dict) -> bool:
        """Проверяет, является ли событие финальным ответом"""
        data = event_dict.get("data", {})
        if isinstance(data, dict):
            return data.get("type") in ("response.output_text.done", "response.done")
        return False

    async def stream(self, request: GenerateChatRequest) -> AsyncIterator[bytes]:
        agent = self._get_agent(request.model)

        session_id = self._build_session_id(request.session_id, request.document_ids)
        # Используем файловую базу данных вместо in-memory для сохранения истории
        session = SQLiteSession(session_id, db_path=str(settings.sessions_db_path))

        document_hint = await self._describe_documents(request.document_ids)

        user_prompt = request.prompt
        if document_hint:
            user_prompt = (
                f"{request.prompt}\n\n"
                "Document scope available for this chat:\n"
                f"{document_hint}\n"
                "Leverage the `hybrid_search` tool to inspect or refine these documents."
            )

        run_stream = Runner.run_streamed(agent, user_prompt, session=session)

        logs: list[dict] = []
        text_fragments: list[str] = []

        # Для группировки аргументов tool calls
        current_tool_call = None
        tool_call_args_buffer = []
        buffered_tool_call = None  # Буферизованное событие tool call для отправки
        sent_tool_call_ids = set()  # Отслеживаем отправленные tool calls по ID

        async for event in run_stream.stream_events():
            event_dict = self._event_to_dict(event)
            
            # Получаем тип события
            event_type = getattr(event, 'type', event_dict.get('type', 'unknown'))
            
            # Пропускаем технические события
            if self._is_technical_event(event_dict, event_type):
                # Сохраняем в логи для истории, но не отправляем пользователю
                logs.append(event_dict)
                continue
            
            # Проверяем, является ли это событием tool call или tool result
            is_tool_call = self._is_tool_call_event(event_dict, event_type)
            is_tool_result = self._is_tool_result_event(event_dict, event_type)
            
            # Обрабатываем аргументы tool call (группируем их)
            data = event_dict.get("data", {})
            is_args_delta = isinstance(data, dict) and data.get("type") == "response.function_call_arguments.delta"
            
            if is_args_delta:
                delta = data.get("delta", "")
                if current_tool_call and buffered_tool_call:
                    tool_call_args_buffer.append(delta)
                logs.append(event_dict)
                continue
            
            # Если накопились аргументы и есть буферизованное событие tool call, отправляем его
            # Но только если это НЕ новый tool call (чтобы избежать двойной отправки)
            if buffered_tool_call and tool_call_args_buffer and not is_tool_call:
                args_str = "".join(tool_call_args_buffer)
                try:
                    args_dict = json.loads(args_str) if args_str else {}
                    if "data" in buffered_tool_call and isinstance(buffered_tool_call["data"], dict):
                        if "item" not in buffered_tool_call["data"]:
                            buffered_tool_call["data"]["item"] = {}
                        buffered_tool_call["data"]["item"]["arguments"] = args_dict
                except:
                    pass
                
                # Проверяем, был ли этот tool_call уже отправлен
                prev_tool_call_id = None
                if isinstance(buffered_tool_call.get("item"), dict):
                    prev_tool_call_id = buffered_tool_call["item"].get("id") or buffered_tool_call["item"].get("call_id")
                
                if prev_tool_call_id not in sent_tool_call_ids:
                    # Отправляем буферизованное событие tool call с аргументами
                    payload = {
                        "type": "tool_call",
                        "event": buffered_tool_call,
                    }
                    chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                    yield chunk.encode("utf-8")
                    if prev_tool_call_id:
                        sent_tool_call_ids.add(prev_tool_call_id)
                
                tool_call_args_buffer = []
                buffered_tool_call = None
                current_tool_call = None
                logs.append(event_dict)
                continue
            elif buffered_tool_call and not is_args_delta and not is_tool_call:
                # Если есть буферизованное событие tool call, но следующее событие не delta аргументов и не новый tool call,
                # отправляем tool call без аргументов (или с пустыми аргументами)
                payload = {
                    "type": "tool_call",
                    "event": buffered_tool_call,
                }
                chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                yield chunk.encode("utf-8")
                
                tool_call_args_buffer = []
                buffered_tool_call = None
                current_tool_call = None
            
            # Обрабатываем tool call события
            if is_tool_call:
                # Сначала обрабатываем предыдущий буферизованный tool call, если есть
                if buffered_tool_call:
                    # Если есть накопленные аргументы, добавляем их
                    if tool_call_args_buffer:
                        args_str = "".join(tool_call_args_buffer)
                        try:
                            args_dict = json.loads(args_str) if args_str else {}
                            if "data" in buffered_tool_call and isinstance(buffered_tool_call["data"], dict):
                                if "item" not in buffered_tool_call["data"]:
                                    buffered_tool_call["data"]["item"] = {}
                                buffered_tool_call["data"]["item"]["arguments"] = args_dict
                        except:
                            pass
                    
                    # Отправляем предыдущий tool call только если он еще не был отправлен
                    prev_tool_call_id = None
                    if isinstance(buffered_tool_call.get("item"), dict):
                        prev_tool_call_id = buffered_tool_call["item"].get("id") or buffered_tool_call["item"].get("call_id")
                    
                    if prev_tool_call_id not in sent_tool_call_ids:
                        payload = {
                            "type": "tool_call",
                            "event": buffered_tool_call,
                        }
                        chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                        yield chunk.encode("utf-8")
                        if prev_tool_call_id:
                            sent_tool_call_ids.add(prev_tool_call_id)
                    
                    # Очищаем буфер
                    tool_call_args_buffer = []
                    buffered_tool_call = None
                    current_tool_call = None
                
                # Теперь обрабатываем новый tool call
                # Нормализуем тип события для tool calls
                if event_type == "raw_response_event" or event_type == "run_item_stream_event":
                    event_type = "tool_call"
                
                # Извлекаем аргументы из raw_item если они есть
                item = event_dict.get("item", {})
                if isinstance(item, dict):
                    raw_item = item.get("raw_item", {})
                    if isinstance(raw_item, dict):
                        arguments_str = raw_item.get("arguments", "")
                        if arguments_str and isinstance(arguments_str, str):
                            try:
                                arguments_dict = json.loads(arguments_str)
                                if "data" not in event_dict:
                                    event_dict["data"] = {}
                                if "item" not in event_dict["data"]:
                                    event_dict["data"]["item"] = {}
                                event_dict["data"]["item"]["arguments"] = arguments_dict
                            except:
                                pass
                
                # Сохраняем событие tool call для буферизации
                current_tool_call = event_dict
                buffered_tool_call = copy.deepcopy(event_dict)
                
                # Проверяем, есть ли уже аргументы
                has_arguments = buffered_tool_call.get("data", {}).get("item", {}).get("arguments")
                
                # Получаем ID tool call для отслеживания дубликатов
                tool_call_id = None
                if isinstance(item, dict):
                    tool_call_id = item.get("id") or item.get("call_id")
                
                # Если аргументы уже есть, отправляем сразу и очищаем буфер
                # Но только если этот tool_call еще не был отправлен
                if has_arguments:
                    if tool_call_id not in sent_tool_call_ids:
                        payload = {
                            "type": "tool_call",
                            "event": buffered_tool_call,
                        }
                        chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                        yield chunk.encode("utf-8")
                        if tool_call_id:
                            sent_tool_call_ids.add(tool_call_id)
                    buffered_tool_call = None
                    current_tool_call = None
                    tool_call_args_buffer = []
                else:
                    # Ждем аргументы, сохраняем в логи но не отправляем пока
                    logs.append(event_dict)
                    continue
            elif buffered_tool_call and not is_args_delta and not is_tool_call and not is_tool_result:
                # Если есть буферизованное событие tool call, но следующее событие не delta аргументов и не новый tool call,
                # отправляем tool call без аргументов (или с пустыми аргументами)
                payload = {
                    "type": "tool_call",
                    "event": buffered_tool_call,
                }
                chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                yield chunk.encode("utf-8")
                
                tool_call_args_buffer = []
                buffered_tool_call = None
                current_tool_call = None
            elif is_tool_result:
                # Нормализуем тип события для tool results
                if event_type == "raw_response_event" or event_type == "run_item_stream_event":
                    event_type = "tool_result"
                
                # Извлекаем результат из item или raw_item
                item = event_dict.get("item", {})
                if isinstance(item, dict):
                    # Проверяем output в item
                    if "output" in item:
                        if "data" not in event_dict:
                            event_dict["data"] = {}
                        event_dict["data"]["result"] = item["output"]
                    # Проверяем raw_item.output
                    raw_item = item.get("raw_item", {})
                    if isinstance(raw_item, dict) and "output" in raw_item:
                        output_str = raw_item.get("output", "")
                        if isinstance(output_str, str):
                            try:
                                # Пробуем распарсить как JSON
                                output_dict = json.loads(output_str.replace("'", '"'))
                                if "data" not in event_dict:
                                    event_dict["data"] = {}
                                event_dict["data"]["result"] = output_dict
                            except:
                                if "data" not in event_dict:
                                    event_dict["data"] = {}
                                event_dict["data"]["result"] = output_str
            
            # Извлекаем text_delta только для финального ответа
            text_delta = ""
            is_text_delta = self._is_text_delta_event(event_dict)
            if is_text_delta:
                text_delta = self._extract_text_delta(event_dict)
                if text_delta:
                    text_fragments.append(text_delta)
            
            # Сохраняем все события в логи (включая text_delta для истории)
            logs.append(event_dict)

            # Отправляем событие только если это НЕ text_delta событие
            # text_delta события отправляются отдельно для обновления текста
            if not is_text_delta:
                # Отправляем все события кроме text_delta (tool calls, reasoning, etc.)
                # Используем нормализованный тип события
                payload = {
                    "type": event_type,
                    "event": event_dict,
                }
                chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                yield chunk.encode("utf-8")
            elif text_delta:
                # Отправляем text_delta отдельно для обновления финального текста
                # НЕ добавляем это событие в секцию мышления
                # Отправляем сразу, чтобы пользователь видел текст постепенно
                payload = {
                    "type": "text_delta",
                    "text_delta": text_delta,
                    "event": event_dict,
                }
                chunk = json.dumps(payload, ensure_ascii=False) + "\n"
                yield chunk.encode("utf-8")
                # Flush буфер для немедленной отправки
                import sys
                if hasattr(sys.stdout, 'flush'):
                    sys.stdout.flush()
        
        # Завершаем последний tool call если есть
        if buffered_tool_call and tool_call_args_buffer:
            args_str = "".join(tool_call_args_buffer)
            try:
                args_dict = json.loads(args_str) if args_str else {}
                if "data" in buffered_tool_call and isinstance(buffered_tool_call["data"], dict):
                    if "item" in buffered_tool_call["data"]:
                        buffered_tool_call["data"]["item"]["arguments"] = args_dict
            except:
                pass
            
            # Отправляем буферизованное событие tool call с аргументами
            payload = {
                "type": "tool_call",
                "event": buffered_tool_call,
            }
            chunk = json.dumps(payload, ensure_ascii=False) + "\n"
            yield chunk.encode("utf-8")

        final_output = getattr(run_stream, "final_output", None)
        if final_output is None:
            final_output = "".join(text_fragments)

        self._session_logs[session_id] = {
            "request": request.model_dump() if hasattr(request, 'model_dump') else request.dict() if hasattr(request, 'dict') else str(request),
            "events": logs,
            "final_output": final_output,
        }

        final_payload = {
            "type": "final",
            "session_id": session_id,
            "final_output": final_output,
        }

        yield (json.dumps(final_payload, ensure_ascii=False) + "\n").encode("utf-8")

    def get_session_log(self, session_id: str) -> dict | None:
        return self._session_logs.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Удаляет сессию из логов"""
        if session_id in self._session_logs:
            del self._session_logs[session_id]
            return True
        return False

    async def execute(self, request: GenerateChatRequest) -> dict:
        session_id = self._build_session_id(request.session_id, request.document_ids)

        final_payload = self._session_logs.get(session_id)
        if final_payload:
            return {
                "answer": final_payload.get("final_output"),
                "session_id": session_id,
                "context": [],
            }

        chunks: list[str] = []

        async for chunk in self.stream(request):
            chunks.append(chunk.decode("utf-8"))

        stored = self._session_logs.get(session_id, {})

        return {
            "answer": stored.get("final_output"),
            "session_id": session_id,
            "context": [],
            "stream": chunks,
        }
