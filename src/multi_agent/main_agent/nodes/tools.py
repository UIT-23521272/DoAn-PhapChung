import asyncio
import logging
from difflib import SequenceMatcher

from langgraph.store.base import BaseStore
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from multi_agent.common.utils import split_model_and_provider
from multi_agent.main_agent.tools.memory import upsert_memory_func
from browser import web_quick_search_func
from multi_agent.main_agent.tools.report import finalAnswerFormatter_func
from multi_agent.common.global_state import State_global
from configuration import Configuration

logger = logging.getLogger(__name__)

# Module-level web search cache to avoid redundant API calls
_web_search_cache: dict[str, tuple[str, int, int]] = {}


def _is_similar_query(query: str, past_queries: list[str], threshold: float = 0.75) -> str | None:
    """Check if query is similar to a past query. Returns the matching past query or None."""
    query_lower = query.lower().strip()
    for past_q in past_queries:
        ratio = SequenceMatcher(None, query_lower, past_q.lower().strip()).ratio()
        if ratio >= threshold:
            return past_q
    return None


async def tools(state: State_global, config: RunnableConfig, *, store: BaseStore):
    """
    Executes the tools called by the LLM.
    Allows multiple tool calls, but only one `web_quick_search` per step.
    If more than one `web_quick_search` is requested, only the first is executed,
    and an informative message is returned. 
    """
    tool_calls = state.messages[-1].tool_calls
    done = False

    if not tool_calls:
        raise ValueError("No tool calls found.")

    results = []

    input_tokens_count = state.inputTokens
    output_tokens_count = state.outputTokens

    # Handle upsert_memory
    upsert_calls = [tc for tc in tool_calls if tc["name"] == "upsert_memory"]
    if upsert_calls:
        saved_memories = await asyncio.gather(
            *[upsert_memory_func(**tc["args"], store=store) for tc in upsert_calls]
        )
        results.extend([
            {
                "role": "tool",
                "content": mem,
                "tool_call_id": tc["id"],
            }
            for tc, mem in zip(upsert_calls, saved_memories)
        ])

    # Handle web_quick_search (only one allowed) with caching and deduplication
    web_calls = [tc for tc in tool_calls if tc["name"] == "web_quick_search"]
    past_queries = list(state.past_search_queries) if state.past_search_queries else []
    
    if web_calls:
        first_web_call = web_calls[0]
        query_used = first_web_call["args"].get("query", "unknown")
        
        # Check for similar past query (deduplication)
        similar_query = _is_similar_query(query_used, past_queries)
        
        if similar_query and similar_query in _web_search_cache:
            # Return cached result for similar query
            cached_response, cached_in, cached_out = _web_search_cache[similar_query]
            response_with_query = (
                f"[CACHED - similar to previous query: '{similar_query}']\n"
                f"Search result for query: '{query_used}'\n{cached_response}"
            )
            logger.info(f"Web search cache hit: '{query_used}' similar to '{similar_query}'")
            results.append({
                "role": "tool",
                "content": response_with_query,
                "tool_call_id": first_web_call["id"],
            })
        elif query_used in _web_search_cache:
            # Exact cache hit
            cached_response, cached_in, cached_out = _web_search_cache[query_used]
            response_with_query = (
                f"[CACHED] Search result for query: '{query_used}'\n{cached_response}"
            )
            results.append({
                "role": "tool",
                "content": response_with_query,
                "tool_call_id": first_web_call["id"],
            })
        else:
            # No cache hit — perform actual search
            configurable = Configuration.from_runnable_config(config)
            llm = init_chat_model(**split_model_and_provider(configurable.model),timeout=100)
            
            (response, inCount, outCount) = web_quick_search_func(**first_web_call["args"], llm_model=llm, strategy=state.strategy,context_window_size= configurable.context_window_size)
            input_tokens_count += inCount
            output_tokens_count += outCount
            
            # Cache the result
            _web_search_cache[query_used] = (response, inCount, outCount)
            past_queries.append(query_used)
            
            response_with_query = (
                f"Search result for query: '{query_used}'\n{response}"
            )
            results.append({
                "role": "tool",
                "content": response_with_query,
                "tool_call_id": first_web_call["id"],
            })

        if len(web_calls) > 1:
            skipped_calls = len(web_calls) - 1
            skipped_calls_content = '\n'.join([text["args"].get("query","unknown") for text in web_calls[1:]])
            results.append({
                "role": "tool",
                "content": f"Only one web_quick_search is allowed per step. "
                           f"{skipped_calls} additional call(s) were skipped.\n"
                           f"Skipped call(s): {skipped_calls_content}",
                "tool_call_id": web_calls[1]["id"]  
            }) 

    #Handles formatting of the final answer
    final_answer_calls = [tc for tc in tool_calls if tc["name"] == "final_answer_formatter"]
    if len(final_answer_calls) > 1:
        raise(ValueError('Final answer formatting called more than once'))
    if final_answer_calls:
        formatted_answers = [
            finalAnswerFormatter_func(**tc["args"])
            for tc in final_answer_calls
        ]
        done = True # Set the done flag to True, the multi_agent.main_agent has finished its task
        results.extend([
            {
                "role": "tool",
                "content": f"{content}",
                "tool_call_id": tc["id"],
            }
            for tc, content in zip(final_answer_calls, formatted_answers)
        ])

    return {
        "messages": results,
        "steps": state.steps - 1,
        "done": done,
        "inputTokens": input_tokens_count,
        "outputTokens": output_tokens_count,
        "past_search_queries": past_queries,
    }

__all__ = ["tools"]