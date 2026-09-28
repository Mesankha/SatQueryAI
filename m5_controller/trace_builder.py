"""
ExecutionTrace builder — immutable, logged every request.
"""
import time
import asyncio
from typing import Any, Callable, Awaitable

from shared.schemas import ExecutionTrace, ToolCall, UniformError


class TraceBuilder:
    def __init__(self, task_selected: str) -> None:
        self.trace = ExecutionTrace(task_selected=task_selected)

    def add_tool_call(self, tool_name: str, model_name: str, params: dict[str, Any], latency_ms: int, success: bool) -> None:
        self.trace.tools_called.append(
            ToolCall(
                tool_name=tool_name,
                model_name=model_name,
                params=params,
                latency_ms=latency_ms,
                success=success,
            )
        )

    def set_candidates(self, considered: int, after_filter: int) -> None:
        self.trace.candidates_considered = considered
        self.trace.candidates_after_filter = after_filter

    def set_fallback_used(self, used: bool = True) -> None:
        self.trace.fallback_used = used

    def build(self) -> ExecutionTrace:
        return self.trace


def timed_call(builder: TraceBuilder, tool_name: str, model_name: str, params: dict[str, Any], fn: Callable) -> Any:
    """Sync helper to wrap a call with timing + trace logging."""
    start = time.perf_counter()
    try:
        result = fn()
        success = True
        if isinstance(result, UniformError):
            success = False
    except Exception as exc:
        result = UniformError(module=tool_name, error_type="model_unavailable", message=str(exc), fallback_applied=False)
        success = False
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        builder.add_tool_call(tool_name, model_name, params, latency_ms, success)
    return result


async def timed_call_async(builder: TraceBuilder, tool_name: str, model_name: str, params: dict[str, Any], fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
    """Async helper to wrap an async call with timing + trace logging."""
    start = time.perf_counter()
    try:
        result = await fn(*args, **kwargs)
        success = True
        if isinstance(result, UniformError):
            success = False
    except Exception as exc:
        result = UniformError(module=tool_name, error_type="model_unavailable", message=str(exc), fallback_applied=False)
        success = False
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        builder.add_tool_call(tool_name, model_name, params, latency_ms, success)
    return result
