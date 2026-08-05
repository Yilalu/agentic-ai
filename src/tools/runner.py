"""Uniform tool invocation with retry, failure capture, and audit records.

Nodes never call a tool directly. Routing every call through `call_tool` means
one place decides how failures are retried, how they reach the graph, and how
they appear in the trace.
"""

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool

import src.config as config
from src.config import MAX_TOOL_ATTEMPST
from src.schemas import ToolCall
from src.tools.readonly import ToolError

logger = logging.getLogger(__name__)


def _summarize(result: Any) -> str:
    if isinstance(result, dict):
        return f"dict({', '.join(list(result)[:6])})"
    if isinstance(result, list):
        return f"{len(result)} record(s)"
    return str(result)[:160]


def call_tool(tool: BaseTool, args: dict, *, attempts: int | None = None) -> tuple[Any, ToolCall]:
    """Invoke a tool, retrying transient failures.

    Returns `(result, record)`. On terminal failure the result is `None` and
    `record.ok` is False, so the calling node decides whether it can continue
    without that data or must degrade.
    """

    max_attempts = attempts or MAX_TOOL_ATTEMPST

    forced = config.FORCE_TOOL_FAILURE.strip()
    if forced and forced == tool.name:
        return None, ToolCall(
            tool=tool.name,
            args=args,
            ok=False,
            error=f"injected failure via FORCE_TOOL_FAILURE={forced}",
            attempts=max_attempts,
        )

    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = tool.invoke(args)
            return result, ToolCall(
                tool=tool.name,
                args=args,
                ok=True,
                result=_summarize(result),
                attempts=attempt,
            )
        except ToolError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - any tool failure is recoverable here
            last_error = f"{type(exc).__name__}: {exc}"

        logger.warning("tool %s failed on attempt %s: %s", tool.name, attempt, last_error)
        if attempt < max_attempts:
            time.sleep(0.2 * attempt)

    return None, ToolCall(
        tool=tool.name,
        args=args,
        ok=False,
        error=last_error or "unknown tool failure",
        attempts=max_attempts,
    )
