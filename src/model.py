"""Chat model helpers for Gemini structured outputs."""

from dotenv import load_dotenv
from typing import TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from . import config
from .config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME, GEMINI_MODEL

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


def _check_model_and_embedding_model():
    load_dotenv()  # Load environment variables from .env file
    if not GEMINI_API_KEY:
        raise LLMUnavailable(
            "GEMINI_API_KEY is not set in the environment variables or configuration error."
        )
    if not EMBEDDING_MODEL_NAME:
        raise ValueError(
            "EMBEDDING_MODEL_NAME is not set in the environment variables or configuration error."
        )


def create_model():
    _check_model_and_embedding_model()
    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
    )
    return model


def create_embedding_model():
    _check_model_and_embedding_model()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embedding_model


def invoke_structured(prompt: ChatPromptTemplate, schema: type[T], inputs: dict) -> T:
    if config.FORCE_LLM_FAILURE:
        raise LLMUnavailable("injected failure via FORCE_LLM_FAILURE")
    if not GEMINI_API_KEY:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    try:
        model = create_model().with_structured_output(schema)
        result = (prompt | model).invoke(inputs)
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(str(exc)) from exc

    if not isinstance(result, schema):
        raise LLMUnavailable(f"unexpected model result: {type(result)!r}")
    return result


def _as_text(content: object) -> str:
    """Normalize a chat model's `.content`.

    Some models return a plain string; others return a list of content
    blocks (e.g. `[{"type": "text", "text": "..."}]`) even for plain text
    replies. Blindly `str()`-ing the latter would print the list's Python
    repr, braces and all, straight to the customer.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts) if parts else str(content)
    return str(content)


def invoke_text(prompt: ChatPromptTemplate, inputs: dict) -> str:
    """Plain-text model call, for roles that don't need a structured schema."""

    if config.FORCE_LLM_FAILURE:
        raise LLMUnavailable("injected failure via FORCE_LLM_FAILURE")
    if not GEMINI_API_KEY:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    try:
        result = (prompt | create_model()).invoke(inputs)
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(str(exc)) from exc

    return _as_text(result.content)
