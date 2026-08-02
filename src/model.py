from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from .config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME, GEMINI_MODEL

def _check_model_and_embedding_model():
    load_dotenv()  # Load environment variables from .env file
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables or configuration error.")
    if not EMBEDDING_MODEL_NAME:
        raise ValueError("EMBEDDING_MODEL_NAME is not set in the environment variables or configuration error.")

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