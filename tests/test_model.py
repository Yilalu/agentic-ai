"""Tests for the model creation functions. we will use pytest to run these tests"""

import pytest
from src.model import create_model, create_embedding_model

def test_create_model():
    """Test that the model is created successfully."""
    model = create_model()
    assert model is not None
    assert hasattr(model, "invoke")  # Check if the model has an invoke method
def test_create_embedding_model():
    """Test that the embedding model is created successfully."""
    embedding_model = create_embedding_model()
    assert embedding_model is not None
    assert hasattr(embedding_model, "embed_documents")  # Check if the embedding model has an embed_documents method
