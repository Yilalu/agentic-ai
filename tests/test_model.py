"""Tests for the model creation functions. we will use pytest to run these tests"""

import pytest
from src.model import create_model, create_embedding_model


def test_create_model():
    model = create_model()
    assert model is not None
    assert hasattr(model, "invoke")


def test_create_embedding_model():
    embedding_model = create_embedding_model()
    assert embedding_model is not None
    assert hasattr(embedding_model, "embed_documents")
