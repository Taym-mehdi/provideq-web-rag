"""ProvideQ Paperclip Web RAG evidence retrieval."""

from typing import Any

__all__ = ["run_pipeline"]


def run_pipeline(*args: Any, **kwargs: Any):
    """Load the full pipeline lazily so lightweight retrieval modules stay importable."""

    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
