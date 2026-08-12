import importlib

from .chunking import ChunkingPlugin, MarkdownChunking
from .ingestion import EmptyIngestion, IngestionPlugin
from .response import DefaultResponse, ResponsePlugin, set_html_format
from .retrieval import HybridRetrieval, RetrievalPlugin

_registry: dict[str, object] = {}


def _load_plugin(namespace: str, dotted_path: str, default_factory) -> object:
    cache_key = f"{namespace}::{dotted_path}"
    if cache_key in _registry:
        return _registry[cache_key]
    if not dotted_path:
        instance = default_factory()
        _registry[cache_key] = instance
        return instance
    try:
        module_path, class_name = dotted_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls()
    except Exception:
        instance = default_factory()
    _registry[cache_key] = instance
    return instance


def get_chunking() -> ChunkingPlugin:
    from .. import config
    return _load_plugin("chunking", config.CHUNKING_PLUGIN, MarkdownChunking)


def get_retrieval() -> RetrievalPlugin:
    from .. import config
    return _load_plugin("retrieval", config.RETRIEVAL_PLUGIN, HybridRetrieval)


def get_response() -> ResponsePlugin:
    from .. import config
    return _load_plugin("response", config.RESPONSE_PLUGIN, DefaultResponse)


def get_ingestion() -> IngestionPlugin:
    from .. import config
    return _load_plugin("ingestion", config.INGESTION_PLUGIN, EmptyIngestion)


__all__ = [
    "ChunkingPlugin",
    "MarkdownChunking",
    "RetrievalPlugin",
    "HybridRetrieval",
    "ResponsePlugin",
    "DefaultResponse",
    "IngestionPlugin",
    "EmptyIngestion",
    "set_html_format",
    "get_chunking",
    "get_retrieval",
    "get_response",
    "get_ingestion",
]
