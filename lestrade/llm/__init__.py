from .ollama import OllamaLLM
from .openai_api import OpenAILLM

_ollama = OllamaLLM()
_openai = OpenAILLM()


def get_llm(model: str):
    if model.startswith("ollama/") or model.startswith("local/"):
        return _ollama, model.split("/", 1)[1]
    return _openai, model


async def list_all_models():
    ollama_models = await _ollama.list_models()
    prefixed = [f"ollama/{m}" for m in ollama_models]
    external_models = await _openai.list_models()
    return prefixed + external_models
