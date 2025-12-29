from .base import BaseAPIClient
from .openai_client import OpenAIClient

_CLIENT_REGISTRY = {
    "openai": OpenAIClient,
}


def get_api_client(api_config) -> BaseAPIClient:
    """
    Config에서 적절한 API 클라이언트 생성

    Args:
        api_config: Hydra config (api.name, api.model 등)

    Returns:
        BaseAPIClient 인스턴스
    """
    api_name = api_config.name

    if api_name not in _CLIENT_REGISTRY:
        available = ", ".join(_CLIENT_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 API: {api_name}. 사용 가능: {available}")

    client_class = _CLIENT_REGISTRY[api_name]

    return client_class(
        model=api_config.model,
        max_tokens=api_config.max_tokens,
        temperature=api_config.temperature,
        detail=api_config.get("detail", "high"),
    )


def register_api_client(name: str, client_class: type):
    """새 API 클라이언트 등록 (확장용)"""
    _CLIENT_REGISTRY[name] = client_class


__all__ = [
    "BaseAPIClient",
    "OpenAIClient",
    "get_api_client",
    "register_api_client",
]
