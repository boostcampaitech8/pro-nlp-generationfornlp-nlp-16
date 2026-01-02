from pathlib import Path
from anthropic import Anthropic
from .base import BaseAPIClient


class ClaudeClient(BaseAPIClient):
    """Claude Vision 클라이언트"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0,
        detail: str = "high",  # 사용 안 함, 호환성 유지
    ):
        super().__init__(model, max_tokens, temperature)
        self.client = Anthropic()

    def call_vision(self, image_path: Path, prompt: str, detail: str = None) -> dict:
        """Claude Vision API 호출"""
        base64_image = self.encode_image(image_path)

        # 이미지 확장자로 media_type 결정
        suffix = image_path.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        content = response.content[0].text
        return self.parse_json_response(content)
