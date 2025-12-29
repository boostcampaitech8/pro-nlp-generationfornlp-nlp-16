from pathlib import Path
from openai import OpenAI
from .base import BaseAPIClient


class OpenAIClient(BaseAPIClient):
    """OpenAI GPT-4o Vision 클라이언트"""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0,
        detail: str = "high",
    ):
        super().__init__(model, max_tokens, temperature)
        self.detail = detail
        self.client = OpenAI()

    def call_vision(self, image_path: Path, prompt: str, detail: str = None) -> dict:
        """OpenAI Vision API 호출"""
        if detail is None:
            detail = self.detail

        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content
        return self.parse_json_response(content)
