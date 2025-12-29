from abc import ABC, abstractmethod
from pathlib import Path


class BaseAPIClient(ABC):
    """Vision API 클라이언트 추상 클래스"""

    def __init__(self, model: str, max_tokens: int = 4096, temperature: float = 0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def call_vision(self, image_path: Path, prompt: str, detail: str = "high") -> dict:
        """
        이미지와 프롬프트로 Vision API 호출

        Args:
            image_path: 이미지 파일 경로
            prompt: 텍스트 프롬프트
            detail: 이미지 분석 상세도

        Returns:
            파싱된 JSON 응답
        """
        pass

    @staticmethod
    def parse_json_response(content: str) -> dict:
        """API 응답에서 JSON 추출 및 파싱"""
        import json

        # 마크다운 코드 블록 제거
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """이미지를 base64로 인코딩"""
        import base64

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
