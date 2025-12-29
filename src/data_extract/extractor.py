import time
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from .api import BaseAPIClient
from .pdf_utils import temporary_images, pdf_to_images


def extract_questions(
    api_client: BaseAPIClient, image_path: Path, prompt: str
) -> list[dict]:
    """단일 이미지에서 문제 추출"""
    result = api_client.call_vision(image_path, prompt)
    return result.get("questions", [])


def extract_answers(
    api_client: BaseAPIClient, image_path: Path, prompt: str
) -> dict[str, int]:
    """단일 이미지에서 정답 추출"""
    result = api_client.call_vision(image_path, prompt)
    return result.get("answers", {})


def extract_from_pdf(
    api_client: BaseAPIClient,
    pdf_path: Path,
    prompt: str,
    extract_fn: Callable,
    request_delay: float = 0.5,
    save_images: bool = False,
    image_dir: Path = None,
    desc: str = "Extracting",
) -> list:
    """
    PDF에서 데이터 추출 (공통 로직)

    Args:
        api_client: API 클라이언트
        pdf_path: PDF 파일 경로
        prompt: 추출용 프롬프트
        extract_fn: 추출 함수 (extract_questions 또는 extract_answers)
        request_delay: 요청 간 딜레이 (초)
        save_images: 이미지 저장 여부
        image_dir: 이미지 저장 경로 (save_images=True일 때)
        desc: tqdm 설명

    Returns:
        추출된 데이터 리스트
    """
    all_results = []
    errors = []

    if save_images:
        if image_dir is None:
            raise ValueError("image_dir is required when save_images=True")
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = pdf_to_images(pdf_path, image_dir)

        for img_path in tqdm(image_paths, desc=desc):
            try:
                result = extract_fn(api_client, img_path, prompt)
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, dict):
                    all_results.append(result)
                time.sleep(request_delay)
            except Exception as e:
                errors.append((img_path.name, str(e)))
    else:
        with temporary_images(pdf_path) as image_paths:
            for img_path in tqdm(image_paths, desc=desc):
                try:
                    result = extract_fn(api_client, img_path, prompt)
                    if isinstance(result, list):
                        all_results.extend(result)
                    elif isinstance(result, dict):
                        all_results.append(result)
                    time.sleep(request_delay)
                except Exception as e:
                    errors.append((img_path.name, str(e)))

    if errors:
        print(f"[WARNING] {len(errors)} page(s) failed:")
        for name, err in errors[:5]:
            print(f"   - {name}: {err}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")

    return all_results


def extract_all_questions(
    api_client: BaseAPIClient,
    question_pdf: Path,
    question_prompt: str,
    request_delay: float = 0.5,
    save_images: bool = False,
    image_dir: Path = None,
) -> list[dict]:
    """문제지 PDF에서 모든 문제 추출"""
    return extract_from_pdf(
        api_client=api_client,
        pdf_path=question_pdf,
        prompt=question_prompt,
        extract_fn=extract_questions,
        request_delay=request_delay,
        save_images=save_images,
        image_dir=image_dir,
        desc="Extracting questions",
    )


def extract_all_answers(
    api_client: BaseAPIClient,
    answer_pdf: Path,
    answer_prompt: str,
    request_delay: float = 0.5,
    save_images: bool = False,
    image_dir: Path = None,
) -> dict[str, int]:
    """정답지 PDF에서 모든 정답 추출"""
    results = extract_from_pdf(
        api_client=api_client,
        pdf_path=answer_pdf,
        prompt=answer_prompt,
        extract_fn=extract_answers,
        request_delay=request_delay,
        save_images=save_images,
        image_dir=image_dir,
        desc="Extracting answers",
    )

    all_answers = {}
    for r in results:
        if isinstance(r, dict):
            all_answers.update(r)

    return all_answers
