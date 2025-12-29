import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

import pypdfium2 as pdfium


def pdf_to_images(
    pdf_path: Path, output_dir: Path, scale: float = 2.0, prefix: str = None
) -> list[Path]:
    """
    PDF를 페이지별 이미지로 변환

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
        scale: 렌더링 스케일 (높을수록 고해상도)
        prefix: 파일명 접두사 (없으면 PDF 파일명 사용)

    Returns:
        생성된 이미지 파일 경로 리스트
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = pdf_path.stem

    pdf = pdfium.PdfDocument(str(pdf_path))
    image_paths = []

    try:
        for i, page in enumerate(pdf):
            image = page.render(scale=scale).to_pil()
            image_path = output_dir / f"{prefix}_page_{i+1:02d}.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
    finally:
        pdf.close()

    return image_paths


@contextmanager
def temporary_images(
    pdf_path: Path, scale: float = 2.0
) -> Generator[list[Path], None, None]:
    """
    PDF를 임시 이미지로 변환 (사용 후 자동 삭제)

    Args:
        pdf_path: PDF 파일 경로
        scale: 렌더링 스케일

    Yields:
        임시 이미지 파일 경로 리스트

    Example:
        with temporary_images(pdf_path) as images:
            for img in images:
                process(img)
        # 자동으로 임시 파일 삭제됨
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="pdf_extract_"))

    try:
        image_paths = pdf_to_images(pdf_path, temp_dir, scale=scale)
        yield image_paths
    finally:
        # 임시 디렉토리 전체 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_pdf_page_count(pdf_path: Path) -> int:
    """PDF 페이지 수 반환"""
    pdf = pdfium.PdfDocument(str(pdf_path))
    count = len(pdf)
    pdf.close()
    return count
