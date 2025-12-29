"""
PDF 시험지 -> 데이터셋 변환 스크립트 (Batch 모드)

사용법:
    # config 파일에 exams 리스트 설정 후 실행
    uv run python extract.py

    # CLI에서 직접 지정 (단일 파일)
    uv run python extract.py \
        'exams=[{question_pdf: data/raw/q.pdf, answer_pdf: data/raw/a.pdf, exam_id: test}]'

    # 다른 시험 타입
    uv run python extract.py exam=sat_korean

    # PDF 입력용 변환 이미지 저장 (기본값: 비활성화)
    uv run python extract.py options.save_images=true
"""

from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_extract import (
    get_api_client,
    extract_all_questions,
    extract_all_answers,
    merge_questions_and_answers,
    convert_to_train_format,
    save_dataset,
    validate_dataset,
)


def process_single_exam(
    cfg: DictConfig,
    api_client,
    question_pdf: Path,
    answer_pdf: Path,
    exam_id: str,
    output_dir: Path,
    image_dir: Path = None,
) -> pd.DataFrame:
    """단일 시험지 처리"""

    print("")
    print("-" * 40)
    print(f"Processing: {exam_id}")
    print("-" * 40)
    print(f"  Question PDF: {question_pdf}")
    print(f"  Answer PDF: {answer_pdf}")

    print("")
    print("  [1/3] Extracting answers...")
    answers = extract_all_answers(
        api_client=api_client,
        answer_pdf=answer_pdf,
        answer_prompt=cfg.exam.answer_prompt,
        request_delay=cfg.api_common.request_delay,
        save_images=cfg.options.save_images,
        image_dir=image_dir / exam_id / "answers" if image_dir else None,
    )
    print(f"    -> {len(answers)} answers extracted")

    print("")
    print("  [2/3] Extracting questions...")
    questions = extract_all_questions(
        api_client=api_client,
        question_pdf=question_pdf,
        question_prompt=cfg.exam.question_prompt,
        request_delay=cfg.api_common.request_delay,
        save_images=cfg.options.save_images,
        image_dir=image_dir / exam_id / "questions" if image_dir else None,
    )
    print(f"    -> {len(questions)} questions extracted")

    print("")
    print("  [3/3] Building dataset...")
    merged = merge_questions_and_answers(questions, answers, exam_id)
    df = convert_to_train_format(merged)

    if cfg.output.keep_individual:
        individual_path = output_dir / f"{exam_id}.csv"
        save_dataset(df, individual_path)

    issues = validate_dataset(df, choices_count=cfg.exam.choices_count)
    if issues:
        print(f"  [WARNING] {len(issues)} issues found:")
        for issue in issues[:5]:
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print("  -> Validation passed")

    return df


@hydra.main(version_base=None, config_path="conf/extraction", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if not cfg.exams or len(cfg.exams) == 0:
        raise ValueError("exams list is empty. Please configure exams in config.yaml")

    valid_exams = []
    for exam in cfg.exams:
        if not exam.question_pdf or not exam.answer_pdf:
            print(f"[SKIP] Missing PDF path: {exam}")
            continue

        question_pdf = Path(hydra.utils.to_absolute_path(exam.question_pdf))
        answer_pdf = Path(hydra.utils.to_absolute_path(exam.answer_pdf))

        if not question_pdf.exists():
            print(f"[SKIP] Question PDF not found: {question_pdf}")
            continue
        if not answer_pdf.exists():
            print(f"[SKIP] Answer PDF not found: {answer_pdf}")
            continue

        exam_id = exam.exam_id or question_pdf.stem.replace("_문제", "").replace(
            "_question", ""
        )

        valid_exams.append(
            {
                "question_pdf": question_pdf,
                "answer_pdf": answer_pdf,
                "exam_id": exam_id,
            }
        )

    if not valid_exams:
        raise ValueError("No valid exams found. Check your PDF paths.")

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output.dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = None
    if cfg.options.save_images:
        image_dir = Path(hydra.utils.to_absolute_path(cfg.options.image_dir))

    print("")
    print("=" * 50)
    print(f"[{cfg.exam.display_name}] Batch Extraction")
    print("=" * 50)
    print(f"Total exams: {len(valid_exams)}")
    print(f"Output directory: {output_dir}")

    api_client = get_api_client(cfg.api)
    print(f"API: {cfg.api.name} ({cfg.api.model})")

    all_dfs = []
    success_count = 0
    fail_count = 0

    for i, exam in enumerate(valid_exams, 1):
        print("")
        print(f"[{i}/{len(valid_exams)}]", end="")

        try:
            df = process_single_exam(
                cfg=cfg,
                api_client=api_client,
                question_pdf=exam["question_pdf"],
                answer_pdf=exam["answer_pdf"],
                exam_id=exam["exam_id"],
                output_dir=output_dir,
                image_dir=image_dir,
            )
            all_dfs.append(df)
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] {exam['exam_id']}: {e}")
            fail_count += 1

    if cfg.output.merge and all_dfs:
        print("")
        print("-" * 40)
        print("Merging all datasets...")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_path = output_dir / cfg.output.merge_filename
        save_dataset(merged_df, merged_path)

    print("")
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    if cfg.output.merge and all_dfs:
        total_questions = sum(len(df) for df in all_dfs)
        print(f"  Total questions: {total_questions}")
        print(f"  Merged file: {merged_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
