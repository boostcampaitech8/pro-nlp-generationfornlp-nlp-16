import os
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from src.model.model import load_model_for_inference
from src.data.dataset import load_test_data
from src.inference.generate_description import (
    generate_descriptions_batch,
    prepare_test_sample,
)


def save_descriptions(results: list, output_path: str = "descriptions.csv"):
    """
    생성된 description 결과를 CSV 파일로 저장한다.

    Args:
        results:
            [{"id": ..., "description": ...}, ...] 형태의 리스트
        output_path:
            CSV 파일 저장 경로
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"총 {len(results)}개의 description을 {output_path}에 저장했습니다.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    description 생성 inference 전체 파이프라인을 실행하는 메인 함수
    """

    print("=" * 60)
    print("Description Generation Inference")
    print("=" * 60)

    # 1. 설정 및 경로 준비
    checkpoint_path = cfg.inference.checkpoint_dir
    checkpoint_step = cfg.inference.checkpoint_step
    test_data_path = hydra.utils.to_absolute_path(cfg.data.test_path)
    output_path = cfg.inference.get("description_output_file", "descriptions.csv")

    original_cwd = hydra.utils.get_original_cwd()

    # 2. 체크포인트 자동 탐색
    if not checkpoint_path:
        print("Checkpoint 경로가 지정되지 않아 자동 탐색을 수행합니다.")

        outputs_roots = [
            os.path.join(original_cwd, "outputs", "dapt"),
            os.path.join(original_cwd, "outputs", "train"),
            os.path.join(original_cwd, "outputs"),
        ]

        found_path = None
        for root in outputs_roots:
            if not os.path.exists(root):
                continue

            for date in sorted(os.listdir(root), reverse=True):
                date_dir = os.path.join(root, date)
                if not os.path.isdir(date_dir):
                    continue

                for time in sorted(os.listdir(date_dir), reverse=True):
                    run_dir = os.path.join(date_dir, time)
                    if not os.path.isdir(run_dir):
                        continue

                    if os.path.exists(os.path.join(run_dir, "final_model")):
                        found_path = os.path.join(run_dir, "final_model")
                        break
                    elif any(d.startswith("checkpoint-") for d in os.listdir(run_dir)):
                        found_path = run_dir
                        break

                if found_path:
                    break
            if found_path:
                break

        if not found_path:
            raise ValueError("사용 가능한 checkpoint를 찾지 못했습니다.")

        checkpoint_path = found_path
        print(f"자동 탐색된 checkpoint: {checkpoint_path}")

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

    # 3. 특정 checkpoint 선택
    if (
        "final_model" not in checkpoint_path
        and "checkpoint-" not in os.path.basename(checkpoint_path)
        and os.path.isdir(checkpoint_path)
    ):
        if checkpoint_step == "best":
            checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])
        else:
            candidate = os.path.join(checkpoint_path, f"checkpoint-{checkpoint_step}")
            if os.path.exists(candidate):
                checkpoint_path = candidate

    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Test Data Path: {test_data_path}")
    print(f"Output Path: {output_path}")

    # 4. 모델 로드
    print("\n모델을 로드합니다...")
    torch_dtype = cfg.inference.get("torch_dtype", "bfloat16")
    model, tokenizer = load_model_for_inference(
        checkpoint_path,
        torch_dtype=torch_dtype,
    )
    print(f"모델 로드 완료 (dtype={torch_dtype})")

    # 5. 테스트 데이터 로드
    print("\n테스트 데이터를 로드합니다...")
    test_df = load_test_data(test_data_path)
    
    # 실사용시 주석처리!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    # test_df = test_df.head(5)      # 디버그용
    print(f"테스트 샘플 수: {len(test_df)}")

    # 6. description 생성용 샘플 준비
    test_samples = []
    for _, row in test_df.iterrows():
        sample = prepare_test_sample(
            sample_id=row["id"],
            paragraph=row["paragraph"],
            question=row["question"],
            question_plus=row.get("question_plus", None),
            choices=row["choices"],
        )

        test_samples.append(sample)

    print(f"총 {len(test_samples)}개의 샘플을 준비했습니다.")

    # 7. generation 설정
    gen_cfg = cfg.inference.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 384)
    temperature = gen_cfg.get("temperature", 0.7)
    top_p = gen_cfg.get("top_p", 0.9)
    top_k = gen_cfg.get("top_k", 50)
    repetition_penalty = gen_cfg.get("repetition_penalty", 1.1)
    do_sample = gen_cfg.get("do_sample", True)

    print("\nGeneration 설정:")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  repetition_penalty: {repetition_penalty}")
    print(f"  do_sample: {do_sample}")

    # 8. description 생성
    print("\ndescription 생성을 시작합니다...")
    results = generate_descriptions_batch(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        show_progress=True,
    )

    # 9. 결과 저장
    print("\n결과를 저장합니다...")
    save_descriptions(results, output_path)

    # 10. 샘플 출력
    print("\n샘플 결과 (상위 3개):")
    for i in range(min(3, len(results))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"ID: {results[i]['id']}")
        print(results[i]["description"])

    print("\nDescription generation이 완료되었습니다.")


if __name__ == "__main__":
    main()
