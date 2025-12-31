"""
Korean SAT Solver - Inference Script
"""

from src.data import load_test_data, process_test_dataset
from src.model import load_model_for_inference
from src.inference import run_inference, save_predictions
import hydra
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # just logging config
    # print(OmegaConf.to_yaml(cfg))

    checkpoint_path = cfg.inference.checkpoint_dir  # 사용자가 지정한 체크포인트 폴더
    checkpoint_step = (
        cfg.inference.checkpoint_step
    )  # "best" 또는 특정 스텝 숫자 (예: 1000)

    # huggingface 모델 바로 사용하는 경우
    use_huggingface = cfg.inference.get("use_huggingface", False)
    test_data_path = hydra.utils.to_absolute_path(
        cfg.data.test_path
    )  # 테스트 데이터 경로
    output_path = cfg.inference.output_file  # 추론 결과 저장할 파일명

    if use_huggingface:
        print(f"Using HuggingFace model directly: {checkpoint_path}")
    else:
        if not checkpoint_path:
            print(
                "Checkpoint path is empty. Searching for the latest checkpoint in 'outputs/'..."
            )

        # [중요] Hydra가 작업 경로를 바꿨으므로, 원래 실행했던 위치(프로젝트 루트)를 알아내야 함
        original_cwd = hydra.utils.get_original_cwd()
        outputs_roots = [
            os.path.join(original_cwd, "outputs"),  # 기본 경로
            os.path.join(
                original_cwd, "outputs", "train"
            ),  # 학습 스크립트가 하위 train 폴더를 쓰는 경우 대응
        ]

        found_path = None
        for outputs_root in outputs_roots:
            if not os.path.exists(outputs_root):
                continue

            # 1차 분류: 날짜별 폴더 (예: 2024-05-20) -> 최신 날짜부터 정렬(reverse=True)
            dates = sorted(
                [
                    d
                    for d in os.listdir(outputs_root)
                    if os.path.isdir(os.path.join(outputs_root, d))
                ],
                reverse=True,
            )

            for date in dates:
                date_dir = os.path.join(outputs_root, date)
                # 2차 분류: 시간별 폴더 (예: 14-30-00) -> 최신 시간부터 정렬
                times = sorted(
                    [
                        t
                        for t in os.listdir(date_dir)
                        if os.path.isdir(os.path.join(date_dir, t))
                    ],
                    reverse=True,
                )

                for time in times:
                    run_dir = os.path.join(date_dir, time)

                    # [핵심 검증] 그냥 폴더만 있다고 되는 게 아니라,
                    # 그 안에 진짜 'checkpoint-'로 시작하는 모델 파일이 있는지 확인
                    # (방금 막 실행해서 비어있는 폴더나, 에러나서 꺼진 폴더는 무시하기 위함)
                    if any(d.startswith("checkpoint-") for d in os.listdir(run_dir)):
                        found_path = run_dir  # 찾았다!
                        break
                if found_path:
                    break
            if found_path:
                break

        # 찾은 경로 적용
        if found_path:
            checkpoint_path = found_path
            print(f"Auto-detected latest run with checkpoint: {checkpoint_path}")
        else:
            # 다 뒤졌는데 없으면 에러 발생
            raise ValueError(
                "Checkpoint path is empty and no previous checkpoints were found..."
            )

        if not checkpoint_path:
            raise ValueError(
                "Checkpoint path must be specified (inference.checkpoint_dir)"
            )

        # 경로가 절대 경로가 아니면 절대 경로로 변환 (안전장치)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

        # 사용자가 지정한 경로가 구체적인 파일(checkpoint-1000)이 아니라, 상위 폴더일 때
        if "checkpoint-" not in os.path.basename(checkpoint_path) and os.path.isdir(
            checkpoint_path
        ):

            # 1. 가장 많이 학습된(best) 거 가져오기
            if checkpoint_step == "best":
                # 폴더 내의 모든 'checkpoint-xxx' 목록을 가져옴
                checkpoints = [
                    d
                    for d in os.listdir(checkpoint_path)
                    if d.startswith("checkpoint-")
                ]
                if checkpoints:
                    # 숫자 기준 정렬 (문자열 정렬하면 1000이 500보다 앞에 올 수 있으니 int 변환 필수)
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    target_checkpoint = checkpoints[-1]  # 제일 큰 숫자(마지막) 선택
                    checkpoint_path = os.path.join(checkpoint_path, target_checkpoint)

            # 2. 특정 스텝(예: 500)을 가져오기
            else:
                target_checkpoint = f"checkpoint-{checkpoint_step}"
                possible_path = os.path.join(checkpoint_path, target_checkpoint)
                if os.path.exists(possible_path):
                    checkpoint_path = possible_path

    print(f"Checkpoint Path: {checkpoint_path}")  # 최종 결정된 모델 경로
    print(f"Test Data Path: {test_data_path}")  # 테스트 데이터 경로
    print(f"Output Path: {output_path}")  # 결과 저장할 파일명

    # Load model
    print("Loading model from checkpoint...")
    torch_dtype = cfg.inference.torch_dtype
    use_peft = cfg.inference.get("use_peft", True)
    model, tokenizer = load_model_for_inference(
        checkpoint_path, torch_dtype=torch_dtype, use_peft=use_peft
    )

    # Load and preprocess test data
    print("Loading test data...")
    test_df = load_test_data(test_data_path)
    print(f"Test dataset size: {len(test_df)}")

    # Process test dataset
    print("Processing test dataset...")
    test_dataset = process_test_dataset(test_df)

    # Run inference
    print("Running inference...")
    enable_thinking = cfg.inference.get("enable_thinking", False)
    infer_results = run_inference(model, tokenizer, test_dataset, enable_thinking)

    # Save results
    print(f"Saving predictions to {output_path}...")
    save_predictions(infer_results, output_path)
    print(f"Inference completed! Results saved to {output_path} (in {os.getcwd()})")


if __name__ == "__main__":
    main()
