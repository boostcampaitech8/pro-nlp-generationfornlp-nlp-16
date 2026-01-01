import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


def run_inference_generate(
    model, tokenizer, test_dataset: list, enable_thinking: bool = False
) -> list:
    """
    Run inference using generate() method (for generative models like EXAONE 32B)
    """
    infer_results = []

    model.eval()
    with torch.inference_mode():
        for idx, data in enumerate(tqdm(test_dataset)):
            _id = data["id"]
            messages = data["messages"]

            print(f"\n[DEBUG {idx}] Starting inference for ID: {_id}")

            template_params = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }

            print(f"[DEBUG {idx}] Applying chat template...")
            if enable_thinking:
                try:
                    # thinking 사용
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=False,
                        **template_params,
                    ).to("cuda")
                except Exception as e:
                    print(
                        f"Thinking 모드에서 오류 발생하여 일반 모드로 전환합니다. (ID: {_id})\n오류: {e}"
                    )
                    # thinking 사용 불가 시 일반 모드
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=False,
                        **template_params,
                    ).to("cuda")
            else:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    **template_params,
                ).to("cuda")

            print(f"[DEBUG {idx}] Input shape: {inputs.shape}")
            print(f"[DEBUG {idx}] EOS token ID: {tokenizer.eos_token_id}")
            print(f"[DEBUG {idx}] Starting text generation...")

            # 실제 텍스트 생성
            import time

            start_time = time.time()
            outputs = model.generate(
                inputs,
                max_new_tokens=100,  # 50 -> 100으로 조금 증가
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.time() - start_time
            print(f"[DEBUG {idx}] Generation took {elapsed:.2f}s")
            print(f"[DEBUG {idx}] Generated {outputs.shape[1] - inputs.shape[1]} new tokens")

            print(f"[DEBUG {idx}] Generation completed. Output shape: {outputs.shape}")

            # 생성된 텍스트 디코딩 (입력 부분 제외)
            generated_text = tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            )

            print(
                f"[DEBUG {idx}] Generated text (first 100 chars): {generated_text[:100]}"
            )

            # 생성된 텍스트에서 마지막 숫자(1~5) 추출
            # 역순으로 검색하여 가장 마지막에 나온 숫자를 정답으로 사용
            predict_value = None
            for char in reversed(generated_text.strip()):
                if char in ["1", "2", "3", "4", "5"]:
                    predict_value = char
                    break

            # 숫자를 찾지 못한 경우 기본값 "1"
            if predict_value is None:
                print(
                    f"Warning: No valid answer found for ID {_id}. Generated: '{generated_text}'. Using default '1'"
                )
                predict_value = "1"

            print(f"[DEBUG {idx}] Final answer: {predict_value}")
            infer_results.append({"id": _id, "answer": predict_value})

    return infer_results


def run_inference_logit(
    model, tokenizer, test_dataset: list, enable_thinking: bool = False
) -> list:
    """
    Run inference using logits (for classification-style models or fine-tuned models)
    """
    infer_results = []

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            template_params = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }

            if enable_thinking:
                try:
                    # thinking 사용
                    outputs = model(
                        tokenizer.apply_chat_template(
                            messages,
                            enable_thinking=True,
                            **template_params,
                        ).to("cuda")
                    )
                except Exception as e:
                    print(
                        f"Thinking 모드에서 오류 발생하여 일반 모드로 전환합니다. (ID: {_id})\n오류: {e}"
                    )
                    # thinking 사용 불가 시 일반 모드
                    outputs = model(
                        tokenizer.apply_chat_template(
                            messages,
                            enable_thinking=False,
                            **template_params,
                        ).to("cuda")
                    )
            else:
                outputs = model(
                    tokenizer.apply_chat_template(
                        messages,
                        **template_params,
                    ).to("cuda")
                )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [
                logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)
            ]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(target_logit_list, dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    return infer_results


def run_inference(
    model,
    tokenizer,
    test_dataset: list,
    enable_thinking: bool = False,
    inference_mode: str = "logit",
) -> list:
    """
    Run inference with the specified mode

    Args:
        model: The model to use for inference
        tokenizer: The tokenizer
        test_dataset: Test dataset
        enable_thinking: Whether to enable thinking mode
        inference_mode: "generate" for generative models, "logit" for classification models

    Returns:
        List of inference results
    """
    if inference_mode == "generate":
        return run_inference_generate(model, tokenizer, test_dataset, enable_thinking)
    elif inference_mode == "logit":
        return run_inference_logit(model, tokenizer, test_dataset, enable_thinking)
    else:
        raise ValueError(
            f"Unknown inference_mode: {inference_mode}. Must be 'generate' or 'logit'"
        )


def save_predictions(infer_results: list, output_path: str = "output.csv"):
    """
    Save predictions to CSV file
    """
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
