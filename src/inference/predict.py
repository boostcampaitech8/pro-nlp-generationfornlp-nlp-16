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
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]

            template_params = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }

            if enable_thinking:
                try:
                    # thinking 사용
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=True,
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

            # 실제 텍스트 생성
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # 생성된 텍스트 디코딩 (입력 부분 제외)
            generated_text = tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True
            )

            # 생성된 텍스트에서 첫 번째 숫자(1~5) 추출
            predict_value = None
            for char in generated_text.strip():
                if char in ["1", "2", "3", "4", "5"]:
                    predict_value = char
                    break

            # 숫자를 찾지 못한 경우 기본값 "1"
            if predict_value is None:
                print(f"Warning: No valid answer found for ID {_id}. Generated: '{generated_text}'. Using default '1'")
                predict_value = "1"

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
    model, tokenizer, test_dataset: list, enable_thinking: bool = False, inference_mode: str = "logit"
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
        raise ValueError(f"Unknown inference_mode: {inference_mode}. Must be 'generate' or 'logit'")



def save_predictions(infer_results: list, output_path: str = "output.csv"):
    """
    Save predictions to CSV file
    """
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
