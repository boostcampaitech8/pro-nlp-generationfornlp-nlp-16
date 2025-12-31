import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


def run_inference(
    model, tokenizer, test_dataset: list, enable_thinking: bool = False
) -> list:
    """
    Run inference on test dataset
    """
    infer_results = []

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            template_params = {
                "tokenizer": True,
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
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
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


def save_predictions(infer_results: list, output_path: str = "output.csv"):
    """
    Save predictions to CSV file
    """
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
