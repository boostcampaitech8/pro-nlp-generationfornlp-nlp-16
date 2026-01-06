import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_inference(model, tokenizer, test_dataset: list) -> list:
    """
    Run inference on test dataset
    """
    infer_results = []

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("cuda")

            outputs = model.generate(
                inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            
            # 숫자만 추출
            if response in ["1", "2", "3", "4", "5"]:
                predict_value = response
            else:
                # fallback: 첫 글자가 숫자면 사용
                predict_value = response[0] if response and response[0].isdigit() else "1"

            infer_results.append({"id": _id, "answer": predict_value})

    return infer_results


def save_predictions(infer_results: list, output_path: str = "output.csv"):
    """
    Save predictions to CSV file
    """
    pd.DataFrame(infer_results).to_csv(output_path, index=False)