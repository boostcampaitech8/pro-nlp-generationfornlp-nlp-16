import re
import torch
import pandas as pd
from tqdm import tqdm

pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

COT_PROMPT_TEMPALATE = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

위 문제를 단계별로 분석하세요.
1. 지문에서 관련 정보를 찾으세요.
2. 각 선택지를 검토하세요.
3. 마지막에 "따라서 정답은 N번이다."로 끝내세요.

분석: """

def extract_answer(text: str) -> str:
    match = re.search(r'정답[은는이가]?\s*(\d)\s*번?', text)
    if match :
        return match.group(1)

    match = re.search(r'(\d)번이다', text)
    if match :
        return match.group(1)

    numbers = re.findall(r'[1-5]', text)
    if numbers :
        return numbers[-1]

    return "1"

def generate_cot_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> str:

    messages = [{"role": "user", "content" : prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)


    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return generated

def run_inference_cot(
    model,
    tokenizer,
    test_dataset: list,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
    verbose: bool = True,
) -> list:

    infer_results = []
    reasoning_results = []

    model.eval()

    for idx, data in enumerate(tqdm(test_dataset, desc= "CoT Inference")):
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]

        if isinstance(choices, list):
            choices_str = "\n".join([f"{i+1} - {c}" for i, c in enumerate(choices)])
        else:
            choices_str = choices

        prompt = COT_PROMPT_TEMPALATE.format(
            paragraph=paragraph,
            question=question,
            choices=choices_str,
        )    
        
        generated = generate_cot_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        answer = extract_answer(generated)

        infer_results.append({
            "id": _id,
            "answer": answer
        })

        reasoning_results.append({
            "id": _id,
            "answer": answer,
            "reasoning": generated
        })

        #첫 3개 샘플 출력(확인용)
        if verbose and idx < 3:
            print(f"\n{'='*50}")
            print(f"[샘플 {idx+1}]")
            print(f"생성된 풀이:\n{generated[:300]}...")
            print(f"추출된 정답: {answer}")
            print(f"{'='*50}")

    return infer_results, reasoning_results

def save_predictions_cot(
    infer_results:list,
    reasoning_results: list,
    output_path: str = "output_cot.csv",
):
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print(f"제출용 저장: {output_path}")

    reasoning_path = output_path.replace(".csv", "_reasoning.csv")
    pd.DataFrame(reasoning_results).to_csv(reasoning_path, index=False)
    print(f"풀이 포함 저장: {reasoning_path}")

