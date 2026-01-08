import os
import ast
import re
import torch
import pandas as pd
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

CHECKPOINT_PATH = "/data/ephemeral/home/han/han/outputs/train/2026-01-04/15-30-42/checkpoint-670"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "output_cot.csv"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 400

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def extract_answer(text):

    text = text.strip()

    match = re.search(r'정답은?\s*(\d)\s*번', text)
    if match:
        return match.group(1)

    match = re.search(r'(\d)\s*번이다', text)
    if match:
        return match.group(1)

    match = re.search(r'따라서\s*(\d)', text)
    if match:
        return match.group(1)

    matches = re.findall(r'[1-5]', text)
    if matches:
        return matches[-1]
    
    return "1"


def prepare_prompts(df, tokenizer):
    prompts = []
    
    for idx, row in df.iterrows():
        problems = ast.literal_eval(row['problems'])
        question = problems['question']
        choices = "\n".join([f"{i+1} - {c}" for i, c in enumerate(problems['choices'])])
        
        # question_plus 처리 (row 레벨에 있음)
        question_plus = row.get('question_plus', '')
        if pd.isna(question_plus):
            question_plus = ""
        
        # 학습 때와 동일한 프롬프트 사용
        if question_plus:
            user_content = PROMPT_QUESTION_PLUS.format(
                paragraph=row['paragraph'],
                question=question,
                question_plus=question_plus,
                choices=choices,
            )
        else:
            user_content = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row['paragraph'],
                question=question,
                choices=choices,
            )

        messages = [
            {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
            {"role": "user", "content": user_content}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append({"id": row['id'], "prompt": prompt})
    
    return prompts


def batch_inference(model, tokenizer, prompts, batch_size, max_new_tokens):
    results = []
    reasoning_data = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch Inference"):
        batch = prompts[i:i+batch_size]
        batch_prompts = [p['prompt'] for p in batch]
        batch_ids = [p['id'] for p in batch]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
        
        # 디코딩
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            answer = extract_answer(generated)
            
            results.append({"id": batch_ids[j], "answer": answer})
            reasoning_data.append({
                "id": batch_ids[j], 
                "answer": answer, 
                "reasoning": generated
            })
            
            if len(results) <= 3:
                print(f"\n{'='*50}")
                print(f"[샘플 {len(results)}]")
                print(f"생성: {generated[:200]}...")
                print(f"추출된 정답: {answer}")
                print(f"{'='*50}")
    
    return results, reasoning_data


def main():
    print(f"Batch Inference (batch_size={BATCH_SIZE}, max_tokens={MAX_NEW_TOKENS})")

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded")

    print("Preparing data...")
    df = pd.read_csv(TEST_PATH)
    prompts = prepare_prompts(df, tokenizer)
    print(f"{len(prompts)}개 프롬프트 준비 완료")

    print("Running Inference...")
    results, reasoning_data = batch_inference(
        model, tokenizer, prompts, BATCH_SIZE, MAX_NEW_TOKENS
    )

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(reasoning_data).to_csv("output_cot_reasoning.csv", index=False)
    print(f"\n완료!")
    print(f"   제출용: {OUTPUT_PATH}")
    print(f"   풀이포함: output_cot_reasoning.csv")


if __name__ == "__main__":
    main()
