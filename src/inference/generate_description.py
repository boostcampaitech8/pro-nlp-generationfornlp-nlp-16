import torch
from typing import Dict, List, Optional
from .description_prompt import get_prompt_template, SYSTEM_PROMPT_DESCRIPTION
from tqdm import tqdm

def generate_description_single(
    model,
    tokenizer,
    paragraph: str,
    question: str,
    choices: List[str],
    question_plus: Optional[str] = None,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    단일 테스트 샘플(문제 1개)에 대해 설명(description, hint)을 생성.

    이 함수는 LLM 추론의 최소 단위로,
    하나의 문제를 입력 받아 하나의 설명 문자열을 반환.

    Args:
        model:
            로드된 SKT A.X 계열 모델
            (PEFT/LoRA 어댑터가 적용된 모델일 수도 있음)
        tokenizer:
            모델에 대응되는 tokenizer
        paragraph:
            문제의 지문(독해 텍스트)
        question:
            질문 문장
        choices:
            객관식 선택지 리스트
        prompt_template:
            프롬프트 템플릿 이름
            (예: "basic", "structured")
        max_new_tokens:
            생성할 최대 토큰 수
        temperature:
            샘플링 온도 (값이 클수록 다양성 증가)
        top_p:
            nucleus sampling 파라미터
        top_k:
            top-k sampling 파라미터
        repetition_penalty:
            반복 생성 방지를 위한 패널티
        do_sample:
            샘플링 사용 여부
            (False면 greedy decoding)

    Returns:
        문제 해결에 도움을 주는 설명 문자열 (정답은 직접 포함하지 않음)
    """

    # 1. 프롬프트 템플릿

    user_prompt = get_prompt_template(
        paragraph=paragraph,
        question=question,
        choices=choices,
        question_plus=question_plus,
    )

    # 2. Chat 메시지 구성
    # system: 모델의 역할 및 행동 규칙 정의
    # user: 실제 문제 정보
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DESCRIPTION},
        {"role": "user", "content": user_prompt},
    ]

    # 3. Chat template 적용
    # tokenizer가 모델별 chat format을 자동 적용
    # add_generation_prompt=True:
    #   assistant role 토큰까지 포함하여
    #   generation 시작 위치를 명확히 지정
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # 4. 텍스트 생성 (추론)
    # gradient 계산 비활성화 (inference 전용)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 5. 출력 디코딩
    # output은 [프롬프트 + 생성 토큰] 형태이므로
    # 입력 길이 이후 토큰만 잘라서 디코딩
    len_input = len(input_ids[0])
    description = tokenizer.decode(
        output[0][len_input:],
        skip_special_tokens=True,
    )

    # 앞뒤 공백 제거
    description = description.strip()

    return description

def generate_descriptions_batch(
    model,
    tokenizer,
    test_dataset: List[Dict],
    max_new_tokens: int = 500,
    temperatures: List[float] = [0.7],
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    show_progress: bool = True,
) -> List[Dict]:
    """
    여러 개의 테스트 샘플에 대해 설명(description)을 순차적으로 생성.

    내부적으로는 generate_description_single을 반복 호출하는 방식.

    Args:
        model:
            로드된 모델
        tokenizer:
            tokenizer
        test_dataset:
            테스트 샘플 리스트
            각 샘플은 다음 키를 포함해야 함:
            - id
            - paragraph
            - question
            - choices
        prompt_template:
            사용할 프롬프트 템플릿
        max_new_tokens:
            생성할 최대 토큰 수
        temperatures:
            샘플링 온도 리스트 (각 temperature마다 description 생성)
        top_p:
            nucleus sampling 파라미터
        top_k:
            top-k sampling 파라미터
        repetition_penalty:
            반복 패널티
        do_sample:
            샘플링 사용 여부
        show_progress:
            tqdm progress bar 표시 여부

    Returns:
        다음 형태의 딕셔너리 리스트:
        [
            {"id": sample_id, "description_1": 설명1, "description_2": 설명2, ...},
            ...
        ]
        (temperatures 개수만큼 description_N 컬럼 생성)
    """

    # 추론 모드로 전환 (dropout 등 비활성화)
    model.eval()

    results = []

    # 진행 상황 표시 여부에 따라 iterator 선택
    iterator = (
        tqdm(test_dataset, desc="Generating descriptions")
        if show_progress
        else test_dataset
    )

    # 각 샘플에 대해 순차적으로 description 생성
    for sample in iterator:
        sample_id = sample["id"]
        paragraph = sample["paragraph"]
        question = sample["question"]
        question_plus = sample["question_plus"]
        choices = sample["choices"]

        # 여러 temperature로 description 생성
        descriptions = []
        for temp in temperatures:
            description = generate_description_single(
                model=model,
                tokenizer=tokenizer,
                paragraph=paragraph,
                question=question,
                question_plus=question_plus,
                choices=choices,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            descriptions.append(description)

        # 결과 저장 (description_1, description_2, ... 컬럼으로)
        result = {"id": sample_id}
        for i, desc in enumerate(descriptions, 1):
            result[f"description_{i}"] = desc

        results.append(result)

    return results


def prepare_test_sample(
    sample_id: str,
    paragraph: str,
    question: str,
    choices: List[str],
    question_plus: Optional[str],
) -> Dict:
    """
    단일 테스트 샘플을 description 생성용 표준 포맷으로 변환한다.

    Args:
        sample_id:
            샘플 고유 ID
        paragraph:
            지문
        question:
            질문 문장
        choices:
            선택지 리스트

    Returns:
        테스트 샘플 딕셔너리
    """

    return {
        "id": sample_id,
        "paragraph": paragraph,
        "question": question,
        "question_plus": question_plus,
        "choices": choices,
    }