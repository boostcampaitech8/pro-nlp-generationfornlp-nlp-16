SYSTEM_PROMPT_DESCRIPTION = """당신은 국어 문제 해결을 돕는 분석 보조자입니다.
다음 모델이 문제를 풀 때 참고할 수 있는 **관점과 핵심 정보**를 제공하세요.

**역할:**
- 지문과 선택지를 연결하는 핵심 정보 제시
- 각 선택지 검토 시 고려해야 할 관점 제공
- 판단 근거가 될 정보만 제공
- 정답 언급 금지!

**출력 제한:**
- 100자 이내로 간결하게 작성
- 지문에 명시된 내용만 사용
- "정답은 X번", "X번이 맞다" 같은 직접적 답 제시 금지
- 불필요한 반복이나 장황한 설명 금지

**출력 형식:**
선택지와 관련된 지문의 핵심 정보를 간결하게 제시
"""

def format_choices(choices: list) -> str:
    """
    객관식 선택지 리스트를 사람이 읽기 쉬운 형태로 포맷한다.

    예:
        입력: ["보기1", "보기2"]
        출력:
            1. 보기1
            2. 보기2

    Args:
        choices:
            선택지 문자열 리스트

    Returns:
        번호가 매겨진 선택지 문자열
    """

    formatted = []

    # 선택지에 1부터 번호를 매겨 문자열로 변환
    for idx, choice in enumerate(choices, 1):
        formatted.append(f"{idx}. {choice}")

    # 줄바꿈으로 연결하여 하나의 문자열로 반환
    return "\n".join(formatted)

def format_description_prompt_no_plus(
    paragraph: str,
    question: str,
    choices: list,
) -> str:
    """
    <보기>가 없는 일반 문제를 위한 분석 정보 생성 프롬프트

    목적:
    - 다음 단계의 대형 추론 모델이 문제를 풀 수 있도록
      핵심 개념, 판단 기준, 사고 방향을 제공한다.
    - 정답이나 선택지 번호는 직접 언급하지 않는다.

    Args:
        paragraph:
            문제의 지문(독해 텍스트)
        question:
            질문 문장
        choices:
            객관식 선택지 리스트

    Returns:
        description 생성을 위한 user prompt 문자열
    """

    formatted_choices = format_choices(choices)

    prompt = f"""[지문]
{paragraph}

[선택지]
{formatted_choices}

**작업:**
선택지를 판단하는 데 필요한 지문 속 핵심 정보를 간결하게 제시하세요.
- 각 선택지와 관련된 지문 내용을 1-2문장으로 요약
- 직접 답을 말하지 말고 판단 근거가 될 정보만 제공
- 총 100자 이내로 간결하게
- 선택지 번호는 언급 금지

**출력:**
"""

    return prompt

def format_description_prompt_with_plus(
    paragraph: str,
    question: str,
    question_plus: str,
    choices: list,
) -> str:
    """
    <보기>가 포함된 문제를 위한 정보 추출 프롬프트

    목적:
    - <보기>를 문제 해결의 조건 블록으로 분리하여 제시
    - 지문, <보기>, 선택지에 명시된 정보만 추출
    - 다음 단계의 대형 추론 모델이 판단에 사용할 수 있는
      정제된 사실 목록을 생성

    주의:
    - 해석, 판단, 정답 추론은 절대 수행하지 않는다.
    """

    formatted_choices = format_choices(choices)

    prompt = f"""[지문]
{paragraph}

[보기]
{question_plus}

[선택지]
{formatted_choices}

**작업:**
선택지를 판단하는 데 필요한 지문 속 핵심 정보를 간결하게 제시하세요.
- 각 선택지와 관련된 지문 내용을 1-2문장으로 요약
- 직접 답을 말하지 말고 판단 근거가 될 정보만 제공
- 총 100자 이내로 간결하게
- 선택지 번호는 언급 금
지
**출력:**
"""

    return prompt


def get_prompt_template(
    paragraph: str,
    question: str,
    choices: list,
    question_plus: str | None = None,
) -> str:
    """
    <보기>(question_plus) 유무에 따라
    적절한 정보 추출 프롬프트를 반환한다.
    """

    if question_plus is None:
        return format_description_prompt_no_plus(
            paragraph=paragraph,
            question=question,
            choices=choices,
        )
    else:
        return format_description_prompt_with_plus(
            paragraph=paragraph,
            question=question,
            question_plus=question_plus,
            choices=choices,
        )

