SYSTEM_PROMPT_DESCRIPTION = """당신은 국어 문제 해결을 위한 정보 정리기입니다.
해석, 판단, 정답 추론을 하지 말고,
지문에서 선택지와 직접적으로 관련된 내용만 발췌·정리하세요.
다음 모델이 이 정리된 정보를 바탕으로 판단을 수행합니다.

출력 제한:
- 지문을 문제에 맞게 요약하는 것입니다.
- 지문에 명시적으로 드러난 내용만 사용하세요.
- 선택지와 무관한 지문 내용은 포함하지 마세요.
- 설명, 평가, 추론, 정답은 절대 포함하지 마세요.
- 지문과 선택지의 단어가 다를 경우, 의미 판단 없이 형태만 병기하세요.
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

    prompt = f"""당신은 문제 해결을 위한 '정보 정리기'입니다.
해석이나 판단을 하지 말고, 지문에서 각 선택지와 직접 관련 있는 부분만 정리하세요.

[지문]
{paragraph}

[선택지]
{formatted_choices}

작업 지침:
- 선택지와 관련 없는 지문 내용은 제외하세요.
- 지문에 실제로 등장하는 표현만 사용하세요.
- 요약은 사실 나열 수준으로만 작성하세요.

출력 형식:

선택지 관련 지문 발췌/요약
- 선택지의 핵심 표현과 직접 대응되는 지문 문장 또는 요지 정리
- 선택지별로 구분하여 작성 (선택지 번호는 쓰지 말 것)

[정리 결과]
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

    prompt = f"""당신은 문제 해결을 위한 '정보 정리기'입니다.
해석이나 판단을 하지 말고, 지문에서 각 선택지와 직접 관련 있는 부분만 정리하세요.

[지문]
{paragraph}

[보기]
{question_plus}

[선택지]
{formatted_choices}

작업 지침:
- 선택지와 관련 없는 지문 내용은 제외하세요.
- 지문에 실제로 등장하는 표현만 사용하세요.
- 요약은 사실 나열 수준으로만 작성하세요.

출력 형식:

선택지 관련 지문 발췌/요약
- 선택지의 핵심 표현과 직접 대응되는 지문 문장 또는 요지 정리
- 선택지별로 구분하여 작성 (선택지 번호는 쓰지 말 것)

[정리 결과]
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

