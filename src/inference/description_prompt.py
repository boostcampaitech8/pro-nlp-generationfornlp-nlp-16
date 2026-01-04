SYSTEM_PROMPT_DESCRIPTION = """당신은 국어 문제 해결을 위한 정보 추출기입니다.
해석, 판단, 정답 추론을 하지 말고,
지문, <보기>, 선택지에 명시적으로 드러난 정보만 구조적으로 정리하세요.
다음 모델이 이 정보를 바탕으로 판단을 수행합니다.

출력 제한:
- 각 항목은 한 줄 또는 한 문장으로만 작성하세요.
- 조건/관계 문장은 최대 3개까지만 작성하세요.
- 단어 쌍, 근거 항목 등 나열형 정보는 각 항목당 최대 5개까지만 작성하세요.
- 설명을 덧붙이지 말고, 사실 진술만 작성하세요.
- 지문과 선택지의 단어가 다를 경우, 의미가 같은지 다른지 판단하지 말고 '단어 형태'만 나열하세요.
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

    prompt = f"""당신은 문제 해결을 위한 '정보 추출기'입니다.
해석이나 판단을 하지 말고, 지문과 선택지에 명시적으로 나타난 정보만 정리하세요.
다음 모델이 이 정보를 바탕으로 판단을 수행합니다.

[지문]
{paragraph}

[문제]
{question}

[선택지]
{formatted_choices}

작업 지침:
- 정답, 옳고 그름, 추론 결과는 절대 작성하지 마세요.
- 지문에 실제로 등장하는 표현만 사용하세요.
- 추상적인 요약이나 해설 문장은 쓰지 마세요.

출력 형식:

1. 지문 내 조건/관계 문장
- 원인–결과, 조건–결론 형태의 문장을 그대로 발췌하거나 짧게 재작성
- 예: “A일 경우 B이다”, “X는 Y를 전제로 한다”

2. 선택지별 근거 후보
- 각 선택지 문장에서 핵심 명사/동사를 추출
- 해당 표현이 지문 어디(문단/문장)에 등장하는지 표시
- 지문에 없는 경우: “지문 내 직접 대응 표현 없음”

3. 단어 형태 차이
- 지문과 선택지에서 형태가 비슷하지만 다른 단어 쌍이 있는지 나열
- 예: ‘증가’ vs ‘급증’, ‘일부’ vs ‘전체’

4. 판별에 필요한 최소 질문
- 이 문제를 판단하기 위해 반드시 확인해야 하는 사실 질문 1개
- 예: “지문에서 X가 Y의 원인으로 명시되었는가?”
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

    prompt = f"""당신은 문제 해결을 위한 '정보 추출기'입니다.
해석이나 판단을 하지 말고, 지문·<보기>·선택지에 명시적으로 나타난 정보만 정리하세요.
다음 모델이 이 정보를 바탕으로 판단을 수행합니다.

[지문]
{paragraph}

[보기]
{question_plus}

[문제]
{question}

[선택지]
{formatted_choices}

작업 지침:
- 정답, 옳고 그름, 추론 결과는 절대 작성하지 마세요.
- 지문과 <보기>에 실제로 등장하는 표현만 사용하세요.
- 추상적인 요약이나 해설 문장은 쓰지 마세요.

출력 형식:

1. 지문 내 조건/관계 문장
- 지문에서 명시된 원인–결과, 조건–결론 문장을 발췌 또는 짧게 재작성
- 예: “A일 경우 B이다”, “X는 Y를 전제로 한다”

2. <보기>의 조건/제약 사항
- <보기>에서 제시된 규칙, 전제, 제한 조건을 문장 단위로 정리
- 지문과 연결되는 표현이 있다면 해당 문장 위치 표시

3. 선택지별 근거 후보
- 각 선택지 문장에서 핵심 명사/동사를 추출
- 해당 표현이 지문 또는 <보기> 어디(문단/문장)에 등장하는지 표시
- 지문과 <보기> 모두에 없는 경우: “직접 대응 표현 없음”

4. 단어 형태 차이
- 지문/보기와 선택지에서 형태가 비슷하지만 다른 단어 쌍 나열
- 예: ‘필요조건’ vs ‘충분조건’, ‘일부’ vs ‘전체’

5. 판별에 필요한 최소 질문
- 이 문제를 판단하기 위해 반드시 확인해야 하는 사실 질문 1개
- 예: “<보기>의 조건이 지문의 사례에 적용되는가?”
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

