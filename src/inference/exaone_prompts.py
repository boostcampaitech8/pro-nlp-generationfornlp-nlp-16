"""
EXAONE Inference용 프롬프트 템플릿

MoA (Mixture of Agents) 스타일의 프롬프트를 포함합니다.
"""

# 기본 System 프롬프트 (description 없이 사용)
SYSTEM_PROMPT_BASIC = """당신은 객관식 문제를 해결하는 학생입니다.
제시문과 질문으로 이루어진 문제가 주어집니다.

문제를 해결할 때에는 내부적으로 단계적인 추론을 충분히 수행한 후,
그 결과만을 바탕으로 각 선택지가 왜 맞거나 틀렸는지를
제시문에 근거하여 간결하고 객관적으로 설명하세요.

풀이 설명에는 불필요한 사고 과정이나 중간 추론 단계를 모두 나열하지 말고,
정답을 정당화하는 핵심 근거만을 논리적으로 정리하여 서술하세요.

풀이가 끝나면, 맨 마지막 줄에 아래 형식으로 정답만 제시하세요.
{"정답": "번호"}"""


# MoA 스타일 System 프롬프트 (description과 함께 사용)
SYSTEM_PROMPT_MOA = """당신은 수능 국어 문제를 해결하는 전문가입니다.

주어진 지문과 질문을 분석하고, 제공된 보조 정보들을 참고하여 정답을 도출하세요.

**중요한 지침:**
1. 제공된 보조 정보는 서로 다른 관점에서 작성되었으며, 일부는 편향되거나 부정확할 수 있습니다.
2. 이 정보들을 맹목적으로 따르지 말고, 지문 내용을 기반으로 비판적으로 평가하세요.
3. 단계적 추론을 통해 각 선택지를 검토하고, 지문에 근거한 객관적 판단을 내리세요.
4. 최종 답변은 제공된 정보를 종합하되, 지문 내용이 최우선 판단 기준입니다.

**출력 형식:**
- 각 선택지에 대한 간결한 분석을 제공하세요.
- 정답을 정당화하는 핵심 근거만을 논리적으로 서술하세요.
- 맨 마지막 줄에 아래 형식으로 정답을 제시하세요.

{"정답": "번호"}
"""


def create_user_prompt_basic(paragraph: str, question: str, choices: list, question_plus: str = None) -> str:
    """
    기본 user 프롬프트 생성 (description 없이)

    Args:
        paragraph: 지문
        question: 질문
        choices: 선택지 리스트
        question_plus: 보기 (옵션)

    Returns:
        포맷된 user 프롬프트
    """
    user_content = f"<제시문>\n{paragraph}\n\n"

    if question_plus:
        user_content += f"<보기>\n{question_plus}\n\n"

    user_content += f"<질문>\n{question}\n"

    for i, choice in enumerate(choices, 1):
        user_content += f"{i}. {choice}\n"

    return user_content


def create_user_prompt_with_descriptions(
    paragraph: str,
    question: str,
    choices: list,
    description_1: str,
    description_2: str,
    question_plus: str = None
) -> str:
    """
    MoA 스타일 user 프롬프트 생성 (description 포함)

    Args:
        paragraph: 지문
        question: 질문
        choices: 선택지 리스트
        description_1: 첫 번째 보조 정보
        description_2: 두 번째 보조 정보
        question_plus: 보기 (옵션)

    Returns:
        포맷된 user 프롬프트
    """
    user_content = f"<지문>\n{paragraph}\n\n"

    if question_plus:
        user_content += f"<보기>\n{question_plus}\n\n"

    user_content += f"<질문>\n{question}\n\n"

    user_content += "<선택지>\n"
    for i, choice in enumerate(choices, 1):
        user_content += f"{i}. {choice}\n"

    user_content += "\n<보조 정보>\n"
    user_content += "다음은 이 문제를 해결하는 데 도움이 될 수 있는 두 가지 관점의 분석입니다:\n\n"

    user_content += f"[분석 1]\n{description_1}\n\n"
    user_content += f"[분석 2]\n{description_2}\n\n"

    user_content += "위 보조 정보를 참고하되, 지문 내용을 기반으로 독립적으로 판단하여 정답을 선택하세요.\n"

    return user_content
