import os
import ast
import time
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# ⚙️ 설정 (여기만 수정하세요)
# =============================================================================

# OpenAI API 키 (https://platform.openai.com/api-keys)
OPENAI_API_KEY = "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"  # 여기에 입력 또는 환경변수 OPENAI_API_KEY 사용

# 샘플 개수 (None이면 전체, 숫자면 해당 개수만)
SAMPLE_SIZE = 3000  # 추천: 3000개 ($0.50)

# 파일 경로
INPUT_PATH = "data/combined_train.csv"
OUTPUT_PATH = "data/combined_train_with_reasoning.csv"

# API 설정
MAX_WORKERS = 5
MAX_TOKENS = 400
RETRY_COUNT = 3
RETRY_DELAY = 2
TEMPERATURE = 0.3

SYSTEM_PROMPT = """당신은 한국 수능/모의고사 문제 분석 전문가입니다.
연역적 추론(Deductive Reasoning)을 사용하여 문제를 분석합니다.

연역적 추론 방법:
1. 지문에서 핵심 사실(전제)을 파악
2. 정답 선택지가 이 전제와 논리적으로 일치하는지 검증
3. 근거를 바탕으로 결론 도출

반드시 한국어로만, 자연스러운 줄글 형태로 작성하세요."""

USER_PROMPT_TEMPLATE = """[문제]
지문: {paragraph}

질문: {question}
{question_plus}
선택지:
{choices}

정답: {answer}번

[작성 방법]
1. 지문에서 문제 해결에 필요한 핵심 내용을 파악하여 서술
2. 정답 선택지가 왜 지문과 일치하는지 구체적 근거 제시
3. 필요시 주요 오답이 틀린 이유 간단히 언급 (선택)
4. 마지막 문장: "따라서 정답은 {answer}번이다."

[규칙]
- 번호 매기지 말고 줄글로 자연스럽게 연결
- 3~5문장으로 간결하게
- 지문에 없는 내용 추론 금지
- 서론/인사말 없이 바로 시작

[예시]
지문 2문단에서 저자는 "기술 발전이 반드시 삶의 질 향상으로 이어지지 않는다"고 주장하고 있다. 3번 선택지의 '기술 만능주의에 대한 비판적 시각'은 이러한 저자의 관점과 정확히 일치한다. 반면 1번은 저자가 오히려 경계하는 입장이다. 따라서 정답은 3번이다."""

NEGATIVE_ADDITION = """
※ 주의: 이 문제는 '틀린 것' 또는 '적절하지 않은 것'을 찾는 문제입니다.
정답인 {answer}번이 왜 지문 내용과 일치하지 않거나 틀린지를 설명하세요.
다른 선택지들은 지문과 일치하므로 오답입니다."""


HISTORY_KEYWORDS = [
    '조선', '고려', '신라', '백제', '고구려', '가야', '발해', '통일신라',
    '삼국', '남북국', '후삼국', '대한제국', '일제', '강점기',
    '세종', '정조', '영조', '태조', '광해군', '연산군', '세조',
    '이순신', '안중근', '김구', '유관순', '안창호',
    '임진왜란', '병자호란', '동학', '3·1운동', '독립운동',
    '과거제', '신분제', '토지제도', '봉건', '개혁',
    '의병', '독립협회', '대한민국임시정부',
    '고분', '유물', '비석', '탑', '불상', '도자기',
]

SOCIETY_KEYWORDS = [
    '경제', '시장', '금리', '물가', 'GDP', '무역', '수요', '공급',
    '인플레이션', '환율', '재정', '통화', '세금', '예산',
    '정부', '법률', '헌법', '국회', '선거', '민주주의', '정당',
    '삼권분립', '기본권', '재판', '위헌', '대통령', '국무총리',
    '복지', '인구', '사회보장', '노동', '고용', '실업',
    '그래프', '표', '통계', '증가율', '비율', '%', '감소',
]

NEGATIVE_PATTERNS = [
    "적절하지 않은", "옳지 않은", "않는 것", "않은 것", "아닌 것",
    "잘못된", "틀린", "부적절한", "해당하지 않는", "거리가 먼",
    "일치하지 않는", "부합하지 않는", "맞지 않는",
]


def classify_subject(row: dict) -> str:
    """키워드 기반 과목 분류"""
    id_str = str(row.get('id', '')).lower()
    if 'history' in id_str:
        return 'history'
    
    paragraph = str(row.get('paragraph', '')) if row.get('paragraph') else ''
    problems = str(row.get('problems', ''))
    text = paragraph + problems
    
    history_score = sum(1 for kw in HISTORY_KEYWORDS if kw in text)
    society_score = sum(1 for kw in SOCIETY_KEYWORDS if kw in text)
    
    if history_score >= 2:
        return 'history'
    elif society_score >= 2:
        return 'society'
    else:
        return 'korean'


def is_negative_question(question: str) -> bool:
    """Negative 문제 여부 판별"""
    return any(pattern in question for pattern in NEGATIVE_PATTERNS)



def init_client():
    """OpenAI 클라이언트 초기화"""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai 패키지가 없습니다.")
        print("   실행: pip install openai")
        exit(1)
    
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY를 설정하세요!")
        print("   방법 1: 스크립트 상단 OPENAI_API_KEY 변수에 직접 입력")
        print("   방법 2: OPENAI_API_KEY='sk-...' uv run python ...")
        exit(1)
    
    return OpenAI(api_key=api_key)


def create_prompt(row: dict) -> tuple:
    """프롬프트 생성"""
    problems = row['problems']
    if isinstance(problems, str):
        problems = ast.literal_eval(problems)
    
    choices = problems['choices']
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    answer = problems['answer']
    question = problems['question']
    
    # paragraph null 처리
    paragraph = row.get('paragraph', '')
    if pd.isna(paragraph) or paragraph is None:
        paragraph = "(지문 없음)"
    
    # question_plus 처리
    question_plus = row.get('question_plus', '')
    if pd.isna(question_plus) or question_plus is None or question_plus == '':
        question_plus_str = ""
    else:
        question_plus_str = f"\n<보기>:\n{question_plus}\n"
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        paragraph=paragraph,
        question=question,
        question_plus=question_plus_str,
        choices=choices_str,
        answer=answer
    )
    
    if is_negative_question(question):
        user_prompt += NEGATIVE_ADDITION.format(answer=answer)
    
    return user_prompt, answer


def generate_reasoning(client, row: dict) -> str:
    """GPT-4o mini로 reasoning 생성"""
    user_prompt, answer = create_prompt(row)
    
    for attempt in range(RETRY_COUNT):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            
            if f"정답은 {answer}번" not in result:
                result += f" 따라서 정답은 {answer}번이다."
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            if attempt < RETRY_COUNT - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                if "rate" in error_msg.lower():
                    wait_time = 10
                time.sleep(wait_time)
            else:
                return f"[ERROR] {error_msg[:100]}"
    
    return "[ERROR] Max retries exceeded"


def process_row(args):
    """병렬 처리용 래퍼"""
    client, idx, row = args
    reasoning = generate_reasoning(client, row)
    return idx, reasoning


def balanced_sample(df: pd.DataFrame, n: int, ratio: dict, seed: int = 42) -> pd.DataFrame:
    """    
    Args:
        df: 원본 데이터프레임
        n: 총 샘플 수
        ratio: 과목별 목표 비율 {'korean': 0.5, 'society': 0.3, 'history': 0.2}
        seed: 랜덤 시드
    """
    random.seed(seed)
    
    # 과목 분류
    df = df.copy()
    df['_subject'] = df.apply(lambda x: classify_subject(x.to_dict()), axis=1)
    
    sampled_parts = []
    actual_counts = {}
    
    for subject, target_ratio in ratio.items():
        subject_df = df[df['_subject'] == subject]
        target_n = int(n * target_ratio)
        
        # 해당 과목 데이터가 목표보다 적으면 전부 사용
        actual_n = min(target_n, len(subject_df))
        
        if actual_n > 0:
            sampled_parts.append(subject_df.sample(n=actual_n, random_state=seed))
            actual_counts[subject] = actual_n
    
    result = pd.concat(sampled_parts, ignore_index=True)
    
    # 목표 개수에 못 미치면 남은 데이터에서 추가 (주로 korean에서)
    if len(result) < n:
        already_ids = set(result['id'].tolist())
        extra_pool = df[~df['id'].isin(already_ids)]
        extra_n = min(n - len(result), len(extra_pool))
        if extra_n > 0:
            extra = extra_pool.sample(n=extra_n, random_state=seed)
            result = pd.concat([result, extra], ignore_index=True)
    
    return result.drop(columns=['_subject']).reset_index(drop=True), actual_counts


def save_checkpoint(df: pd.DataFrame, output_path: str):
    """중간 저장"""
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"⚠️ 중간 저장 실패: {e}")

def main():
    print("=" * 65)
    print("🚀 Reasoning 데이터 생성 (Deductive 전략)")
    print("=" * 65)
    print(f"  모델: GPT-4o mini")
    print(f"  전략: 연역적 추론 3단계")
    print(f"  입력: {INPUT_PATH}")
    print(f"  출력: {OUTPUT_PATH}")
    print(f"  샘플: {SAMPLE_SIZE}개")
    print(f"  비율: korean {int(SAMPLE_RATIO['korean']*100)}% / society {int(SAMPLE_RATIO['society']*100)}% / history {int(SAMPLE_RATIO['history']*100)}%")
    print("=" * 65)
    
    client = init_client()
    print("OpenAI API 연결 성공\n")
    
    if not os.path.exists(INPUT_PATH):
        print(f"입력 파일 없음: {INPUT_PATH}")
        exit(1)
    
    df = pd.read_csv(INPUT_PATH)
    print(f"원본 데이터: {len(df)}개")
    
    null_para = df['paragraph'].isna().sum()
    if null_para > 0:
        print(f"paragraph가 비어있는 데이터: {null_para}개 (처리됨)")
    
    # 과목별 분포 출력
    df_temp = df.copy()
    df_temp['_subject'] = df_temp.apply(lambda x: classify_subject(x.to_dict()), axis=1)
    subject_counts = df_temp['_subject'].value_counts()
    print(f"\n원본 과목별 분포:")
    for subject, count in subject_counts.items():
        pct = count / len(df) * 100
        print(f"   - {subject}: {count}개 ({pct:.1f}%)")
    del df_temp
    
    # 기존 진행분 확인
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        if 'reasoning' in existing_df.columns:
            valid_mask = existing_df['reasoning'].notna() & ~existing_df['reasoning'].str.startswith('[ERROR]', na=False)
            done_count = valid_mask.sum()
            print(f"\n기존 진행분: {done_count}개 완료")
            
            if done_count >= SAMPLE_SIZE:
                print("이미 목표 달성!")
                return
            
            df = existing_df.copy()
    
    # reasoning 컬럼 없으면 추가
    if 'reasoning' not in df.columns:
        df['reasoning'] = None
    
    # 비율 완화 샘플링
    todo_mask = df['reasoning'].isna() | df['reasoning'].str.startswith('[ERROR]', na=False)
    todo_df = df[todo_mask].copy()
    
    already_done = len(df) - len(todo_df)
    needed = SAMPLE_SIZE - already_done
    
    if needed <= 0:
        print("이미 목표 달성!")
        return
    
    print(f"\n 비율 완화 샘플링 중... (필요: {needed}개)")
    print(f"   목표 비율: korean {int(SAMPLE_RATIO['korean']*100)}% / society {int(SAMPLE_RATIO['society']*100)}% / history {int(SAMPLE_RATIO['history']*100)}%")
    
    sampled, actual_counts = balanced_sample(todo_df, needed, SAMPLE_RATIO, RANDOM_SEED)
    todo_indices = sampled.index.tolist()
    
    print(f"\n📊 샘플링 결과:")
    for subject, count in actual_counts.items():
        pct = count / len(sampled) * 100
        print(f"   - {subject}: {count}개 ({pct:.1f}%)")
    
    print(f"\n📝 생성 대상: {len(todo_indices)}개")
    
    if not todo_indices:
        print("모든 샘플 완료!")
        return
    
    
    # 태스크 생성
    tasks = [(client, idx, df.loc[idx].to_dict()) for idx in todo_indices]
    
    # 병렬 처리
    print(f"\n생성 시작...\n")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, task): task[1] for task in tasks}
        
        with tqdm(total=len(futures), desc="Generating", ncols=80) as pbar:
            for future in as_completed(futures):
                idx, reasoning = future.result()
                df.at[idx, 'reasoning'] = reasoning
                completed += 1
                pbar.update(1)
                
                # 50개마다 중간 저장
                if completed % 50 == 0:
                    save_checkpoint(df, OUTPUT_PATH)
    
    # 최종 저장
    df.to_csv(OUTPUT_PATH, index=False)
    
    # 결과 통계
    total_with_reasoning = df['reasoning'].notna().sum()
    error_count = df['reasoning'].str.startswith('[ERROR]', na=False).sum()
    success_count = total_with_reasoning - error_count
    
    print("\n" + "=" * 65)
    print("완료!")
    print("=" * 65)
    print(f"  성공: {success_count}개")
    print(f"  에러: {error_count}개")
    print(f"  저장: {OUTPUT_PATH}")
    
    if error_count > 0:
        print(f"\n 에러 {error_count}개는 재실행하면 자동 재시도됩니다.")
    
    # 샘플 출력
    print("\n" + "-" * 65)
    print("📝 샘플 확인")
    print("-" * 65)
    
    success_df = df[df['reasoning'].notna() & ~df['reasoning'].str.startswith('[ERROR]', na=False)]
    if len(success_df) > 0:
        sample = success_df.iloc[-1]
        problems = sample['problems']
        if isinstance(problems, str):
            problems = ast.literal_eval(problems)
        
        print(f"질문: {problems['question'][:60]}...")
        print(f"정답: {problems['answer']}번")
        print(f"\nReasoning:\n{sample['reasoning']}")
    
    print("=" * 65)


if __name__ == "__main__":
    main()
