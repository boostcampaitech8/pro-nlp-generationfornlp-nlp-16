"""
Reasoning 데이터 생성 스크립트
- GPT-4o mini 사용 (무료 크레딧 $5로 충분)
- 3,000개 샘플링 기본
- 중단 후 재개 지원

사용법:
1. pip install openai pandas tqdm
2. OPENAI_API_KEY 설정
3. python generate_reasoning.py
"""

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
OPENAI_API_KEY = " [API_KEY_REMOVED] # 여기에 입력 또는 환경변수 OPENAI_API_KEY 사용

# 샘플 개수 (None이면 전체, 숫자면 해당 개수만)
SAMPLE_SIZE = 3000  # 추천: 3000개 ($0.50)

# 파일 경로
INPUT_PATH = "data/combined_train.csv"
OUTPUT_PATH = "data/combined_train_with_reasoning.csv"

# API 설정
MAX_WORKERS = 5      # 병렬 처리 수
MAX_TOKENS = 500     # 출력 토큰 제한 (3~6문장, 여유있게)
RETRY_COUNT = 3      # 재시도 횟수
RETRY_DELAY = 2      # 재시도 대기(초)

# =============================================================================
# 🎯 프롬프트 (검토 완료 - 수정하지 마세요)
# =============================================================================

SYSTEM_PROMPT = """당신은 논리적인 한국 수능 문제 해설 전문가입니다.
단순한 정답 확인이 아니라, 정답에 도달하는 '논리적 사고 과정'을 명확하게 보여줍니다.
학생이 이해하기 쉽도록 인과관계를 중심으로 서술하세요."""

USER_PROMPT_TEMPLATE = """[문제]
지문: {paragraph}

질문: {question}
{question_plus}

선택지:
{choices}

정답: {answer}번

[작성 규칙]
1. 문제의 핵심 요구사항과 지문의 관련 내용을 자연스럽게 연결하여 서술
2. 정답 선지가 도출되는 구체적인 근거를 지문에서 인용하거나 재진술
3. 매력적인 오답이 있다면 왜 틀렸는지 논리적 접속사(반면, 하지만 등)를 사용해 반박
4. 번호를 매기지 말고 줄글 형태로 자연스럽게 이어질 것
5. 마지막 문장은 반드시 "따라서 정답은 {answer}번이다."로 종료
6. 핵심 논리를 담아 3~6문장 내외로 간결하게 작성

[과목별 예시]

(국어) 이 문제는 글쓴이의 관점을 파악해야 한다. 지문 2문단에서 "기술의 발전이 인간 소외를 초래한다"고 명시하고 있으며, 3번 선지의 '기술 만능주의 비판'이 이와 맥락을 같이 한다. 반면 1번은 지문의 논지와 정반대되는 내용이다. 따라서 정답은 3번이다.

(한국사) 자료에 제시된 '토지 조사 사업'은 1910년대 일제 강점기의 대표적인 식민 정책이다. 4번의 '경작권 부정'은 신고주의 원칙에 따른 이 사업의 핵심 결과와 일치한다. 2번의 지계 발급은 대한제국 광무개혁 시기의 사실이므로 시기가 맞지 않는다. 따라서 정답은 4번이다."""


# =============================================================================
# 코드 (수정 불필요)
# =============================================================================

def init_client():
    """OpenAI 클라이언트 초기화"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai 패키지가 없습니다.")
        print("   실행: pip install openai")
        exit(1)
    
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY를 설정하세요!")
        print("   방법 1: 스크립트 상단 OPENAI_API_KEY 변수에 직접 입력")
        print("   방법 2: export OPENAI_API_KEY='sk-...'")
        exit(1)
    
    return OpenAI(api_key=api_key)


def create_prompt(row: dict) -> tuple:
    """프롬프트 생성"""
    problems = row['problems']
    if isinstance(problems, str):
        problems = ast.literal_eval(problems)
    
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(problems['choices'])])
    answer = problems['answer']
    question = problems['question']
    paragraph = row['paragraph']
    
    # question_plus 처리
    question_plus = row.get('question_plus', '')
    if pd.isna(question_plus) or question_plus == '':
        question_plus_str = ""
    else:
        question_plus_str = f"\n<보기>\n{question_plus}"
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        paragraph=paragraph,
        question=question,
        question_plus=question_plus_str,
        choices=choices_str,
        answer=answer
    )
    
    return user_prompt, answer


def generate_reasoning(client, row: dict) -> str:
    """GPT-4o mini로 reasoning 생성"""
    user_prompt, answer = create_prompt(row)
    
    for attempt in range(RETRY_COUNT):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=MAX_TOKENS,
                temperature=0.3,  # 일관성을 위해 낮게
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            
            # 마지막 문장 검증 (없으면 추가)
            if f"정답은 {answer}번" not in result:
                result += f" 따라서 정답은 {answer}번이다."
            
            return result
            
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"\n⚠️ Error for {row['id']}: {e}")
                return f"[ERROR] {str(e)}"
    
    return "[ERROR] Max retries exceeded"


def process_single(args):
    """단일 샘플 처리 (병렬용)"""
    idx, row, client = args
    reasoning = generate_reasoning(client, row)
    return idx, row['id'], reasoning


def sample_data(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """층화 샘플링 (다양한 문제 유형 포함)"""
    if n >= len(df):
        return df
    
    random.seed(seed)
    
    # 문제 유형 분류 (negative 문제 우선 포함)
    def get_question_type(row):
        problems = ast.literal_eval(row['problems']) if isinstance(row['problems'], str) else row['problems']
        question = problems['question']
        if any(kw in question for kw in ['않은', '않는', '틀린', '아닌']):
            return 'negative'
        elif '있는 대로' in question or '모두 고른' in question:
            return 'combination'
        return 'positive'
    
    df = df.copy()
    df['_type'] = df.apply(get_question_type, axis=1)
    
    # 각 유형별로 비율 맞춰서 샘플링
    sampled = []
    type_counts = df['_type'].value_counts()
    
    for qtype in type_counts.index:
        type_df = df[df['_type'] == qtype]
        # 비율 유지하되 최소 개수 보장
        type_n = max(int(n * len(type_df) / len(df)), min(50, len(type_df)))
        type_n = min(type_n, len(type_df))
        sampled.append(type_df.sample(n=type_n, random_state=seed))
    
    result = pd.concat(sampled).drop(columns=['_type'])
    
    # 목표 개수에 맞추기
    if len(result) < n:
        remaining = df[~df['id'].isin(result['id'])].drop(columns=['_type'])
        extra = remaining.sample(n=min(n - len(result), len(remaining)), random_state=seed)
        result = pd.concat([result, extra])
    elif len(result) > n:
        result = result.sample(n=n, random_state=seed)
    
    return result.reset_index(drop=True)


def main():
    print("=" * 60)
    print("🚀 Reasoning 데이터 생성")
    print("=" * 60)
    print(f"모델: GPT-4o mini")
    print(f"입력: {INPUT_PATH}")
    print(f"출력: {OUTPUT_PATH}")
    print(f"샘플 수: {SAMPLE_SIZE if SAMPLE_SIZE else '전체'}")
    print("=" * 60)
    
    # 클라이언트 초기화
    client = init_client()
    print("✅ OpenAI API 연결 성공")
    
    # 데이터 로드
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ 원본 데이터: {len(df)}개")
    
    # 샘플링
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df_target = sample_data(df, SAMPLE_SIZE)
        print(f"✅ 샘플링: {len(df_target)}개 선택")
    else:
        df_target = df
    
    # 이미 처리된 데이터 확인 (중단 후 재개용)
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        existing_ids = set(existing_df['id'].tolist())
        df_to_process = df_target[~df_target['id'].isin(existing_ids)]
        print(f"✅ 기존 진행: {len(existing_ids)}개")
        print(f"✅ 남은 작업: {len(df_to_process)}개")
    else:
        df_to_process = df_target
        existing_df = None
    
    if len(df_to_process) == 0:
        print("✅ 모든 데이터 처리 완료!")
        return
    
    # 비용 예상
    estimated_cost = len(df_to_process) * 0.00017  # 대략적 추정
    print(f"\n💰 예상 비용: ${estimated_cost:.2f}")
    print(f"   (무료 크레딧 $5 내에서 가능)")
    
    input("\n⏎ Enter를 눌러 시작...")
    
    # Reasoning 생성
    print(f"\n🔄 생성 시작 (병렬: {MAX_WORKERS})")
    
    results = {}
    rows_list = df_to_process.to_dict('records')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single, (i, row, client)): i 
            for i, row in enumerate(rows_list)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            idx, row_id, reasoning = future.result()
            results[idx] = reasoning
            
            # 중간 저장 (100개마다)
            if len(results) % 100 == 0:
                _save_intermediate(df_to_process, results, existing_df)
    
    # 최종 저장
    df_to_process = df_to_process.copy()
    df_to_process['reasoning'] = [results[i] for i in range(len(results))]
    
    if existing_df is not None:
        final_df = pd.concat([existing_df, df_to_process], ignore_index=True)
    else:
        final_df = df_to_process
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"저장: {OUTPUT_PATH}")
    print(f"총 샘플: {len(final_df)}개")
    
    error_count = final_df['reasoning'].str.contains(r'\[ERROR\]', regex=True).sum()
    if error_count > 0:
        print(f"⚠️ 에러: {error_count}개 (재실행하면 자동 재시도)")
    
    # 샘플 출력
    print("\n" + "-" * 60)
    print("📝 샘플 확인")
    print("-" * 60)
    sample = final_df[~final_df['reasoning'].str.contains(r'\[ERROR\]', regex=True, na=False)].iloc[0]
    problems = ast.literal_eval(sample['problems']) if isinstance(sample['problems'], str) else sample['problems']
    print(f"질문: {problems['question'][:50]}...")
    print(f"정답: {problems['answer']}번")
    print(f"\nReasoning:")
    print(sample['reasoning'])


def _save_intermediate(df_to_process, results, existing_df):
    """중간 저장"""
    try:
        temp_df = df_to_process.iloc[:len(results)].copy()
        temp_df['reasoning'] = [results[i] for i in range(len(results))]
        if existing_df is not None:
            temp_df = pd.concat([existing_df, temp_df], ignore_index=True)
        temp_df.to_csv(OUTPUT_PATH + ".tmp", index=False)
    except:
        pass


if __name__ == "__main__":
    main()