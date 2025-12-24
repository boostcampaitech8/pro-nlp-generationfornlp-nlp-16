import pandas as pd
from datasets import load_dataset
import ast  # 텍스트로 된 딕셔너리를 진짜 딕셔너리로 바꿔주는 도구

# 1. 데이터 로드
dataset = load_dataset("csv", data_files={"train": "/data/ephemeral/home/han/han/data/train.csv"})
df = pd.DataFrame(dataset['train'])

print(f"✅ 데이터 로드 완료: {len(df)}개")

# 2. 'problems' 컬럼에서 'answer' 꺼내기 (핵심!)
def extract_answer(problem_str):
    try:
        # 문자열("{'answer': 1 ...}")을 진짜 딕셔너리로 변환
        problem_dict = ast.literal_eval(problem_str)
        return problem_dict['answer']
    except:
        return None

# 'answer' 컬럼을 새로 만듭니다
df['answer'] = df['problems'].apply(extract_answer)

# 3. 정답 분포 계산
counts = df['answer'].value_counts().sort_index()
total = len(df)

print("\n=== 📊 [데이터셋 정답 분포 분석] ===")
print(f"총 데이터 개수: {total}개")
print("-" * 30)
print(f"{'정답':<5} {'개수':<10} {'비율(%)'}")
print("-" * 30)

for label, count in counts.items():
    if pd.isna(label): continue # 혹시 에러난 건 건너뜀
    ratio = (count / total) * 100
    bar = "█" * int(ratio // 2) 
    print(f"{int(label):<5} {count:<10} {ratio:.2f}%  {bar}")

print("-" * 30)

# 4. 결론 출력
max_label = counts.idxmax()
max_ratio = (counts.max() / total) * 100
print(f"🚨 결론: 정답 {int(max_label)}번에 데이터가 {max_ratio:.1f}% 쏠려있음.")
print("   -> 모델이 학습을 포기하고 '무조건 1번'만 찍게 만드는 원인.")