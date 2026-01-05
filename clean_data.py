import pandas as pd
import os

# 파일 경로 설정
input_path = "data/combined_train_with_reasoning.csv"
output_path = "data/train_clean_3000.csv"

# 파일이 있는지 확인
if not os.path.exists(input_path):
    print(f"❌ 오류: 파일이 없습니다 -> {input_path}")
    exit(1)

# 데이터 로드
print("📂 데이터 불러오는 중...")
df = pd.read_csv(input_path)
total_count = len(df)

# 필터링: reasoning이 있고(notna), 빈 문자열이 아닌('') 것만 남김
print("🧹 데이터 청소 중...")
df_clean = df[df['reasoning'].notna() & (df['reasoning'] != "")]
clean_count = len(df_clean)

# 저장
print(f"💾 저장 중... ({output_path})")
df_clean.to_csv(output_path, index=False)

# 결과 출력
print("=" * 40)
print(f"✅ 청소 완료!")
print(f" - 원본 데이터: {total_count}개")
print(f" - 남은 데이터: {clean_count}개")
print(f" - 삭제된 데이터: {total_count - clean_count}개")
print("=" * 40)