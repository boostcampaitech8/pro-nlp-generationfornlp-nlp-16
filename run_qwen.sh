#!/bin/bash

# 1. 세션 이름 설정
SESSION_NAME="qwen_experiment"

# 2. 실행할 명령어 정의
# 학습 명령어
CMD_TRAIN="uv run python train.py model=qwen_qlora"

# 추론 명령어
CMD_INFERENCE="uv run python inference.py model=qwen_qlora"

# 3. tmux 세션 생성 (백그라운드 모드: -d)
tmux new-session -d -s $SESSION_NAME

# 4. 명령어 전송
# '&&'를 사용하여 train이 성공적으로 끝났을 때만 inference가 실행되도록 합니다.
tmux send-keys -t $SESSION_NAME "$CMD_TRAIN && $CMD_INFERENCE" C-m

# 5. 사용자 안내 메시지 출력
echo "---------------------------------------------------"
echo "Tmux 세션 '$SESSION_NAME'이 생성되고 작업이 시작되었습니다."
echo "확인하려면 다음 명령어를 입력하세요:"
echo "tmux attach -t $SESSION_NAME"
echo "---------------------------------------------------"