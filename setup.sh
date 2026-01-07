#!/bin/bash
set -e

echo "🚀 [Native C++] NLP 추론 전용 환경 구축 (Only /data/ephemeral)..."

########################################
# [0/5] 작업 공간 및 캐시 경로 설정
########################################

echo "[0/5] 작업 공간 및 캐시 경로 설정..."

WORK_DIR="/data/ephemeral/home/workspace"
mkdir -p "$WORK_DIR"
chmod 777 "$WORK_DIR"

export TMPDIR="/data/ephemeral/tmp"
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export XDG_CACHE_HOME="/data/ephemeral/home/shared/cache"
export PIP_CACHE_DIR="/data/ephemeral/home/shared/cache/pip"
export UV_CACHE_DIR="/data/ephemeral/home/shared/cache/uv"
export HF_HOME="/data/ephemeral/home/shared/cache/huggingface"

mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME"

cd "$WORK_DIR"
echo "📂 현재 작업 위치: $(pwd)"

########################################
# timezone / noninteractive
########################################

export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Seoul
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

########################################
# [1/5] 시스템 패키지 설치
########################################

echo "📦 [1/5] 시스템 패키지 설치 (Build tools)..."

apt-get update
apt-get install -y \
    vim wget build-essential cmake git curl htop tmux unzip sudo tree

apt-get clean
rm -rf /var/cache/apt/archives/*

########################################
# [2/5] CUDA 12.2 설치
########################################

echo "📦 [2/5] CUDA 12.2 확인 및 설치..."

CUDA_RUN="/data/ephemeral/workspace/cuda_installer.run"

if [ ! -d "/usr/local/cuda-12.2" ]; then
    echo "⬇️ CUDA 다운로드..."
    wget -O "$CUDA_RUN" \
      https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
    chmod +x "$CUDA_RUN"

    echo "⚙️ CUDA 설치 중..."
    sh "$CUDA_RUN" --silent --toolkit

    ln -sf /usr/local/cuda-12.2 /usr/local/cuda
    rm -f "$CUDA_RUN"
    echo "✅ CUDA 설치 완료"
else
    echo "✅ CUDA가 이미 존재합니다."
fi

rm -f "$CUDA_RUN"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

########################################
# [3/5] Python Client (uv)
########################################

echo "🐍 [3/5] uv 환경 구축..."

# uv 설치 (공식 방식)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv 기본 설치 경로 보장
export PATH="$HOME/.local/bin:$PATH"pip install uv
uv sync

source .venv/bin/activate

########################################
# [4/5] llama.cpp 빌드
########################################

echo "🔨 [4/5] Llama.cpp 원본 엔진 빌드 (GPU 가속)..."

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp
rm -rf build

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 6

echo "✅ 빌드 완료: $(pwd)/build/bin/llama-server 생성됨"
cd ..

########################################
# [5/5] 모델 디렉토리
########################################

echo "💾 [5/5] 모델 저장소 준비..."
mkdir -p models

########################################
# ~/.bashrc 영구 등록 (환경변수 선언부)
########################################

echo "[+] ~/.bashrc에 Ephemeral 환경 변수 등록 중..."

BASHRC="$HOME/.bashrc"
ENV_START="# >>> EPHEMERAL NLP ENV >>>"

if ! grep -q "$ENV_START" "$BASHRC"; then
cat << 'EOF' >> "$BASHRC"

# >>> EPHEMERAL NLP ENV >>>
# Native C++ / NLP inference (ephemeral only)

export WORK_DIR="/data/ephemeral/home/workspace"

export TMPDIR="/data/ephemeral/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export XDG_CACHE_HOME="/data/ephemeral/home/shared/cache"
export PIP_CACHE_DIR="/data/ephemeral/home/shared/cache/pip"
export UV_CACHE_DIR="/data/ephemeral/home/shared/cache/uv"
export HF_HOME="/data/ephemeral/home/shared/cache/huggingface"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDACXX="/usr/local/cuda/bin/nvcc"

mkdir -p "$WORK_DIR" \
         "$TMPDIR" \
         "$XDG_CACHE_HOME" \
         "$PIP_CACHE_DIR" \
         "$UV_CACHE_DIR" \
         "$HF_HOME"

# <<< EPHEMERAL NLP ENV <<<
EOF
    echo "[+] ~/.bashrc 등록 완료"
else
    echo "[!] ~/.bashrc에 이미 Ephemeral ENV 블록이 존재합니다. 건너뜁니다."
fi

# 현재 쉘에도 즉시 반영
source "$HOME/.bashrc"

########################################
# 완료
########################################

echo "🎉 [설치 완료]"
