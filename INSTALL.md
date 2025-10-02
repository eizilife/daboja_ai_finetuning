# 🛠️ 설치 가이드

자세한 설치 및 환경 설정 방법을 안내합니다.

## 📋 시스템 요구사항

### 최소 요구사항
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.10 이상
- **GPU**: NVIDIA GPU with CUDA 11.8+ (권장: RTX 3060 12GB 이상)
- **RAM**: 16GB 이상
- **Storage**: 50GB 이상 (모델 + 데이터)

### 권장 사양
- **GPU**: RTX 4070/4080 또는 A100
- **RAM**: 32GB
- **Storage**: SSD 100GB

---

## 🚀 빠른 설치

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/llama3.2-korean-summarization.git
cd llama3.2-korean-summarization
```

### 2. Python 가상환경 생성
```bash
# conda 사용 시
conda create -n llama32-ko python=3.10
conda activate llama32-ko

# venv 사용 시
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. CUDA 설정 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

## 🔧 상세 설치 가이드

### Windows 사용자

#### CUDA 설치
1. [NVIDIA Driver](https://www.nvidia.com/drivers/) 최신 버전 설치
2. [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) 설치
3. 환경변수 확인:
```cmd
echo %CUDA_PATH%
nvcc --version
```

#### PyTorch 설치
```bash
# CUDA 11.8용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1용 (최신)
pip install torch torchvision torchaudio
```

#### 추가 Windows 설정
```bash
# Visual Studio Build Tools 필요 시
pip install --upgrade setuptools wheel
```

### Linux 사용자

#### CUDA 설치
```bash
# Ubuntu 20.04/22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### 환경변수 설정
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### macOS 사용자

⚠️ **주의**: macOS는 NVIDIA GPU를 지원하지 않아 CPU 학습만 가능합니다.

```bash
# MPS (Apple Silicon) 지원
pip install torch torchvision torchaudio

# Rosetta 2 필요 시 (Intel Mac)
arch -arm64 pip install torch torchvision torchaudio
```

---

## 🔑 Hugging Face 설정

### 계정 설정
1. [Hugging Face](https://huggingface.co) 회원가입
2. [Llama 3.2 라이선스](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 동의
3. Access Token 생성

### 로그인 방법

#### 방법 1: CLI 로그인
```bash
huggingface-cli login
# 토큰 입력 후 엔터
```

#### 방법 2: 환경변수
```bash
export HF_TOKEN="hf_your_token_here"
```

#### 방법 3: .env 파일
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

---

## 📦 데이터 준비

### AIHub 데이터셋 다운로드
1. [AIHub 문서요약 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=97) 접속
2. 회원가입 후 데이터 신청
3. 다운로드 후 압축 해제:

```bash
mkdir -p summary-ai/data/raw
# 다운로드한 파일들을 다음 구조로 배치:
# summary-ai/data/raw/
# ├── Train_신문기사_data/
# ├── Train_사설_data/
# ├── Valid_신문기사_data/
# └── Valid_사설_data/
```

---

## ⚡ 설치 검증

### 기본 검증
```bash
python -c "
import torch
import transformers
import peft
import bitsandbytes

print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

### 모델 로딩 테스트
```bash
cd summary-ai/scripts
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
print('✅ 토크나이저 로딩 성공')
"
```

---

## 🚨 문제 해결

### CUDA Out of Memory
```bash
# GPU 메모리 확인
nvidia-smi

# 메모리 부족 시 스왑 파일 증설 (Windows)
# 시스템 > 고급 시스템 설정 > 성능 설정 > 가상 메모리 변경
```

### Import 오류
```bash
# 캐시 정리
pip cache purge
pip install --upgrade --force-reinstall torch transformers

# 권한 문제 (Linux)
sudo chown -R $USER:$USER ~/.cache/huggingface
```

### Windows에서 bitsandbytes 오류
```bash
# 사전 컴파일된 버전 설치
pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

### 느린 다운로드 속도
```bash
# 한국 미러 사용
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📊 벤치마크 테스트

설치 완료 후 간단한 성능 테스트:

```bash
cd summary-ai/scripts
python test_model.py --quick-test
```

예상 결과:
- RTX 3060: ~5초
- RTX 4070: ~3초
- A100: ~1초

---

## 💡 성능 최적화 팁

### GPU 활용률 최적화
```python
# 환경변수 설정
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 메모리 효율성
```python
# gradient checkpointing 활성화
torch.utils.checkpoint.checkpoint_sequential()
```

### Windows 특화 설정
```python
# 멀티프로세싱 문제 해결
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
dataloader_num_workers = 0
```

---

## 🆘 지원

설치 관련 문제는 다음을 확인하세요:
1. [Issues](https://github.com/yourusername/llama3.2-korean-summarization/issues)
2. [Discussions](https://github.com/yourusername/llama3.2-korean-summarization/discussions)
3. [Wiki](https://github.com/yourusername/llama3.2-korean-summarization/wiki)

---

**✅ 설치 완료 후 [사용법 가이드](USAGE.md)를 확인하세요!**