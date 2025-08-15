# Training Visualization Guide

이 가이드는 학습 과정을 시각화하고 체크포인트를 관리하는 방법을 설명합니다.

# Training Visualization Guide

이 가이드는 학습 과정을 시각화하고 체크포인트를 관리하는 방법을 설명합니다.

## 학습 실행

### 기본 실행
```bash
uv run python train.py
```

### Loss 함수별 학습
```bash
# DPLossV1으로 학습
uv run python train.py --loss-function DPLossV1

# DPLossV2로 학습 (tau 값 조정)
uv run python train.py --loss-function DPLossV2 --tau 0.5

# DPLossV1Norm으로 학습
uv run python train.py --loss-function DPLossV1Norm

# DPLossV2Norm으로 학습
uv run python train.py --loss-function DPLossV2Norm --tau 2.0
```


### 옵션을 사용한 실행
```bash
# 에포크 수와 체크포인트 저장 간격 설정
uv run python train.py --epochs 50 --save-interval 5

# 배치 크기와 학습률 설정
uv run python train.py --batch-size 256 --lr 0.0001

# 잠재 차원 설정
uv run python train.py --latent-dim 256

# 모든 옵션 조합
uv run python train.py --epochs 100 --batch-size 1024 --save-interval 10 --lr 0.001 --latent-dim 512 --loss-function DPLossV2 --tau 1.5
```

### 환경변수 사용
```bash
# Windows (PowerShell)
$env:SAVE_INTERVAL=20; uv run python train.py

# Linux/Mac
SAVE_INTERVAL=20 uv run python train.py
```

### 사용 가능한 옵션
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--batch-size`: 배치 크기 (기본값: 512)
- `--save-interval`: 체크포인트 저장 간격 (기본값: 10에포크마다)
- `--lr`: 학습률 (기본값: 0.001)
- `--latent-dim`: 잠재 벡터 차원 (기본값: 512)
- `--loss-function`: 손실 함수 선택 (DPLossV1, DPLossV1Norm, DPLossV2, DPLossV2Norm)
- `--tau`: Soft ranking 온도 파라미터 (V2 계열 손실 함수용, 기본값: 1.0)

### 도움말 보기
```bash
uv run python train.py --help
```

## Loss 함수 설명

- **DPLossV1**: 기본 순위 기반 손실 (hard ranking)
- **DPLossV1Norm**: 정규화된 순위 기반 손실 
- **DPLossV2**: 미분 가능한 soft ranking 손실
- **DPLossV2Norm**: 정규화된 soft ranking 손실

V2 계열 손실 함수는 `--tau` 파라미터로 soft ranking의 온도를 조절할 수 있습니다:
- 낮은 tau (0.1-0.5): 더 sharp한 ranking
- 높은 tau (2.0-5.0): 더 smooth한 ranking

## 체크포인트 저장 규칙

- **손실값 기록**: 매 에포크마다 기록됨
- **latest_model.pth**: 매 에포크마다 업데이트됨
- **체크포인트 파일**: 지정된 간격 또는 마지막 에포크에만 저장됨
- **config.json**: 사용된 loss 함수와 모든 하이퍼파라미터 정보 포함

예시: `--save-interval 10`인 경우
- 10, 20, 30, ..., 100 에포크에 저장
- 마지막 에포크가 간격에 맞지 않아도 항상 저장 (예: 95 에포크로 끝나면 95도 저장)

학습이 시작되면 다음과 같은 정보가 출력됩니다:
- 고유한 실험 ID (UUID)
- 체크포인트 저장 경로
- 사용된 Loss 함수
- 에포크별 손실값
- 체크포인트 저장 시점 표시

## 체크포인트 구조

```
checkpoints/
├── <experiment-uuid>/
│   ├── config.json              # 실험 설정
│   ├── train_history.json       # 학습 히스토리
│   ├── latest_model.pth         # 최종 모델
│   ├── checkpoint_epoch_001.pth # 에포크별 체크포인트
│   ├── checkpoint_epoch_002.pth
│   └── ...
```

## 시각화 방법

### 1. 빠른 시각화 (최신 실험)

```bash
uv run python quick_plot.py
```

가장 최근 실험의 학습 곡선을 자동으로 시각화합니다.

### 2. 상세 시각화

```bash
# 사용 가능한 실험 목록 보기
uv run python visualize_training.py --list

# 대화형 모드 (실험 선택) - recall 자동 계산
uv run python visualize_training.py

# 빠른 시각화 (recall 계산 생략)
uv run python visualize_training.py --no-recall

# 에포크별 recall 히스토리 포함 시각화
uv run python visualize_training.py --recall-history

# 특정 실험 ID로 시각화
uv run python visualize_training.py --experiment_id <uuid>

# 그래프를 파일로 저장
uv run python visualize_training.py --save my_plot.png

# 다중 실험 비교 (recall 포함)
uv run python visualize_training.py --compare

# 빠른 다중 실험 비교 (recall 제외)
uv run python visualize_training.py --compare --no-recall

# 다중 실험 비교 + 에포크별 recall 히스토리
uv run python visualize_training.py --compare --recall-history
```

### 시각화 기능

**개별 실험 시각화:**
- Total Loss (로그 스케일)
- Rank Loss (로그 스케일)
- Pair Distance Loss (로그 스케일)
- All Losses Combined (로그 스케일)
- KNN Recall@k (최종 값 또는 에포크별 히스토리, 로그 스케일)
- Performance Summary

**다중 실험 비교:**
- 모든 손실 함수 로그 스케일 비교
- Recall 성능 바 차트
- 실험별 성능 요약 테이블
- 에포크별 recall 히스토리 비교 (로그 스케일, --recall-history 옵션)

**Recall 계산:**
- 저장된 모델(latest_model.pth)로 최종 recall 자동 계산
- --recall-history 옵션으로 에포크별 recall 히스토리 생성
- 3,000개 샘플로 메모리 효율적 처리
- --no-recall 옵션으로 빠른 시각화 가능

## 체크포인트에서 모델 로드

```python
import torch
import json

# 체크포인트 로드
checkpoint = torch.load("checkpoints/<experiment-uuid>/checkpoint_epoch_100.pth")

# 모델 상태 복원
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# 학습 히스토리 확인
train_history = checkpoint["train_history"]
config = checkpoint["config"]
```

## 실험 비교

여러 실험을 비교하려면 각 실험의 `train_history.json` 파일을 읽어서 비교할 수 있습니다:

```python
import json
import matplotlib.pyplot as plt

# 두 실험의 히스토리 로드
with open("checkpoints/exp1/train_history.json") as f:
    hist1 = json.load(f)

with open("checkpoints/exp2/train_history.json") as f:
    hist2 = json.load(f)

# 비교 플롯
plt.figure(figsize=(10, 6))
plt.plot(hist1["epochs"], hist1["total_loss"], label="Experiment 1")
plt.plot(hist2["epochs"], hist2["total_loss"], label="Experiment 2")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.legend()
plt.show()
```

## 파일 설명

- `train.py`: 학습 스크립트 (UUID 기반 체크포인트 저장)
- `visualize_training.py`: 상세한 학습 결과 시각화
- `quick_plot.py`: 빠른 시각화 (최신 실험)
- `test.py`: 모델 평가 스크립트
