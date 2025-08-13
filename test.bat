@echo off
REM distance-preservation-encoder 실험 배치 스크립트
REM uv 가상환경에서 4가지 loss function 실험을 순차적으로 실행

REM uv 환경 활성화 필요 없음(uv는 자체적으로 python 실행)
echo [1/4] DPLossV1
uv run python train.py --epochs 300 --loss-function DPLossV1

echo [2/4] DPLossV1Norm
uv run python train.py --epochs 300 --loss-function DPLossV1Norm

echo [3/4] DPLossV2
uv run python train.py --epochs 300 --loss-function DPLossV2

echo [4/4] DPLossV2Norm
uv run python train.py --epochs 300 --loss-function DPLossV2Norm

echo 모든 실험 완료!
pause