import torch
import os
import json
import uuid
import argparse
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from wiki_dataset import WikiEmbeddingDataset
from model import DPEncoder
from loss import DPLossV1, DPLossV2, DPLossV1Norm, DPLossV2Norm

def knn_recall_at_k(x, z, k=10, batch_size=1000):
    """메모리 효율적인 KNN recall 계산 (배치 처리)"""
    n = x.shape[0]
    overlaps = []
    
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_x = x[i:end_i]
        batch_z = z[i:end_i]
        
        # 현재 배치에 대해 전체 데이터셋과의 거리 계산
        dx_batch = torch.cdist(batch_x, x)  # [batch_size, n]
        dz_batch = torch.cdist(batch_z, z)  # [batch_size, n]
        
        # k+1 nearest neighbors 찾기 (자기 자신 포함)
        _, idx_x_batch = dx_batch.topk(k+1, largest=False)
        _, idx_z_batch = dz_batch.topk(k+1, largest=False)
        
        # 자기 자신 제외 (첫 번째 요소는 항상 자기 자신)
        idx_x_batch = idx_x_batch[:, 1:].cpu().numpy()
        idx_z_batch = idx_z_batch[:, 1:].cpu().numpy()
        
        # 배치 내 각 샘플에 대해 overlap 계산
        for a, b in zip(idx_x_batch, idx_z_batch):
            overlaps.append(len(set(a) & set(b)) / k)
    
    return np.mean(overlaps)

def calculate_recall_during_training(model, dataset, device, k=5, num_samples=1000):
    """학습 중 recall 계산 (샘플링된 데이터로)"""
    model.eval()
    
    # 메모리 절약을 위해 샘플링
    np.random.seed(42)  # 일관된 샘플링을 위해
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    X = torch.stack([dataset[i] for i in indices]).to(device)
    
    with torch.no_grad():
        Z = model(X)
    
    recall = knn_recall_at_k(X, Z, k, batch_size=500)
    model.train()  # 다시 training 모드로
    return recall

def parse_args():
    parser = argparse.ArgumentParser(description="Train Distance Preservation Encoder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--loss-function", type=str, default="DPLossV2", 
                       choices=["DPLossV1", "DPLossV1Norm", "DPLossV2", "DPLossV2Norm"],
                       help="Loss function to use")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for soft ranking (V2 losses)")
    parser.add_argument("--lambda-rank", type=float, default=1.0, help="Weight for rank loss")
    parser.add_argument("--lambda-pairdist", type=float, default=0.3, help="Weight for pair distance loss")
    return parser.parse_args()

# Load environment variables
load_dotenv()

# Parse command line arguments
args = parse_args()

# Get device from environment with fallback logic
device_env = os.getenv('DEVICE', 'cpu')
if device_env == 'cuda' and not torch.cuda.is_available():
    print("Warning: CUDA requested but not available. Falling back to CPU.")
    device = torch.device('cpu')
elif device_env == 'mps' and not torch.backends.mps.is_available():
    print("Warning: MPS requested but not available. Falling back to CPU.")
    device = torch.device('cpu')
elif device_env == 'auto':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:
    device = torch.device(device_env)

print(f"Using device: {device}")
data_path = "wikipedia-22-12-ko-embeddings-100k.parquet"

# 학습 설정 (명령행 인수 우선, 환경변수 대체, 기본값 최후)
batch_size = args.batch_size
input_dim = 1024  # 임베딩 차원
hidden_dims = [768]
latent_dim = args.latent_dim
k = 5
lambda_rank = args.lambda_rank
lambda_pairdist = args.lambda_pairdist
lr = args.lr
epochs = args.epochs
save_interval = args.save_interval
loss_function_name = args.loss_function
tau = args.tau

# 환경변수에서 추가 설정 가져오기
save_interval = int(os.getenv('SAVE_INTERVAL', save_interval))

print(f"Training Configuration:")
print(f"  - Epochs: {epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {lr}")
print(f"  - Latent dimension: {latent_dim}")
print(f"  - Loss function: {loss_function_name}")
print(f"  - Save interval: every {save_interval} epochs")
print("Loading dataset...")
dataset = WikiEmbeddingDataset(data_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Dataset size: {len(dataset)}")
print("Dataset loaded")

# UUID 생성 및 체크포인트 디렉토리 생성
experiment_id = str(uuid.uuid4())
checkpoint_dir = os.path.join("checkpoints", experiment_id)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Experiment ID: {experiment_id}")
print(f"Checkpoint directory: {checkpoint_dir}")

# 실험 설정 저장
config = {
    "experiment_id": experiment_id,
    "timestamp": datetime.now().isoformat(),
    "batch_size": batch_size,
    "hidden_dims": hidden_dims,
    "latent_dim": latent_dim,
    "k": k,
    "lambda_rank": lambda_rank,
    "lambda_pairdist": lambda_pairdist,
    "lr": lr,
    "epochs": epochs,
    "save_interval": save_interval,
    "loss_function": loss_function_name,
    "tau": tau,
    "device": str(device),
    "data_path": data_path
}

with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)



# 실제 입력 차원 확인
actual_input_dim = dataset[0].shape[0]
print(f"Actual input dimension: {actual_input_dim}")

model = DPEncoder(actual_input_dim, hidden_dims, latent_dim).to(device)

# Loss 함수 동적 선택
loss_classes = {
    "DPLossV1": DPLossV1,
    "DPLossV1Norm": DPLossV1Norm,
    "DPLossV2": DPLossV2,
    "DPLossV2Norm": DPLossV2Norm
}

loss_class = loss_classes[loss_function_name]

# V2 계열 loss는 tau 파라미터가 필요
if "V2" in loss_function_name:
    criterion = loss_class(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist, tau=tau).to(device)
else:
    criterion = loss_class(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist).to(device)

print(f"Using loss function: {loss_function_name}")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 손실값 기록을 위한 리스트
train_history = {
    "epochs": [],
    "total_loss": [],
    "rank_loss": [],
    "pairdist_loss": [],
    "epoch_times": [],
    "epoch_timestamps": [],
    "cumulative_time": [],
    "training_start_time": datetime.now().isoformat(),
    "batch_times": [],
    "recall_history": [],  # 에포크별 recall 값
    "recall_epochs": []    # recall이 계산된 에포크들
}

# 전체 학습 시작 시간
training_start_time = time.time()
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

for epoch in range(epochs):
    epoch_start_time = time.time()
    epoch_timestamp = datetime.now()
    
    model.train()
    total_loss, total_rank, total_pair = 0, 0, 0
    n = 0
    batch_times = []
    
    for batch_idx, batch in enumerate(loader):
        batch_start_time = time.time()
        
        x = batch.to(device)
        z = model(x)
        loss, rank_loss, pairdist_loss = criterion(x, z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_times.append(batch_time)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rank += rank_loss.item() * bs
        total_pair += (pairdist_loss.item() if isinstance(pairdist_loss, torch.Tensor) else pairdist_loss) * bs
        n += bs
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    cumulative_time = epoch_end_time - training_start_time
    
    # 에포크별 평균 손실 계산
    avg_total_loss = total_loss / n
    avg_rank_loss = total_rank / n
    avg_pair_loss = total_pair / n
    
    # 배치 시간 통계
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    # ETA 계산
    if epoch > 0:
        avg_epoch_time = cumulative_time / (epoch + 1)
        eta_seconds = avg_epoch_time * (epochs - epoch - 1)
        eta = str(timedelta(seconds=int(eta_seconds)))
    else:
        eta = "Calculating..."
    
    print(f"[Epoch {epoch+1:03d}/{epochs}] Total: {avg_total_loss:.4f} | Rank: {avg_rank_loss:.4f} | PairDist: {avg_pair_loss:.4f}")
    print(f"  ├─ Epoch Time: {epoch_duration:.2f}s | Avg Batch: {avg_batch_time:.3f}s | ETA: {eta}")
    print(f"  └─ Cumulative: {str(timedelta(seconds=int(cumulative_time)))} | Timestamp: {epoch_timestamp.strftime('%H:%M:%S')}")
    
    # 손실값 및 시간 기록 (매 에포크마다)
    train_history["epochs"].append(epoch + 1)
    train_history["total_loss"].append(avg_total_loss)
    train_history["rank_loss"].append(avg_rank_loss)
    train_history["pairdist_loss"].append(avg_pair_loss)
    train_history["epoch_times"].append(epoch_duration)
    train_history["epoch_timestamps"].append(epoch_timestamp.isoformat())
    train_history["cumulative_time"].append(cumulative_time)
    train_history["batch_times"].append({
        "epoch": epoch + 1,
        "avg_batch_time": avg_batch_time,
        "total_batches": len(batch_times),
        "batch_times": batch_times
    })
    
    # 체크포인트 저장 (지정된 간격 또는 마지막 에포크)
    should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs
    
    if should_save:
        save_start_time = time.time()
        print(f"  ├─ Saving checkpoint at epoch {epoch+1}...")
        
        # KNN Recall 계산
        print(f"  ├─ Calculating KNN Recall@{k}...")
        recall_start_time = time.time()
        current_recall = calculate_recall_during_training(model, dataset, device, k=k)
        recall_duration = time.time() - recall_start_time
        print(f"  ├─ Recall@{k}: {current_recall:.4f} (calculated in {recall_duration:.2f}s)")
        
        # recall history에 추가
        train_history["recall_history"].append(current_recall)
        train_history["recall_epochs"].append(epoch + 1)
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history": train_history,
            "config": config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        torch.save(checkpoint, checkpoint_path)
        save_duration = time.time() - save_start_time
        print(f"  └─ Checkpoint saved in {save_duration:.2f}s")
    
    # 최신 모델은 매 에포크마다 업데이트
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "latest_model.pth"))
    print()  # 에포크 간 구분을 위한 빈 줄

# 전체 학습 완료 시간
training_end_time = time.time()
total_training_time = training_end_time - training_start_time

# 최종 학습 히스토리에 전체 시간 정보 추가
train_history["training_end_time"] = datetime.now().isoformat()
train_history["total_training_time_seconds"] = total_training_time
train_history["total_training_time_formatted"] = str(timedelta(seconds=int(total_training_time)))

# 시간 통계 계산
total_epochs = len(train_history["epoch_times"])
avg_epoch_time = sum(train_history["epoch_times"]) / total_epochs
min_epoch_time = min(train_history["epoch_times"])
max_epoch_time = max(train_history["epoch_times"])

train_history["time_statistics"] = {
    "average_epoch_time": avg_epoch_time,
    "min_epoch_time": min_epoch_time,
    "max_epoch_time": max_epoch_time,
    "total_epochs": total_epochs
}

# 최종 학습 히스토리 저장
with open(os.path.join(checkpoint_dir, "train_history.json"), "w") as f:
    json.dump(train_history, f, indent=2)

print("="*80)
print(f"🎉 Training completed! 🎉")
print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
print(f"Average time per epoch: {avg_epoch_time:.2f}s")
print(f"Fastest epoch: {min_epoch_time:.2f}s | Slowest epoch: {max_epoch_time:.2f}s")
print(f"All checkpoints saved in: {checkpoint_dir}")
print(f"Final model saved as: {os.path.join(checkpoint_dir, 'latest_model.pth')}")
print("="*80)

# 기존 파일명으로도 저장 (호환성을 위해)
torch.save(model.state_dict(), "dpae.pth")