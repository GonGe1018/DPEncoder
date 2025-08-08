import torch
import os
import json
import uuid
import argparse
from datetime import datetime
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from wiki_dataset import WikiEmbeddingDataset
from model import DPEncoder
from loss import DPLossV1, DPLossV2, DPLossV1Norm, DPLossV2Norm

def parse_args():
    parser = argparse.ArgumentParser(description="Train Distance Preservation Encoder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=512, help="Latent dimension")
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
lambda_rank = 1.0
lambda_pairdist = 0.3
lr = args.lr
epochs = args.epochs
save_interval = args.save_interval

# 환경변수에서 추가 설정 가져오기
save_interval = int(os.getenv('SAVE_INTERVAL', save_interval))

print(f"Training Configuration:")
print(f"  - Epochs: {epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {lr}")
print(f"  - Latent dimension: {latent_dim}")
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
    "device": str(device),
    "data_path": data_path
}

with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)



# 실제 입력 차원 확인
actual_input_dim = dataset[0].shape[0]
print(f"Actual input dimension: {actual_input_dim}")

model = DPEncoder(actual_input_dim, hidden_dims, latent_dim).to(device)
criterion = DPLossV1Norm(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 손실값 기록을 위한 리스트
train_history = {
    "epochs": [],
    "total_loss": [],
    "rank_loss": [],
    "pairdist_loss": []
}

for epoch in range(epochs):
    model.train()
    total_loss, total_rank, total_pair = 0, 0, 0
    n = 0
    for batch in loader:
        x = batch.to(device)
        z = model(x)
        loss, rank_loss, pairdist_loss = criterion(x, z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rank += rank_loss.item() * bs
        total_pair += (pairdist_loss.item() if isinstance(pairdist_loss, torch.Tensor) else pairdist_loss) * bs
        n += bs
    
    # 에포크별 평균 손실 계산
    avg_total_loss = total_loss / n
    avg_rank_loss = total_rank / n
    avg_pair_loss = total_pair / n
    
    print(f"[Epoch {epoch+1:03d}] Total: {avg_total_loss:.4f} | Rank: {avg_rank_loss:.4f} | PairDist: {avg_pair_loss:.4f}")
    
    # 손실값 기록 (매 에포크마다)
    train_history["epochs"].append(epoch + 1)
    train_history["total_loss"].append(avg_total_loss)
    train_history["rank_loss"].append(avg_rank_loss)
    train_history["pairdist_loss"].append(avg_pair_loss)
    
    # 체크포인트 저장 (지정된 간격 또는 마지막 에포크)
    should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs
    
    if should_save:
        print(f"  └─ Saving checkpoint at epoch {epoch+1}")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history": train_history,
            "config": config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        torch.save(checkpoint, checkpoint_path)
    
    # 최신 모델은 매 에포크마다 업데이트
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "latest_model.pth"))

# 최종 학습 히스토리 저장
with open(os.path.join(checkpoint_dir, "train_history.json"), "w") as f:
    json.dump(train_history, f, indent=2)

print(f"\nTraining completed! All checkpoints saved in: {checkpoint_dir}")
print(f"Final model saved as: {os.path.join(checkpoint_dir, 'latest_model.pth')}")

# 기존 파일명으로도 저장 (호환성을 위해)
torch.save(model.state_dict(), "dpae.pth")