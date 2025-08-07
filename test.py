import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wiki_dataset import WikiEmbeddingDataset
from model import DPEncoder 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 

def knn_recall_at_k(x, z, k=10, batch_size=1000):
    """
    메모리 효율적인 KNN recall 계산 (배치 처리)
    """
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

def plot_tsne(x, z, num_samples=300, random_seed=42):
    np.random.seed(random_seed)
    idx = np.random.choice(x.shape[0], min(num_samples, x.shape[0]), replace=False)
    x_np = x[idx].cpu().numpy()
    z_np = z[idx].cpu().numpy()
    tsne = TSNE(n_components=2, random_state=random_seed, init="pca", perplexity=30)
    x_2d = tsne.fit_transform(x_np)
    z_2d = tsne.fit_transform(z_np)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x_2d[:, 0], x_2d[:, 1], alpha=0.6)
    axes[0].set_title("Original Embedding (x)")
    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6)
    axes[1].set_title("Latent Embedding (z)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = "wikipedia-22-12-ko-embeddings-100k.parquet"
    model_path = "dpae.pth"
    
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
    hidden_dims = [768]
    latent_dim = 512
    k = 5
    
    # 메모리 절약을 위해 샘플 수 제한
    num_test_samples = 10000  # 전체 데이터셋 대신 10k 샘플만 사용
    print(f"Testing with {num_test_samples} samples for memory efficiency")

    # 데이터셋 로드
    dataset = WikiEmbeddingDataset(data_path)
    
    # 랜덤하게 샘플 선택
    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_test_samples, replace=False)
    X = torch.stack([dataset[i] for i in indices]).to(device)

    # 모델 로드
    input_dim = X.shape[1]
    model = DPEncoder(input_dim, hidden_dims, latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        Z = model(X)

    recall = knn_recall_at_k(X, Z, k, batch_size=500)  # 배치 크기도 줄임
    print(f"KNN Recall@{k}: {recall:.4f}")

    #plot_tsne(X.cpu(), Z.cpu(), num_samples=300)
