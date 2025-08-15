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
    model_path = "checkpoints/d8580568-37d4-467f-bc68-ee0cff523154/latest_model.pth"
    
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
    latent_dim = 512  # Match the trained model
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

    print("Model test completed successfully!")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Z.shape}")

    # Optional: Uncomment to visualize with t-SNE
    plot_tsne(X.cpu(), Z.cpu(), num_samples=300)
