import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wiki_dataset import WikiEmbeddingDataset
from model import DPEncoder 

def knn_recall_at_k(x, z, k=10):
    dx = torch.cdist(x, x)
    dz = torch.cdist(z, z)
    _, idx_x = dx.topk(k+1, largest=False)
    _, idx_z = dz.topk(k+1, largest=False)
    idx_x = idx_x[:, 1:].cpu().numpy()
    idx_z = idx_z[:, 1:].cpu().numpy()
    overlap = []
    for a, b in zip(idx_x, idx_z):
        overlap.append(len(set(a) & set(b)) / k)
    return np.mean(overlap)

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
    data_path = "wikipedia-korean-cohere-embeddings-100k.jsonl"
    model_path = "dpae.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [768]
    latent_dim = 512
    k = 5

    # 데이터셋 로드
    dataset = WikiEmbeddingDataset(data_path)
    X = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)

    # 모델 로드
    input_dim = X.shape[1]
    model = DPEncoder(input_dim, hidden_dims, latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        Z = model(X)

    recall = knn_recall_at_k(X, Z, k)
    print(f"KNN Recall@{k}: {recall:.4f}")

    #plot_tsne(X.cpu(), Z.cpu(), num_samples=300)
