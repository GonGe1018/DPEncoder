import torch
from torch.utils.data import DataLoader
from test_dataset import WikiEmbeddingDataset
from model import DPEncoder
from loss import DPLoss, DPLossV2


device = torch.device("mps" if torch.cuda.is_available() else "cpu")
data_path = "wikipedia-korean-cohere-embeddings-100k.jsonl"


batch_size = 512 

dataset = WikiEmbeddingDataset(data_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

input_dim = dataset[0].shape[0]
hidden_dims = [768]
latent_dim = 512
k = 5
lambda_rank = 1.0
lambda_pairdist = 0.3
lr = 1e-3
epochs = 100

model = DPEncoder(input_dim, hidden_dims, latent_dim).to(device)
# criterion = DPLoss(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist).to(device)
criterion = DPLossV2(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    print(f"[Epoch {epoch+1:03d}] Total: {total_loss/n:.4f} | Rank: {total_rank/n:.4f} | PairDist: {total_pair/n:.4f}")

torch.save(model.state_dict(), "dpae.pth")