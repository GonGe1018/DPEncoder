from model import DPEncoder
import torch

def how_to_use_dpae():

    input_dim = 1536
    hidden_dims = [768]
    latent_dim = 512
    model_path = "/example/dpae.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_size = 1000

    model = DPEncoder(input_dim, hidden_dims, latent_dim)
    model.load_state_dict(torch.load("dpae.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Example input
    
    X = torch.stack([torch.randn(data_size, input_dim) for _ in range(len(1000))]).to(device)

    with torch.no_grad():
        Z = model(X)
