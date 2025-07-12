import torch
from torch.utils.data import Dataset
import json

class WikiEmbeddingDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append({
                    "vector": torch.tensor(obj["embedding"], dtype=torch.float32),
                    "metadata": obj.get("title", None)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["vector"]