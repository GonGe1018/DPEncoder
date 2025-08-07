import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class WikiEmbeddingDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        
        if path.endswith('.parquet'):
            # Read parquet file
            df = pd.read_parquet(path)
            import json
            for _, row in df.iterrows():
                # Check for common embedding column names
                embedding = None
                if 'embedding' in df.columns:
                    embedding = row['embedding']
                elif 'vector' in df.columns:
                    embedding = row['vector']
                elif 'embedding_json' in df.columns:
                    # Handle JSON string embeddings
                    embedding_json = row['embedding_json']
                    if isinstance(embedding_json, str):
                        embedding = json.loads(embedding_json)
                    else:
                        embedding = embedding_json
                else:
                    # Find the first column that looks like an embedding
                    for col in df.columns:
                        col_data = row[col]
                        if isinstance(col_data, (list, np.ndarray)):
                            embedding = col_data
                            break
                        elif isinstance(col_data, str) and col_data.startswith('[') and col_data.endswith(']'):
                            # Try to parse as JSON array
                            try:
                                embedding = json.loads(col_data)
                                break
                            except json.JSONDecodeError:
                                continue
                    else:
                        raise ValueError("No embedding column found in parquet file")
                
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                self.samples.append({
                    "vector": torch.tensor(embedding, dtype=torch.float32),
                    "metadata": row.get("title", None) if "title" in df.columns else None
                })
        else:
            # Read JSONL file (original code)
            import json
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