import torch.nn as nn
import torch
import torch.nn.functional as F

class DPLoss(nn.Module):
    def __init__(self, k=32, lambda_rank=1.0, lambda_pairdist=0.0):
        super().__init__()
        self.k = k
        self.lambda_rank = lambda_rank
        self.lambda_pairdist = lambda_pairdist

    def forward(self, x, z):
        # 순위 기반 손실
        dist_x = torch.cdist(x, x, p=2)
        dist_z = torch.cdist(z, z, p=2)
        rank_x = dist_x.argsort(dim=1).argsort(dim=1) # rank_matrix
        rank_z = dist_z.argsort(dim=1).argsort(dim=1) # rank_matrix
        rank_loss = (rank_x - rank_z).abs().float().mean() / self.k

        # 쌍별 거리값 차이 손실
        pairdist_loss = F.mse_loss(dist_z, dist_x) if self.lambda_pairdist > 0.0 else 0.0

        total = self.lambda_rank * rank_loss + self.lambda_pairdist * pairdist_loss
        return total, rank_loss, pairdist_loss
