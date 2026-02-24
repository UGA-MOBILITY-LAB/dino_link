"""
Projector: DINOv2 hidden_size -> z_dim with LayerNorm for stable VQ.
"""
import torch.nn as nn


class Projector(nn.Module):
    """
    Linear(hidden_size, z_dim) + LayerNorm(z_dim).
    LayerNorm 稳定 z_enc 尺度，避免 commit_loss 爆炸。
    """

    def __init__(self, hidden_size: int, z_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, z_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # self.norm = nn.LayerNorm(z_dim)
        self.norm = nn.LayerNorm(z_dim, elementwise_affine=False)
        # self.norm = nn.LayerNorm(z_dim, elementwise_affine=True)

    def forward(self, x):
        return self.norm(self.linear(x))
