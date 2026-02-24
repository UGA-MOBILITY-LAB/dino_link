"""
Token-level decoder: indices -> dequantized VQ token -> DINO token.
"""
import torch
import torch.nn as nn


class TokenDecoder(nn.Module):
    """
    Decode quantizer indices back to DINO token space.

    Optional 2D patch position (row, col in [-1, 1]) can be concatenated to
    provide lightweight spatial priors for token reconstruction.
    """

    def __init__(
        self,
        z_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        use_pos: bool = True,
    ):
        super().__init__()
        self.use_pos = use_pos
        in_dim = z_dim + 2 if use_pos else z_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        z_q_st: torch.Tensor,
        patch_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_q_st: (B, k, z_dim), straight-through quantized tokens
            patch_pos: (B, k, 2), normalized to [-1, 1]
        Returns:
            decoded_tokens: (B, k, out_dim)
        """
        z_q = z_q_st

        if self.use_pos:
            if patch_pos is None:
                raise ValueError("TokenDecoder(use_pos=True) requires patch_pos.")
            inp = torch.cat([z_q, patch_pos.to(z_q.dtype)], dim=-1)
        else:
            inp = z_q
        out = self.net(inp)
        return self.norm(out)
