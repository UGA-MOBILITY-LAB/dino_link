"""
VQ / RVQ quantizer wrappers.

forward(z) returns:
  - z_q_st: straight-through quantized vectors (N, z_dim)
  - indices: discrete indices (N,)
  - stats: dict with z_enc/z_q_raw/quant_error for logging/loss reuse
"""
import torch
import torch.nn as nn


def _run_kmeans(
    points: torch.Tensor,
    codebook_size: int,
    z_dim: int,
    num_iters: int,
) -> tuple[torch.Tensor, int]:
    """Run simple K-means and return (centers, active_count)."""
    n = points.shape[0]
    if n >= codebook_size:
        init_idx = torch.randperm(n, device=points.device)[:codebook_size]
        centers = points[init_idx].clone()
    else:
        repeats = (codebook_size + n - 1) // n
        centers = points.repeat(repeats, 1)[:codebook_size].clone()
        centers = centers + 0.01 * torch.randn_like(centers)

    iters = max(int(num_iters), 1)
    active_count = 0
    for _ in range(iters):
        assign = find_nearest_indices(points, centers)
        counts = torch.bincount(
            assign, minlength=codebook_size
        ).to(dtype=points.dtype, device=points.device)
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(0, assign.unsqueeze(1).expand(-1, z_dim), points)
        non_empty = counts > 0
        if non_empty.any():
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(1)
        if (~non_empty).any():
            new_centers[~non_empty] = centers[~non_empty]
        centers = new_centers
        active_count = int(non_empty.sum().item())
    return centers, active_count


def vq_effectiveness(
    indices: torch.Tensor,
    z_enc: torch.Tensor,
    z_q: torch.Tensor,
    codebook_size: int,
    eps: float = 1e-10,
) -> dict[str, float]:
    """
    Compute quantization effectiveness metrics.
    indices: (N,) long, z_enc/z_q: (N, z_dim)
    Returns dict with:
      - utilization: fraction of codebook entries used (active_codes / codebook_size)
      - perplexity: exp(entropy); effective number of codes used (higher = better usage)
      - quant_error: MSE(z_enc, z_q) — reconstruction error of quantization
    """
    n = indices.numel()
    unique = indices.unique()
    active_codes = unique.numel()
    utilization = active_codes / max(codebook_size, 1)

    _, counts = indices.view(-1).unique(return_counts=True)
    prob = counts.to(torch.float32) / (counts.sum().to(torch.float32) + eps)
    prob = prob.clamp(min=eps)
    entropy = -(prob * prob.log()).sum().item()
    perplexity = float(torch.exp(torch.tensor(entropy)).item())

    quant_error = torch.nn.functional.mse_loss(z_enc, z_q).item()
    return {
        "quant_utilization": utilization,
        "quant_perplexity": perplexity,
        "quant_error": quant_error,
        "quant_active_codes": float(active_codes),
    }


def find_nearest_indices(query: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Returns (N,) indices into codebook."""
    dist = (query.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(dim=2)
    return dist.argmin(dim=1)


class VQ(nn.Module):
    """
    Standard VQ quantizer.
    forward returns z_q_st + indices + stats.
    """

    def __init__(self, codebook_size: int, z_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.z_dim = z_dim
        self.embedding = nn.Embedding(codebook_size, z_dim)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        z: (N, z_dim)
        Returns:
            z_q_st: (N, z_dim)
            indices: (N,) long
            stats: dict with z_enc / z_q_raw / quant_error
        """
        indices = find_nearest_indices(z, self.embedding.weight)
        z_q_raw = self.embedding(indices)
        z_q_st = z + (z_q_raw - z).detach()
        quant_error = (z - z_q_raw).pow(2).mean()
        stats = {
            "z_enc": z,
            "z_q_raw": z_q_raw,
            "quant_error": quant_error,
        }
        return z_q_st, indices, stats

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (N,) -> (N, z_dim)"""
        return self.embedding(indices)

    def codebook_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight]

    @torch.no_grad()
    def kmeans_update_(
        self,
        z: torch.Tensor,
        num_iters: int = 5,
        sample_size: int = 0,
    ) -> int:
        """
        Periodically refresh codebook by K-means on encoder outputs.

        Args:
            z: (N, z_dim) encoder outputs.
            num_iters: number of K-means refinement iterations.
            sample_size: if >0 and N is larger, randomly subsample before K-means.

        Returns:
            Number of active clusters in the final K-means iteration.
        """
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z shape (N, {self.z_dim}), got {tuple(z.shape)}")

        points = z.detach()
        if sample_size > 0 and points.shape[0] > sample_size:
            sel = torch.randperm(points.shape[0], device=points.device)[:sample_size]
            points = points[sel]
        n = points.shape[0]
        if n == 0:
            return 0

        centers, active_count = _run_kmeans(
            points=points,
            codebook_size=self.codebook_size,
            z_dim=self.z_dim,
            num_iters=num_iters,
        )
        self.embedding.weight.data.copy_(centers)
        return active_count


class RVQ(nn.Module):
    """
    Residual Vector Quantization (RVQ).
    Quantizes residuals stage by stage with independent codebooks.
    """

    def __init__(self, codebook_size: int, z_dim: int, num_quantizers: int = 2):
        super().__init__()
        if int(num_quantizers) < 1:
            raise ValueError(f"num_quantizers must be >= 1, got {num_quantizers}")
        self.codebook_size = int(codebook_size)
        self.z_dim = int(z_dim)
        self.num_quantizers = int(num_quantizers)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.z_dim) for _ in range(self.num_quantizers)]
        )
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1.0, 1.0)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        residual = z
        z_q_raw = torch.zeros_like(z)
        stage_indices = []
        for emb in self.embeddings:
            idx = find_nearest_indices(residual, emb.weight)
            q = emb(idx)
            z_q_raw = z_q_raw + q
            residual = residual - q
            stage_indices.append(idx)

        indices = torch.stack(stage_indices, dim=1)  # (N, num_quantizers)
        z_q_st = z + (z_q_raw - z).detach()
        quant_error = (z - z_q_raw).pow(2).mean()
        stats = {
            "z_enc": z,
            "z_q_raw": z_q_raw,
            "quant_error": quant_error,
        }
        return z_q_st, indices, stats

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            return self.embeddings[0](indices)
        if indices.dim() != 2 or indices.shape[1] != self.num_quantizers:
            raise ValueError(
                f"RVQ decode expects shape (N, {self.num_quantizers}), got {tuple(indices.shape)}"
            )
        z_q = torch.zeros(indices.shape[0], self.z_dim, device=indices.device)
        for i, emb in enumerate(self.embeddings):
            z_q = z_q + emb(indices[:, i])
        return z_q

    def codebook_parameters(self) -> list[nn.Parameter]:
        return [emb.weight for emb in self.embeddings]

    @torch.no_grad()
    def kmeans_update_(
        self,
        z: torch.Tensor,
        num_iters: int = 5,
        sample_size: int = 0,
    ) -> int:
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z shape (N, {self.z_dim}), got {tuple(z.shape)}")

        points = z.detach()
        if sample_size > 0 and points.shape[0] > sample_size:
            sel = torch.randperm(points.shape[0], device=points.device)[:sample_size]
            points = points[sel]
        if points.shape[0] == 0:
            return 0

        residual = points
        active_list = []
        for emb in self.embeddings:
            centers, active = _run_kmeans(
                points=residual,
                codebook_size=self.codebook_size,
                z_dim=self.z_dim,
                num_iters=num_iters,
            )
            emb.weight.data.copy_(centers)
            assign = find_nearest_indices(residual, centers)
            residual = residual - centers[assign]
            active_list.append(active)
        return int(sum(active_list) / max(len(active_list), 1))


