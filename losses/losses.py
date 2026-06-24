import torch
import torch.nn.functional as F


def _logit_laplace_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    logit_laplace_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Logit-Laplace negative log-likelihood:
      L = |logit(x) - mu| / b + log(2b) - log(x(1-x))
    where mu=pred_logits, b=scale, and x in (0, 1).
    """
    b = float(logit_laplace_scale)
    if b <= 0:
        raise ValueError(f"logit_laplace_scale must be > 0, got {b}")

    # DINO token targets are not guaranteed to be in [0, 1].
    # Use sigmoid reparameterization for a stable logit-space objective.
    target_01 = torch.sigmoid(target)
    target_01 = target_01.clamp(min=eps, max=1.0 - eps)

    target_logit = torch.logit(target_01, eps=eps)
    abs_term = (target_logit - pred_logits).abs() / b
    const_term = torch.log(pred_logits.new_tensor(2.0 * b))
    jacobian_term = -torch.log(target_01 * (1.0 - target_01))
    return (abs_term + const_term + jacobian_term).mean()


def compute_losses(
    decoded_tokens: torch.Tensor,
    selected: torch.Tensor,
    quantizer: torch.nn.Module,
    z_enc: torch.Tensor,
    indices: torch.Tensor,
    z_dim: int,
    cfg: dict,
    quant_stats: dict | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """
    Compute all loss components and return (losses_dict, total_loss, extras).

    losses_dict keys:
      - token
      - commit

    extras keys:
      - z_enc_flat
      - z_q_raw
      - token_mse_per_selected
    """
    # 1) Token reconstruction loss:
    #    paper-style combination: L2 + weighted Logit-Laplace
    token_l2_loss = F.mse_loss(decoded_tokens, selected.detach())
    token_logit_laplace_loss = _logit_laplace_loss(
        pred_logits=decoded_tokens,
        target=selected.detach(),
        logit_laplace_scale=cfg["run"].get("logit_laplace_scale", 1.0),
        eps=cfg["run"].get("logit_laplace_eps", 1e-6),
    )
    token_l2_weight = cfg["run"].get("token_l2_weight", 1.0)
    token_logit_laplace_weight = cfg["run"].get("token_logit_laplace_weight", 0.1)
    token_loss = (
        token_l2_weight * token_l2_loss
        + token_logit_laplace_weight * token_logit_laplace_loss
    )
    token_mse_per_selected = (decoded_tokens - selected.detach()).pow(2).mean(dim=-1)

    # 2) Commitment loss
    beta = cfg["run"].get("beta", 0.25)
    if quant_stats is not None and "z_enc" in quant_stats and "z_q_raw" in quant_stats:
        z_enc_flat = quant_stats["z_enc"]
        z_q_raw = quant_stats["z_q_raw"]
    else:
        z_enc_flat = z_enc.reshape(-1, z_dim)
        z_q_raw = quantizer.decode(indices)  # raw codebook vectors (fallback)
    
    # --- debug monitoring start ---
    with torch.no_grad():
        z_enc_max = z_enc_flat.abs().max().item()
        z_q_max = z_q_raw.abs().max().item()
        if z_enc_max > 10 or z_q_max > 10:
            print(f"!!! Numerical anomaly warning !!!")
            print(f"z_enc_flat -> max: {z_enc_max:.4f}, mean: {z_enc_flat.abs().mean():.4f}, shape: {z_enc_flat.shape}")
            print(f"z_q_raw    -> max: {z_q_max:.4f}, mean: {z_q_raw.abs().mean():.4f}, shape: {z_q_raw.shape}")
        
        # Dimension alignment check (most important!)
        if z_enc_flat.shape != z_q_raw.shape:
             # If this triggers, it means the reshape logic caused the MSE to compute the wrong mean
             z_q_raw = z_q_raw.reshape(z_enc_flat.shape)
             print(f"Forced z_q_raw dimensions to align to: {z_q_raw.shape}")


    # Classic VQ-VAE loss (used by both VQ and RVQ):
    #   ||sg[z_e] - e||^2            (updates codebook)
    # + beta * ||z_e - sg[e]||^2     (commitment for encoder)
    codebook_loss = F.mse_loss(z_q_raw, z_enc_flat.detach())
    commitment_loss = beta * F.mse_loss(z_enc_flat, z_q_raw.detach())
    commit_loss = codebook_loss + commitment_loss

    losses = {
        "token": token_loss,
        "commit": commit_loss,
    }
    token_weight = cfg["run"].get("token_weight", 1.0)
    total_loss = (
        token_weight * losses["token"]
        + losses["commit"]
    )

    extras = {
        "z_enc_flat": z_enc_flat,
        "z_q_raw": z_q_raw,
        "token_mse_per_selected": token_mse_per_selected,
    }
    return losses, total_loss, extras

