# DinoLink Project Workflow

## 1. Project Goal

Train a compact discrete representation pipeline for image patch tokens:

`image -> DINOv2 tokens -> Top-K selection -> projector -> quantizer -> token decoder`

The model optimizes token reconstruction quality while improving codebook usage and quantization efficiency.

---

## 2. End-to-End Data Flow

### 2.1 Forward Path (single step)

1. **Image input**  
   Load a batch from dataset (`nuscenes` / `waymo` / `coco`).

2. **Frozen DINOv2 feature extraction**  
   Extract patch tokens and attention scores.

3. **Token selection**  
   Use attention-based Top-K (`model.top_k`) to select informative patch tokens.

4. **Projection to quantization space**  
   `Projector` maps DINO hidden dimension (e.g. 768) to `quantizer.z_dim` (e.g. 32/64/256).

5. **Vector quantization**  
   Quantizer (`vq` / `rvq`) converts projected vectors into discrete code indices and quantized vectors.

6. **Token reconstruction**  
   `TokenDecoder` maps quantized vectors back to DINO token space.

7. **Loss + metrics**  
   Compute reconstruction loss, commit loss, and quantization effectiveness metrics.

### 2.2 Backward Path

- Update trainable modules:
  - `projector`
  - `quantizer` (depends on type)
  - `token_decoder`
- `extractor` (DINOv2) remains frozen.

---

## 3. Loss Design (Current)

In `losses/losses.py`, token reconstruction is a weighted combination:

- `token_l2_weight * L2`
- `token_logit_laplace_weight * Logit-Laplace`

Current intended paper-style default:

- `token_l2_weight = 1.0`
- `token_logit_laplace_weight = 0.1`

Commitment/codebook term:

- `vq` / `rvq`: codebook loss + beta * commitment loss

Total loss:

`loss = token_weight * token_loss + commit_loss`

---

## 4. Core Metrics Meaning

- `token_loss`: training token objective (weighted L2 + weighted Logit-Laplace)
- `token_mse_sel`: pure MSE on selected tokens (monitoring metric)
- `token_cos_sim`: cosine similarity between reconstructed and selected tokens
- `commit_loss`: quantizer-related commitment/codebook loss
- `quant_error`: element-wise MSE between `z_enc` and `z_q`
- `quant_error_sel`: per-token summed L2 in latent space (scale depends on `z_dim`)
- `quant_utilization`: active code ratio
- `quant_perplexity`: effective number of used codes
- `quant_active_codes`: number of codes used in current batch

---

## 5. Train Workflow

1. **Edit config**
   - File: `configs/config.yaml`
   - Key sections: `model`, `quantizer`, `decoder`, `data`, `run`

2. **Start training**
   - `python main.py --config configs/config.yaml --mode train --run_name <name>`

3. **Monitor logs**
   - `logs/<run_name>/metrics.txt`
   - focus on trends of:
     - `token_mse_sel` (down)
     - `token_cos_sim` (up)
     - `quant_error` (down)
     - `quant_utilization/perplexity` (not collapsed)

4. **Inspect visual outputs**
   - `attention_epoch*.png`
   - `selection_epoch*.png`
   - `token_matrices_epoch*.png`

5. **Checkpointing**
   - Saved at `logs/<run_name>/checkpoints/ckpt_epoch*.pt`

---

## 6. Test/Eval Workflow

Run test with a checkpoint:

- `python main.py --config configs/config.yaml --mode test --ckpt <path_to_ckpt>`

or by run name:

- `python main.py --config configs/config.yaml --mode test --run_name <name> --ckpt_epoch <N>`

Outputs:

- `logs/<run_name>/test_metrics.txt`
- `test_attention.png`, `test_selection.png`, `test_token_matrices.png`

---

## 7. Practical Tuning Order

If `token_mse_sel` is still high, tune in this order:

1. Increase `quantizer.z_dim` (e.g. 32 -> 64)
2. Increase `decoder.hidden_dim` (e.g. 512 -> 768)
3. Increase `quantizer.num_quantizers` (for `rvq`) if single-stage quantization is limiting
4. Train longer (at least 20k-30k steps before strong conclusions)
5. If needed, reduce `model.top_k` to lower reconstruction difficulty

---

## 8. Recommended Experiment Tracking

For each run, record:

- config snapshot (`config_used.yaml`)
- final and best `token_mse_sel`
- final and best `quant_error`
- `quant_utilization`, `quant_perplexity`, `quant_active_codes`
- representative `token_matrices` visualization

This makes ablation and regression checks much easier.
