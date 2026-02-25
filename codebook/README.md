# Phase2 Codebook Training

This stage learns a discrete VQ-style codebook (`V=512` by default) that maps Phase1 latent vectors (`D=1536`) to discrete token IDs (`z_ids`).

## Goal

Input rows contain:

- `qid: string`
- `question: string`
- `answer_int: int32`
- `answer_digits: list<int32>`
- `K_star: int32`
- `k_max: int32`
- `latent_vectors: list<list<float32>>` with shape `[K_star, 1536]`

Output rows keep all fields except:

- remove `latent_vectors`
- add `z_ids: list<int32>` with length `K_star`

## Model

`Codebook` stores:

- `embeddings: [V, D]`
- `ema_cluster_size: [V]`
- `ema_embedding_sum: [V, D]`

For each batch of flattened latents `x in R^{N x D}`:

1. L2-normalize `x` and `E`.
2. Compute cosine similarity `S = x_norm @ E_norm^T`.
3. Assign `z = argmax(S, dim=-1)`.
4. Quantize with `q = E[z]`.

## Losses

Let `x` be latent vectors and `q` be quantized vectors:

- Codebook loss: `||sg(x) - q||^2`
- Commitment loss: `beta * ||x - sg(q)||^2`, with `beta=0.25`

Usage regularization:

1. Build batch histogram from `z` with Laplace smoothing `alpha=1.0`.
2. Compute `p_batch`.
3. Update EMA usage distribution: `p_ema <- eta * p_ema + (1-eta) * p_batch`, `eta=0.99`.
4. KL term: `KL(p_ema || Uniform(V))`.
5. Weighted KL: `lambda_kl * KL`, with `lambda_kl=0.01`.

Total:

`total_loss = vq_loss + commit_loss + kl_loss`

## Why KL usage regularization is needed

Pure nearest-neighbor VQ can collapse into a small subset of codes. KL-to-uniform adds pressure toward broader code usage, improving effective vocabulary and reducing dead codes.

## EMA update

After each batch:

- `ema_cluster_size <- decay * ema_cluster_size + (1-decay) * counts`
- `ema_embedding_sum <- decay * ema_embedding_sum + (1-decay) * sum(latents per code)`
- `embeddings <- normalize(ema_embedding_sum / (ema_cluster_size + eps))`

Embeddings are L2-normalized after EMA updates so cosine assignment and quantized vectors stay consistent.

Defaults:

- `ema_decay=0.995`
- `eps=1e-5`

## Expected behavior

Healthy runs usually show:

- perplexity rising above `100`
- dead fraction `< 0.5` early
- dead fraction trending toward `< 0.2` later

(`effective_vocab` is logged as `round(perplexity)`.)

## Training

```bash
python -m codebook.train \
  --input_dir /path/to/phase1_latent_shards \
  --output_dir /path/to/codebook_out
```

Outputs:

- `output_dir/codebook.pt`
- `output_dir/meta.json`

`meta.json` includes:

- `dim`
- `vocab_size`
- `ema_decay`
- `beta`
- `lambda_kl`
- `normalization="cosine"`

## Export Dataset

Rewrite Phase1 parquet shards by replacing `latent_vectors` with `z_ids`:

```bash
python -m codebook.export_dataset \
  --input_dir /path/to/phase1_latent_shards \
  --output_dir /path/to/phase2_z_shards \
  --codebook_path /path/to/codebook_out/codebook.pt
```

Strict mode is default (fail on invalid rows). To skip bad rows during long exports:

```bash
python -m codebook.export_dataset ... --skip_invalid_rows
```

The exporter streams shard-by-shard and does not load the full dataset in RAM.
