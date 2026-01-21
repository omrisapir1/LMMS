import torch
from typing import Optional


def collect_latents_for_kmeans(
    train_ds,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Collect all latent vectors h for k-means clustering.

    Reads directly from the underlying HF dataset (train_ds.ds),
    NOT via __getitem__, to avoid tokenization / padding overhead.

    Returns:
        X: Tensor[N, H] in float32 (default), on CPU unless device is specified.

    Hard-fails on malformed data.
    """
    hf_ds = getattr(train_ds, "ds", None)
    if hf_ds is None:
        raise RuntimeError("train_ds has no underlying HF dataset (.ds)")

    latents = []
    H: Optional[int] = None
    total = 0

    for i, ex in enumerate(hf_ds):
        latent_states = ex.get("latent_states", None)
        K = ex.get("num_latents", None)

        if latent_states is None or K is None:
            raise RuntimeError(f"Row {i} missing latent_states or num_latents")

        if not isinstance(latent_states, (list, tuple)):
            raise RuntimeError(f"Row {i}: latent_states must be list-like")

        if K <= 0 or K > len(latent_states):
            raise RuntimeError(f"Row {i}: invalid num_latents={K}")

        # Take only active latent states
        for j in range(K):
            h = latent_states[j]

            if not isinstance(h, (list, tuple)):
                raise RuntimeError(f"Row {i}, latent {j}: expected list[float]")

            if H is None:
                H = len(h)
                if H <= 0:
                    raise RuntimeError("Latent dimension H must be > 0")
            elif len(h) != H:
                raise RuntimeError(
                    f"Row {i}, latent {j}: inconsistent H={len(h)} (expected {H})"
                )

            latents.append(h)
            total += 1

    if total == 0:
        raise RuntimeError("No latent vectors collected for k-means")

    # Convert to tensor
    X = torch.tensor(latents, dtype=dtype)

    if device is not None:
        X = X.to(device)

    return X


@torch.no_grad()
def kmeans_pp_deterministic(
    X: torch.Tensor,          # [N, H]
    V: int,                   # number of clusters
    n_iters: int,
    seed: int,
) -> torch.Tensor:
    """
    Deterministic k-means++ clustering.

    Args:
        X: Tensor[N, H] (float32 recommended, CPU preferred)
        V: number of clusters
        n_iters: Lloyd iterations
        seed: random seed (controls k-means++ init)

    Returns:
        centroids: Tensor[V, H]
    """
    if X.ndim != 2:
        raise ValueError("X must be [N, H]")
    if V <= 0:
        raise ValueError("V must be > 0")
    if X.size(0) < V:
        raise ValueError("N must be >= V")

    N, H = X.shape
    device = X.device
    dtype = X.dtype

    # ------------------------------------------------------------
    # Deterministic RNG
    # ------------------------------------------------------------
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # ------------------------------------------------------------
    # k-means++ initialization
    # ------------------------------------------------------------
    centroids = torch.empty((V, H), device=device, dtype=dtype)

    # 1) First centroid: uniform random
    first_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
    centroids[0] = X[first_idx]

    # Squared distances to nearest centroid
    d2 = torch.sum((X - centroids[0]) ** 2, dim=1)  # [N]

    for k in range(1, V):
        print(f"k-means++ init: selecting centroid {k+1} / {V}")

         # 2) Subsequent centroids: weighted random by d^2
        if torch.all(d2 == 0):
            # Degenerate case: all points identical
            centroids[k:] = centroids[0]
            break

        probs = d2 / d2.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centroids[k] = X[idx]

        # Update distance-to-nearest-centroid
        new_d2 = torch.sum((X - centroids[k]) ** 2, dim=1)
        d2 = torch.minimum(d2, new_d2)

    # ------------------------------------------------------------
    # Lloyd iterations
    # ------------------------------------------------------------
    for it in range(n_iters):
        print(f"k-means Lloyd iteration {it+1} / {n_iters}")
        # Assign points to nearest centroid
        # distances: [N, V]
        assignments = balanced_assign(X, centroids, topk=3)

        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(V, device=device, dtype=torch.long)

        new_centroids.index_add_(0, assignments, X)
        counts.index_add_(0, assignments, torch.ones_like(assignments))

        # Handle empty clusters deterministically
        for k in range(V):
            if counts[k] == 0:
                # Re-seed with farthest point
                farthest_idx = torch.argmax(d2).item()
                new_centroids[k] = X[farthest_idx]
            else:
                new_centroids[k] /= counts[k]

        # Convergence check (optional but safe)
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break

        centroids = new_centroids

        # Update d2 for empty-cluster handling
        d2 = torch.min(torch.cdist(X, centroids) ** 2, dim=1).values

    return centroids


@torch.no_grad()
def balanced_assign(
    X: torch.Tensor,          # [N, H]
    centroids: torch.Tensor,  # [V, H]
    topk: int = 3,
):
    """
    Capacity-constrained assignment:
    assigns exactly N//V points to each centroid.

    Deterministic, scalable, greedy.

    Returns:
        assignments: LongTensor [N] in [0, V-1]
    """
    assert X.ndim == 2 and centroids.ndim == 2
    assert X.size(1) == centroids.size(1)

    device = X.device
    N = X.size(0)
    V = centroids.size(0)

    capacity = N // V
    if capacity * V != N:
        raise ValueError("N must be divisible by V for exact balancing")

    # ------------------------------------------------------------
    # Compute distances (dominant cost)
    # ------------------------------------------------------------
    distances = torch.cdist(X, centroids)  # [N, V]

    # ------------------------------------------------------------
    # Top-k nearest centroids per point
    # ------------------------------------------------------------
    dists_k, idx_k = torch.topk(
        distances, k=topk, dim=1, largest=False
    )  # [N, topk]

    # ------------------------------------------------------------
    # Greedy assignment with capacity
    # ------------------------------------------------------------
    assignments = torch.full((N,), -1, device=device, dtype=torch.long)
    counts = torch.zeros(V, device=device, dtype=torch.long)

    # Process points in order of increasing best distance
    order = torch.argsort(dists_k[:, 0])

    for i in order.tolist():
        for j in idx_k[i].tolist():
            if counts[j] < capacity:
                assignments[i] = j
                counts[j] += 1
                break

    # ------------------------------------------------------------
    # Deterministic repair (rare)
    # ------------------------------------------------------------
    unassigned = (assignments == -1).nonzero(as_tuple=True)[0]
    if unassigned.numel() > 0:
        # find centroids with remaining capacity
        free = (counts < capacity).nonzero(as_tuple=True)[0]
        assert free.numel() >= unassigned.numel()

        # assign leftover points deterministically
        for i, j in zip(unassigned.tolist(), free.tolist()):
            assignments[i] = j
            counts[j] += 1

    # Final sanity check
    if not torch.all(counts == capacity):
        raise RuntimeError("Balanced assignment failed")

    return assignments
