import torch
from typing import Optional

def collect_latents_for_kmeans(
    train_ds,
    *,
    device=None,
    dtype=torch.float32,
):
    hf_ds = getattr(train_ds, "ds", None)
    if hf_ds is None:
        raise RuntimeError("train_ds has no underlying HF dataset (.ds)")

    latents = []
    row_ptr = []
    offset = 0
    H = None

    for i, ex in enumerate(hf_ds):
        latent_states = ex["latent_states"]
        K = ex["num_latents"]
        if K<20:
            continue

        if K <= 0:
            continue

        if H is None:
            H = len(latent_states[0])

        start = offset
        for j in range(K):
            latents.append(latent_states[j])
            offset += 1
        end = offset

        row_ptr.append((start, end))

    if not latents:
        raise RuntimeError("No latents collected")

    X = torch.tensor(latents, dtype=dtype)
    if device is not None:
        X = X.to(device)

    return X, row_ptr



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
    Capacity-constrained assignment with remainder handling.
    Guaranteed to terminate.

    Cluster sizes differ by at most 1.
    Deterministic, scalable.
    """
    assert X.ndim == 2 and centroids.ndim == 2
    assert X.size(1) == centroids.size(1)

    device = X.device
    N = X.size(0)
    V = centroids.size(0)

    # -------------------------
    # Capacities
    # -------------------------
    base = N // V
    remainder = N % V

    capacities = torch.full((V,), base, device=device, dtype=torch.long)
    if remainder > 0:
        capacities[:remainder] += 1

    # -------------------------
    # Distances
    # -------------------------
    distances = torch.cdist(X, centroids)  # [N, V]

    # -------------------------
    # Top-k candidates
    # -------------------------
    dists_k, idx_k = torch.topk(
        distances, k=topk, dim=1, largest=False
    )

    assignments = torch.full((N,), -1, device=device, dtype=torch.long)
    counts = torch.zeros(V, device=device, dtype=torch.long)

    # Process closest points first
    order = torch.argsort(dists_k[:, 0])

    # -------------------------
    # Greedy top-k assignment
    # -------------------------
    for i in order.tolist():
        for j in idx_k[i].tolist():
            if counts[j] < capacities[j]:
                assignments[i] = j
                counts[j] += 1
                break

    # -------------------------
    # Global fallback (guaranteed)
    # -------------------------
    unassigned = (assignments == -1).nonzero(as_tuple=True)[0]
    if unassigned.numel() > 0:
        # For determinism: process by increasing best distance
        ua_order = unassigned[
            torch.argsort(dists_k[unassigned, 0])
        ]

        for i in ua_order.tolist():
            free = (counts < capacities).nonzero(as_tuple=True)[0]

            # pick nearest free centroid (global)
            j = free[
                torch.argmin(distances[i, free])
            ].item()

            assignments[i] = j
            counts[j] += 1

    # -------------------------
    # Final sanity (must hold)
    # -------------------------
    if not torch.all(counts == capacities):
        raise RuntimeError(
            f"Balanced assignment failed: counts={counts.tolist()}, capacities={capacities.tolist()}"
        )

    return assignments



@torch.no_grad()
def collect_row_representatives(
    train_ds,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Collect exactly one representative latent per row (mean over K latents).

    Returns:
        X_rows: Tensor[R, H]
    """
    hf_ds = getattr(train_ds, "ds", None)
    if hf_ds is None:
        raise RuntimeError("train_ds has no underlying HF dataset (.ds)")

    reps = []
    H: Optional[int] = None

    for i, ex in enumerate(hf_ds):
        latent_states = ex.get("latent_states", None)
        K = ex.get("num_latents", None)
        if K<20:
            continue

        if latent_states is None or K is None:
            raise RuntimeError(f"Row {i} missing latent_states or num_latents")

        if K <= 0 or K > len(latent_states):
            raise RuntimeError(f"Row {i}: invalid num_latents={K}")

        row_latents = latent_states[:K]

        if H is None:
            H = len(row_latents[0])
        else:
            for h in row_latents:
                if len(h) != H:
                    raise RuntimeError(f"Row {i}: inconsistent latent dim")

        # Mean latent for this row
        h_mean = torch.tensor(row_latents, dtype=dtype).mean(dim=0)
        reps.append(h_mean)

    if not reps:
        raise RuntimeError("No row representatives collected")

    X_rows = torch.stack(reps)

    if device is not None:
        X_rows = X_rows.to(device)

    return X_rows


@torch.no_grad()
def kmeans_pp_row_aware(
    X: torch.Tensor,          # [N, H] all latents
    X_rows: torch.Tensor,     # [R, H] one per row
    V: int,
    n_iters: int,
    row_ptr,
    assign_mode,
    seed: int,
) -> torch.Tensor:
    """
    KMeans with row-aware KMeans++ seeding.
    """
    if X.ndim != 2 or X_rows.ndim != 2:
        raise ValueError("X and X_rows must be 2D")
    if X.size(1) != X_rows.size(1):
        raise ValueError("Dim mismatch between X and X_rows")
    if X_rows.size(0) < V:
        raise ValueError("Need at least V rows for row-aware seeding")

    device = X.device
    dtype = X.dtype
    R, H = X_rows.shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # ------------------------------------------------------------
    # Row-aware KMeans++ seeding
    # ------------------------------------------------------------
    centroids = torch.empty((V, H), device=device, dtype=dtype)

    # First centroid
    first_idx = torch.randint(0, R, (1,), generator=g, device=device).item()
    centroids[0] = X_rows[first_idx]

    d2 = torch.sum((X_rows - centroids[0]) ** 2, dim=1)

    for k in range(1, V):
        if torch.all(d2 == 0):
            centroids[k:] = centroids[0]
            break

        probs = d2 / d2.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centroids[k] = X_rows[idx]

        new_d2 = torch.sum((X_rows - centroids[k]) ** 2, dim=1)
        d2 = torch.minimum(d2, new_d2)

    # ------------------------------------------------------------
    # Lloyd iterations (UNCHANGED)
    # ------------------------------------------------------------
    for it in range(n_iters):
        if assign_mode == 'row_exclusive':
            assignments = row_exclusive_assign(X, centroids, row_ptr)
        else:


            assignments = balanced_assign(X, centroids, topk=3)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(V, device=device, dtype=torch.long)

        new_centroids.index_add_(0, assignments, X)
        counts.index_add_(0, assignments, torch.ones_like(assignments))

        for k in range(V):
            if counts[k] == 0:
                farthest_idx = torch.argmax(
                    torch.min(torch.cdist(X, centroids) ** 2, dim=1).values
                ).item()
                new_centroids[k] = X[farthest_idx]
            else:
                new_centroids[k] /= counts[k]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return centroids

@torch.no_grad()
def row_exclusive_assign(
    X: torch.Tensor,              # [N, H]
    centroids: torch.Tensor,      # [V, H]
    row_ptr: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Row-exclusive assignment:
    - Within each row, each centroid can be used at most once
    - Deterministic, greedy
    - For diagnostics / clustering ONLY

    Returns:
        assignments: LongTensor[N]
    """
    assert X.ndim == 2
    assert centroids.ndim == 2
    assert X.size(1) == centroids.size(1)

    device = X.device
    N = X.size(0)
    V = centroids.size(0)

    assignments = torch.full((N,), -1, device=device, dtype=torch.long)

    # Precompute all distances once
    # distances[n, v] = ||X[n] - centroids[v]||
    distances = torch.cdist(X, centroids)  # [N, V]

    for row_idx, (start, end) in enumerate(row_ptr):
        K = end - start
        if K <= 0:
            continue
        if K > V:
            raise RuntimeError(
                f"Row {row_idx}: K={K} > V={V}, row-exclusive assignment impossible"
            )

        row_indices = torch.arange(start, end, device=device)

        # Distances for this row: [K, V]
        row_dists = distances[row_indices]

        # Sort latents by confidence (margin between best and 2nd best)
        top2 = torch.topk(row_dists, k=2, dim=1, largest=False).values
        margins = top2[:, 1] - top2[:, 0]  # larger = more confident
        order = torch.argsort(margins, descending=True)

        used_centroids = set()

        for idx in order.tolist():
            n = row_indices[idx].item()

            # Try centroids in increasing distance order
            cand_centroids = torch.argsort(row_dists[idx])

            assigned = False
            for c in cand_centroids.tolist():
                if c not in used_centroids:
                    assignments[n] = c
                    used_centroids.add(c)
                    assigned = True
                    break

            if not assigned:
                # This should be extremely rare if K <= V
                raise RuntimeError(
                    f"Row {row_idx}: failed to find unused centroid"
                )

        # Final sanity for the row
        if len(used_centroids) != K:
            raise RuntimeError(
                f"Row {row_idx}: assigned {len(used_centroids)} centroids for K={K}"
            )

    # Global sanity
    if (assignments < 0).any():
        bad = (assignments < 0).nonzero(as_tuple=True)[0][:10].tolist()
        raise RuntimeError(f"Unassigned latents at indices {bad}")

    return assignments
