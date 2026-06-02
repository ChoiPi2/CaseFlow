from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score


def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def _farthest_point_init(x_n: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    n = x_n.size(0)
    if k >= n:
        return x_n.clone()

    mean_vec = _normalize_rows(x_n.mean(dim=0, keepdim=True), eps=eps)[0]
    first_idx = (1.0 - x_n @ mean_vec).argmax().item()
    chosen = [first_idx]

    min_dist = 1.0 - (x_n @ x_n[first_idx])
    for _ in range(1, k):
        next_idx = min_dist.argmax().item()
        chosen.append(next_idx)
        dist_to_new = 1.0 - (x_n @ x_n[next_idx])
        min_dist = torch.minimum(min_dist, dist_to_new)

    return x_n[torch.tensor(chosen, device=x_n.device)]


@torch.no_grad()
def fixed_k_spherical_kmeans(
    features: torch.Tensor,
    k: int,
    n_iters: int = 5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
    n = int(features.size(0))
    if n == 0:
        raise ValueError("features must be non-empty")

    x_n = _normalize_rows(features, eps=eps)
    k = max(1, min(int(k), n))

    if k == 1:
        center = _normalize_rows(x_n.mean(dim=0, keepdim=True), eps=eps)
        labels = torch.zeros(n, dtype=torch.long, device=features.device)
        return center, labels

    centers = _farthest_point_init(x_n, k, eps=eps)

    for _ in range(max(1, int(n_iters))):
        sims = x_n @ centers.t()
        labels = sims.argmax(dim=-1)

        new_centers = []
        for k_idx in range(k):
            mask = labels == k_idx
            if mask.any():
                center = _normalize_rows(x_n[mask].mean(dim=0, keepdim=True), eps=eps)[0]
            else:
                center = centers[k_idx]
            new_centers.append(center)
        new_centers = torch.stack(new_centers, dim=0)

        if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    final_labels = (x_n @ centers.t()).argmax(dim=-1)
    return centers, final_labels


def _single_cluster_result(
    x_n: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = _normalize_rows(x_n.mean(dim=0, keepdim=True), eps=eps)
    n = x_n.size(0)
    pi_soft = torch.ones(n, 1, device=device, dtype=dtype)
    k_star = torch.zeros(n, dtype=torch.long, device=device)
    return center, pi_soft, k_star


@torch.no_grad()
def adaptive_spherical_kmeans_assign_with_soft_scores(
    features: torch.Tensor,
    max_k: int,
    tau: float = 1.0,
    n_iters: int = 5,
    eps: float = 1e-8,
    sim_one: float = 0.85,
    k1_quantile: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
    n, _ = features.shape
    if n == 0:
        raise ValueError("features must be non-empty")

    x_n = _normalize_rows(features, eps=eps)
    max_k = max(1, min(int(max_k), int(n)))
    tau = max(float(tau), 1e-3)
    sim_one = float(max(-1.0, min(1.0, sim_one)))
    k1_quantile = float(max(0.0, min(1.0, k1_quantile)))

    if n == 1 or max_k == 1:
        return _single_cluster_result(x_n, features.dtype, features.device, eps)

    if n == 2:
        pair_sim = float((x_n[0] @ x_n[1]).item())
        if pair_sim >= sim_one:
            return _single_cluster_result(x_n, features.dtype, features.device, eps)
        centers, k_star = fixed_k_spherical_kmeans(x_n, k=2, n_iters=n_iters, eps=eps)
        pi_soft = F.softmax((x_n @ centers.t()) / tau, dim=-1)
        return centers, pi_soft, k_star

    sim = x_n @ x_n.t()
    upper_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x_n.device), diagonal=1)
    upper_vals = sim[upper_mask]
    gate_stat = float(torch.quantile(upper_vals, k1_quantile).item())
    if gate_stat >= sim_one:
        return _single_cluster_result(x_n, features.dtype, features.device, eps)

    k_hi = min(max_k, n - 1)
    x_cpu = x_n.detach().cpu().numpy()
    best: tuple[int, float, torch.Tensor, torch.Tensor] | None = None
    fallback: tuple[int, torch.Tensor, torch.Tensor] | None = None

    for k in range(2, k_hi + 1):
        centers, labels = fixed_k_spherical_kmeans(x_n, k=k, n_iters=n_iters, eps=eps)
        uniq = int(labels.unique().numel())
        if fallback is None:
            fallback = (k, centers, labels)
        if uniq < 2 or uniq >= n:
            continue
        try:
            score = float(silhouette_score(x_cpu, labels.detach().cpu().numpy(), metric="cosine"))
        except ValueError:
            continue
        if best is None or score > best[1] or (math.isclose(score, best[1]) and k < best[0]):
            best = (k, score, centers, labels)

    if best is not None:
        _, _, centers, k_star = best
    elif fallback is not None:
        _, centers, k_star = fallback
    else:
        return _single_cluster_result(x_n, features.dtype, features.device, eps)

    centers = _normalize_rows(centers, eps=eps)
    pi_soft = F.softmax((x_n @ centers.t()) / tau, dim=-1)
    return centers, pi_soft, k_star


def compute_chunk_logodds(
    pi_soft: torch.Tensor,
    pi_hard: torch.Tensor,
    alpha_c: torch.Tensor,
    alpha_s: torch.Tensor,
    z_c: torch.Tensor,
    eps: float = 1e-8,
    temp: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = pi_soft.sum(dim=0) + eps
    p_c = (pi_soft * alpha_c.unsqueeze(-1)).sum(dim=0) / denom
    p_s = (pi_soft * alpha_s.unsqueeze(-1)).sum(dim=0) / denom

    logodds = torch.log((p_s + eps) / (p_c + eps))
    logodds_norm = torch.tanh(logodds / temp)

    count_k = pi_hard.sum(dim=0)
    mean_z_c = (pi_hard.T @ z_c) / (count_k.unsqueeze(-1) + eps)
    active = count_k > 0

    return logodds, logodds_norm, mean_z_c, active, p_c, p_s
