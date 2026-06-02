from __future__ import annotations

import math

import torch
import torch.nn.functional as F

STOP_ACTION = -1


                                                                                

def _mix_with_uniform(
    base_probs: torch.Tensor,
    rand_explore_prob: float,
) -> torch.Tensor:
    p = float(max(0.0, min(1.0, rand_explore_prob)))
    if p <= 0.0 or base_probs.numel() == 0:
        return base_probs
    uniform = torch.full_like(base_probs, 1.0 / float(base_probs.numel()))
    mixed = (1.0 - p) * base_probs + p * uniform
    return mixed / mixed.sum().clamp_min(1e-12)

def get_leaf_mask(edge_index: torch.Tensor, n_chunks: int) -> torch.Tensor:
    if edge_index.size(1) == 0:
                                         
        return torch.ones(n_chunks, dtype=torch.bool,
                          device=edge_index.device)
    has_out = torch.zeros(n_chunks, dtype=torch.bool,
                          device=edge_index.device)
    has_out[edge_index[0]] = True
    return ~has_out


def reconstruct_graph(
    h_chunk_all:    torch.Tensor,                               
    edge_index_init: torch.Tensor,                            
    active_mask:    torch.Tensor,                                             
    logodds_all:    torch.Tensor,         
    device:         torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    active_idx = active_mask.nonzero(as_tuple=True)[0]               
    index_map = {int(o): i for i, o in enumerate(active_idx.tolist())}

    h_sub      = h_chunk_all[active_idx].to(device)
    logodds_sub = logodds_all[active_idx].to(device)

    if edge_index_init.size(1) == 0:
        ei_sub = torch.zeros((2, 0), dtype=torch.long, device=device)
        return h_sub, ei_sub, logodds_sub, index_map

                                                     
    src, dst = edge_index_init[0], edge_index_init[1]
    keep = active_mask[src] & active_mask[dst]
    if not keep.any():
        ei_sub = torch.zeros((2, 0), dtype=torch.long, device=device)
        return h_sub, ei_sub, logodds_sub, index_map

    src_k = src[keep]
    dst_k = dst[keep]
    new_src = torch.tensor([index_map[int(s)] for s in src_k.tolist()],
                           dtype=torch.long, device=device)
    new_dst = torch.tensor([index_map[int(d)] for d in dst_k.tolist()],
                           dtype=torch.long, device=device)
    ei_sub = torch.stack([new_src, new_dst], dim=0)
    return h_sub, ei_sub, logodds_sub, index_map


def _forward_single(
    policy_model,
    h_sub:      torch.Tensor,           
    ei_sub:     torch.Tensor,           
    logodds_sub: torch.Tensor,       
    alpha_logodds: float,
    device: torch.device,
    beta_gnn: float = 1.0,
    q_emb: torch.Tensor | None = None,
):
    batch = torch.zeros(h_sub.size(0), dtype=torch.long, device=device)
    action_logits, stop_logit, log_value = policy_model(
        h_sub, ei_sub, batch, logodds_sub,
        q_emb=q_emb.to(device) if isinstance(q_emb, torch.Tensor) else None,
        alpha_logodds=alpha_logodds, beta_gnn=beta_gnn,
    )
    return action_logits, stop_logit.squeeze(0), log_value.squeeze(0)


def active_mask_from_actions(num_nodes: int, actions: list[int]) -> torch.Tensor:
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    for action in actions:
        if action == STOP_ACTION:
            break
        if 0 <= int(action) < num_nodes:
            active_mask[int(action)] = False
    return active_mask


def _terminal_vllm_from_mask(
    tail_names_all: list[list[str]],
    active_mask: torch.Tensor,
    answer_texts: list[str],
) -> float:
    if not answer_texts:
        return 0.1

    gold = {str(a).strip().lower() for a in answer_texts if str(a).strip()}
    if not gold:
        return 0.1

    for idx in active_mask.nonzero(as_tuple=True)[0].tolist():
        if idx >= len(tail_names_all):
            continue
        for name in tail_names_all[idx]:
            if str(name).strip().lower() in gold:
                return 1.0
    return 0.1


def _shape_guidance_return(
    p_c_all: torch.Tensor,
    p_s_all: torch.Tensor,
    tail_names_all: list[list[str]],
    answer_texts: list[str],
    active_mask: torch.Tensor,
    pcau_floor: float,
    pcau_weight: float,
    lambda_non: float,
) -> float:
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        mean_pcau = float(pcau_floor)
        pnon_norm = 0.0
    else:
        p_c_sub = p_c_all[active_idx]
        p_s_sub = p_s_all[active_idx]
        mean_pcau = max(float(p_c_sub.mean().item()), float(pcau_floor))
        pnon_norm = float(p_s_sub.sum().item()) / max(float(p_c_all.size(0)), 1.0)

    terminal_vllm = _terminal_vllm_from_mask(
        tail_names_all=tail_names_all,
        active_mask=active_mask,
        answer_texts=answer_texts,
    )
    return (
        math.log(float(terminal_vllm))
        + pcau_weight * math.log(mean_pcau)
        - lambda_non * pnon_norm
    )


def _candidate_local_indices(
    action_logits: torch.Tensor,
    leaf_indices: torch.Tensor,
    pair_topk: int,
) -> torch.Tensor:
    if pair_topk <= 0 or int(leaf_indices.numel()) <= int(pair_topk):
        return leaf_indices

    leaf_scores = action_logits[leaf_indices]
    order = torch.argsort(leaf_scores, descending=True)
    return leaf_indices[order[: int(pair_topk)]]


def _dedup_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in indices:
        if idx in seen:
            continue
        deduped.append(idx)
        seen.add(idx)
    return deduped


def _guidance_state_score(
    p_c_all: torch.Tensor,
    p_s_all: torch.Tensor,
    tail_names_all: list[list[str]],
    answer_texts: list[str],
    active_mask: torch.Tensor,
    pcau_floor: float,
    pcau_weight: float,
    lambda_non: float,
) -> float:
    return _shape_guidance_return(
        p_c_all=p_c_all,
        p_s_all=p_s_all,
        tail_names_all=tail_names_all,
        answer_texts=answer_texts,
        active_mask=active_mask,
        pcau_floor=pcau_floor,
        pcau_weight=pcau_weight,
        lambda_non=lambda_non,
    )


def _greedy_multistep_guidance_score(
    h_all: torch.Tensor,
    ei_init: torch.Tensor,
    p_c_all: torch.Tensor,
    p_s_all: torch.Tensor,
    tail_names_all: list[list[str]],
    answer_texts: list[str],
    active_mask_start: torch.Tensor,
    min_chunks: int,
    total_steps: int,
    pcau_floor: float,
    pcau_weight: float,
    lambda_non: float,
) -> float:
    active_mask = active_mask_start.clone()
    extra_steps = max(int(total_steps) - 1, 0)

    for _ in range(extra_steps):
        if int(active_mask.sum().item()) <= int(min_chunks):
            break

        h_sub, ei_sub, _, index_map = reconstruct_graph(
            h_all,
            ei_init,
            active_mask,
            torch.zeros_like(p_c_all),
            active_mask.device,
        )
        if h_sub.size(0) == 0:
            break

        current_score = _guidance_state_score(
            p_c_all=p_c_all,
            p_s_all=p_s_all,
            tail_names_all=tail_names_all,
            answer_texts=answer_texts,
            active_mask=active_mask,
            pcau_floor=pcau_floor,
            pcau_weight=pcau_weight,
            lambda_non=lambda_non,
        )

        leaf_mask_sub = get_leaf_mask(ei_sub, h_sub.size(0))
        leaf_indices = leaf_mask_sub.nonzero(as_tuple=True)[0]
        if leaf_indices.numel() == 0:
            break

        reverse_index_map = {v: k for k, v in index_map.items()}
        best_score = current_score
        best_global_idx = None

        for local_idx in leaf_indices.tolist():
            global_idx = int(reverse_index_map[int(local_idx)])
            next_active = active_mask.clone()
            next_active[global_idx] = False
            score = _guidance_state_score(
                p_c_all=p_c_all,
                p_s_all=p_s_all,
                tail_names_all=tail_names_all,
                answer_texts=answer_texts,
                active_mask=next_active,
                pcau_floor=pcau_floor,
                pcau_weight=pcau_weight,
                lambda_non=lambda_non,
            )
            if score > best_score:
                best_score = score
                best_global_idx = global_idx

        if best_global_idx is None:
            break
        active_mask[best_global_idx] = False

    return _guidance_state_score(
        p_c_all=p_c_all,
        p_s_all=p_s_all,
        tail_names_all=tail_names_all,
        answer_texts=answer_texts,
        active_mask=active_mask,
        pcau_floor=pcau_floor,
        pcau_weight=pcau_weight,
        lambda_non=lambda_non,
    )


def _build_hinge_pairs(
    h_all: torch.Tensor,
    ei_init: torch.Tensor,
    p_c_all: torch.Tensor,
    p_s_all: torch.Tensor,
    tail_names_all: list[list[str]],
    answer_texts: list[str],
    active_mask: torch.Tensor,
    index_map: dict[int, int],
    candidate_local_indices: torch.Tensor,
    chosen_global_idx: int | None,
    pair_mode: str,
    pair_gap: float,
    max_pairs_per_state: int,
    min_chunks: int,
    total_steps: int,
    pcau_floor: float,
    pcau_weight: float,
    lambda_non: float,
) -> list[dict]:
    reverse_index_map = {v: k for k, v in index_map.items()}
    global_candidates = [
        int(reverse_index_map[int(local_idx)])
        for local_idx in candidate_local_indices.tolist()
    ]
    if pair_mode == "actual_vs_others" and chosen_global_idx is not None:
        global_candidates.append(int(chosen_global_idx))
    global_candidates = _dedup_preserve_order(global_candidates)

    if len(global_candidates) < 2:
        return []

    candidate_scores: list[tuple[int, float]] = []
    for global_idx in global_candidates:
        next_active = active_mask.clone()
        next_active[int(global_idx)] = False
        score = _greedy_multistep_guidance_score(
            h_all=h_all,
            ei_init=ei_init,
            p_c_all=p_c_all,
            p_s_all=p_s_all,
            tail_names_all=tail_names_all,
            answer_texts=answer_texts,
            active_mask_start=next_active,
            min_chunks=min_chunks,
            total_steps=total_steps,
            pcau_floor=pcau_floor,
            pcau_weight=pcau_weight,
            lambda_non=lambda_non,
        )
        candidate_scores.append((int(global_idx), float(score)))

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    margin = float(pair_gap)
    pairs: list[dict] = []

    if pair_mode == "actual_vs_others":
        if chosen_global_idx is None:
            return []
        actual_score = None
        for action_idx, score in candidate_scores:
            if action_idx == int(chosen_global_idx):
                actual_score = score
                break
        if actual_score is None:
            return []
        for other_idx, other_score in candidate_scores:
            if other_idx == int(chosen_global_idx):
                continue
            gap = abs(float(actual_score - other_score))
            if gap < margin:
                continue
            if actual_score >= other_score:
                winner_action, loser_action = int(chosen_global_idx), int(other_idx)
            else:
                winner_action, loser_action = int(other_idx), int(chosen_global_idx)
            pairs.append({
                "active_mask": active_mask.detach().cpu(),
                "winner_action": winner_action,
                "loser_action": loser_action,
                "score_gap": gap,
            })
    else:
        for hi in range(len(candidate_scores)):
            winner_idx, winner_score = candidate_scores[hi]
            for lo in range(hi + 1, len(candidate_scores)):
                loser_idx, loser_score = candidate_scores[lo]
                gap = float(winner_score - loser_score)
                if gap < margin:
                    continue
                pairs.append({
                    "active_mask": active_mask.detach().cpu(),
                    "winner_action": int(winner_idx),
                    "loser_action": int(loser_idx),
                    "score_gap": gap,
                })

    if max_pairs_per_state > 0 and len(pairs) > int(max_pairs_per_state):
        pairs.sort(key=lambda item: float(item["score_gap"]), reverse=True)
        pairs = pairs[: int(max_pairs_per_state)]
    return pairs


def sample_pruning_trajectory_with_hinge_pairs(
    initial_state,
    policy_model,
    answer_texts: list[str],
    device: torch.device,
    temperature: float = 1.0,
    min_chunks: int = 1,
    greedy: bool = False,
    alpha_logodds: float = 1.0,
    beta_gnn: float = 1.0,
    rand_explore_prob: float = 0.0,
    pair_topk: int = 0,
    pair_mode: str = "all",
    pair_gap: float = 0.0,
    max_pairs_per_state: int = 4,
    pair_horizon: int = 2,
    pcau_floor: float = 0.05,
    pcau_weight: float = 1.0,
    lambda_non: float = 0.1,
    collect_hinge_pairs: bool = True,
) -> dict | None:
    if initial_state is None or initial_state.node_feats.size(0) == 0:
        return None

    h_all = initial_state.node_feats.detach().cpu()
    ei_init = initial_state.edge_index.detach().cpu()
    logodds_all = initial_state.logodds_norm.detach().cpu()
    p_c_all = initial_state.p_c.detach().cpu()
    p_s_all = initial_state.p_s.detach().cpu()
    tail_names_all = getattr(initial_state, "tail_names", [])
    q_emb = getattr(initial_state, "q_emb", None)
    q_emb_cpu = q_emb.detach().cpu() if isinstance(q_emb, torch.Tensor) else None

    num_nodes = h_all.size(0)
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    actions: list[int] = []
    hinge_pairs: list[dict] = []

    policy_model.eval()

    while True:
        if int(active_mask.sum().item()) <= int(min_chunks):
            actions.append(STOP_ACTION)
            break

        h_sub, ei_sub, logodds_sub, index_map = reconstruct_graph(
            h_all, ei_init, active_mask, logodds_all, device,
        )
        if h_sub.size(0) == 0:
            actions.append(STOP_ACTION)
            break

        with torch.no_grad():
            action_logits, stop_logit, _ = _forward_single(
                policy_model, h_sub, ei_sub, logodds_sub,
                alpha_logodds, device, beta_gnn, q_emb=q_emb_cpu,
            )

        leaf_mask_sub = get_leaf_mask(ei_sub, h_sub.size(0))
        leaf_indices = leaf_mask_sub.nonzero(as_tuple=True)[0]
        if leaf_indices.numel() == 0:
            actions.append(STOP_ACTION)
            break

        leaf_logits = action_logits[leaf_indices] / temperature
        stop_scaled = stop_logit / temperature
        all_logits = torch.cat([leaf_logits, stop_scaled.unsqueeze(0)])
        base_probs = F.softmax(all_logits, dim=0)
        probs = _mix_with_uniform(base_probs, rand_explore_prob)

        if greedy:
            chosen = int(torch.argmax(probs).item())
        else:
            chosen = int(torch.multinomial(probs, 1).item())

        if chosen == len(leaf_indices):
            chosen_global_idx = None
            actions.append(STOP_ACTION)
        else:
            reverse_index_map = {v: k for k, v in index_map.items()}
            chosen_sub_idx = int(leaf_indices[chosen].item())
            chosen_global_idx = int(reverse_index_map[chosen_sub_idx])
            actions.append(chosen_global_idx)

        if collect_hinge_pairs:
            candidate_local_indices = _candidate_local_indices(
                action_logits=action_logits,
                leaf_indices=leaf_indices,
                pair_topk=pair_topk,
            )
            if int(candidate_local_indices.numel()) >= 2:
                hinge_pairs.extend(
                    _build_hinge_pairs(
                        h_all=h_all,
                        ei_init=ei_init,
                        p_c_all=p_c_all,
                        p_s_all=p_s_all,
                        tail_names_all=tail_names_all,
                        answer_texts=answer_texts,
                        active_mask=active_mask,
                        index_map=index_map,
                        candidate_local_indices=candidate_local_indices,
                        chosen_global_idx=chosen_global_idx,
                        pair_mode=pair_mode,
                        pair_gap=pair_gap,
                        max_pairs_per_state=max_pairs_per_state,
                        min_chunks=min_chunks,
                        total_steps=pair_horizon,
                        pcau_floor=pcau_floor,
                        pcau_weight=pcau_weight,
                        lambda_non=lambda_non,
                    )
                )

        if chosen_global_idx is None:
            break

        active_mask[chosen_global_idx] = False

    final_active_mask = active_mask_from_actions(num_nodes, actions)
    terminal_vllm = _terminal_vllm_from_mask(
        tail_names_all=tail_names_all,
        active_mask=final_active_mask,
        answer_texts=answer_texts,
    )

    return {
        "h_chunk_all": h_all,
        "edge_index_init": ei_init,
        "logodds_all": logodds_all,
        "p_c_all": p_c_all,
        "p_s_all": p_s_all,
        "actions": actions,
        "terminal_vllm": terminal_vllm,
        "initial_num_chunks": int(num_nodes),
        "tail_names_all": tail_names_all,
        "rand_explore_prob": float(rand_explore_prob),
        "q_emb": q_emb_cpu,
        "hinge_pairs": hinge_pairs,
    }
