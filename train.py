from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
THIS_DIR = Path(__file__).resolve().parent
ENCODER_DIR = PROJECT_ROOT / "CausalEncoder"

for path_str in [str(PROJECT_ROOT), str(ENCODER_DIR), str(THIS_DIR)]:
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sentence_transformers import SentenceTransformer
from model import CausalEncoder
from opts import parse_caseflow_train_args, setup_seed

from chunking import adaptive_spherical_kmeans_assign_with_soft_scores, compute_chunk_logodds
from gflownet import (
    PolicyFlowModel,
    build_chunk_graph_state,
    evaluate_hinge_loss,
    subtb_loss,
)
from graph_builder import (
    OfflineEmbeddingStore,
    build_graph_with_llm_sampling,
    extract_graph_payload,
    get_topic_name_from_item,
    load_local_qa_split,
    parse_answers,
)
from rollout import sample_pruning_trajectory_with_hinge_pairs


def load_qa_split(dataset_name: str, split: str) -> list[dict]:
    return load_local_qa_split(dataset_name, split)


def get_embedding_prefix(args) -> str:
    return args.emb_prefix.strip() or args.dataset_name


def select_training_subset(datas: list[dict], args) -> list[dict]:
    if int(getattr(args, "max_questions", 0)) <= 0:
        return datas

    total = len(datas)
    n = min(int(args.max_questions), total)
    print(f"[CaseFlow-Train] Subset selected: {n}/{total}", flush=True)
    return datas[:n]


def encode_full_graph(
    graph_data,
    encoder: CausalEncoder,
    device:  torch.device,
):
    graph_data = graph_data.to(device)
    encoder.eval()
    with torch.no_grad():
        z_c, z_s, alpha_c, alpha_s, *_ = encoder(graph_data, return_alpha=True)
    return z_c, z_s, alpha_c.squeeze(-1), alpha_s.squeeze(-1)


def build_profile(
    z_c: torch.Tensor, z_s: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    ratio = (z_c / (z_s.abs() + 0.1)).clamp(-10.0, 10.0)
    return torch.cat([z_c, z_s, ratio, z_c - z_s], dim=-1)


def _lambda_non_at_step(args, global_step: int) -> float:
    if args.lambda_warmup_steps <= 0:
        return float(args.lam)
    scale = min(float(global_step) / float(args.lambda_warmup_steps), 1.0)
    return float(args.lam) * scale

def _save(path, policy_model, hidden, profile_dim, emb_prefix, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "policy_model_state": policy_model.state_dict(),
        "hidden": hidden,
        "profile_dim": profile_dim,
        "cluster_feature_dim": profile_dim,
        "clusterer": "local_spherical_kmeans",
        "K": args.K,
        "tau": args.tau,
        "kmeans_iters": args.kmeans_iters,
        "sim_one": args.sim_one,
        "k1_quantile": args.k1_quantile,
        "gnn_layers": args.gnn_layers,
        "rand_explore_prob": args.rand_explore_prob,
        "dataset_name": args.dataset_name,
        "emb_prefix": emb_prefix,
        "variant": "hinge_loss",
        "use_hinge_loss": not bool(args.disable_hinge_loss),
        "w_hinge": args.w_hinge,
        "hinge_warmup_steps": args.hinge_warmup_steps,
        "hinge_pair_topk": args.hinge_pair_topk,
        "eval_pair_mode": args.eval_pair_mode,
        "hinge_pair_gap": args.hinge_pair_gap,
        "hinge_margin": args.hinge_margin,
        "hinge_max_pairs_per_state": args.hinge_max_pairs_per_state,
        "hinge_pair_horizon": args.hinge_pair_horizon,
    }
    torch.save(payload, path)


def _hinge_weight_at_step(args, global_step: int) -> float:
    if args.hinge_warmup_steps <= 0:
        return float(args.w_hinge)
    scale = min(float(global_step) / float(args.hinge_warmup_steps), 1.0)
    return float(args.w_hinge) * scale


def main():
    args = parse_caseflow_train_args()
    setup_seed(args.seed)
    os.chdir(PROJECT_ROOT)

    if torch.cuda.is_available():
        gpu_index = int(args.gpu_id)
        torch.cuda.set_device(gpu_index)
        torch.cuda.set_per_process_memory_fraction(0.9, gpu_index)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[CaseFlow-Train] Device: {device}", flush=True)

    sbert = SentenceTransformer(args.sbert_model, device=str(device))
    embed_dim = sbert.get_sentence_embedding_dimension()

    emb_prefix = get_embedding_prefix(args)
    store = OfflineEmbeddingStore(emb_dir=args.emb_dir, device=device, prefix=emb_prefix)
    assert store.emb_dim == embed_dim

    if not os.path.isfile(args.encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder_path}")

    ckpt = torch.load(args.encoder_path, map_location=device)
    hidden = ckpt.get("embed_dim", embed_dim)
    m_args = ckpt.get("args", {})
    encoder = CausalEncoder(
        in_channels=hidden, hidden=hidden,
        num_layers=m_args.get("layers", 3), dropout=0.0,
    ).to(device)
    model_state = ckpt["model_state"]
    own_state = encoder.state_dict()
    filtered = {k: v for k, v in model_state.items()
                if k in own_state and own_state[k].shape == v.shape}
    encoder.load_state_dict(filtered, strict=False)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    profile_dim = 4 * hidden
    policy_model = PolicyFlowModel(hidden=hidden, gnn_layers=args.gnn_layers, dropout=0.1).to(device)
    optimizer = Adam(policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    datas = load_qa_split(args.dataset_name, args.split)
    datas = select_training_subset(datas, args)
    print(f"[CaseFlow-Train] Total: {len(datas)} items", flush=True)

    global_step = 0

    for ep in range(args.epochs):
        iterator = tqdm(datas, desc=f"Epoch {ep + 1}", dynamic_ncols=True)
        for local_idx, item in enumerate(iterator):
            q_idx = local_idx
            question = item["question"]
            answer_texts = parse_answers(item)
            topic_name = get_topic_name_from_item(item)
            graph_triples, graph_label_map = extract_graph_payload(item)
            if not topic_name or not graph_triples:
                continue

            z_q = sbert.encode([question], convert_to_tensor=True).to(device)[0]
            z_a = (
                sbert.encode(answer_texts, convert_to_tensor=True).to(device).mean(dim=0)
                if answer_texts
                else sbert.encode([topic_name], convert_to_tensor=True).to(device)[0]
            )

            graph_data, rel_groups, node_names = build_graph_with_llm_sampling(
                graph_triples, topic_name, store, z_q, z_a, device,
                question=question, args=args, max_hops=args.depth,
                resolve_display=False, label_map=graph_label_map,
            )
            if graph_data.x.size(0) <= 1 or not rel_groups:
                continue

            z_c_all, z_s_all, alpha_c_all, alpha_s_all = encode_full_graph(graph_data, encoder, device)

            chunk_info_list = []
            global_offset = 0

            for hop, relation, tail_indices, parent_names_by_tail in rel_groups:
                if not tail_indices:
                    continue

                idx_t = torch.tensor(tail_indices, dtype=torch.long, device=device)
                z_c_t = z_c_all[idx_t]
                z_s_t = z_s_all[idx_t]
                alpha_c_t = alpha_c_all[idx_t]
                alpha_s_t = alpha_s_all[idx_t]
                tail_names_rel = [node_names[i] for i in tail_indices]

                profile = build_profile(z_c_t, z_s_t, args.eps)

                _centers, pi_soft, k_star = adaptive_spherical_kmeans_assign_with_soft_scores(
                    profile,
                    max_k=args.K,
                    tau=args.tau,
                    n_iters=args.kmeans_iters,
                    eps=args.eps,
                    sim_one=args.sim_one,
                    k1_quantile=args.k1_quantile,
                )
                pi_hard = F.one_hot(k_star, num_classes=pi_soft.size(1)).float()

                (_logodds, logodds_norm, mean_z_c, active,
                 p_c_raw, p_s_raw) = compute_chunk_logodds(
                    pi_soft, pi_hard, alpha_c_t, alpha_s_t, z_c_t, eps=args.eps,
                )

                logodds_norm_act = logodds_norm[active]
                mean_z_c_act = mean_z_c[active]
                p_c_raw_act = p_c_raw[active]
                p_s_raw_act = p_s_raw[active]
                k_act = int(active.sum().item())
                if k_act == 0:
                    continue


                tail_names_by_k = [[] for _ in range(pi_soft.size(1))]
                parent_names_by_k = [set() for _ in range(pi_soft.size(1))]
                for t_idx, k in enumerate(k_star.tolist()):
                    if t_idx < len(tail_names_rel):
                        tail_names_by_k[k].append(tail_names_rel[t_idx])
                    if t_idx < len(parent_names_by_tail):
                        parent_names_by_k[k].update(parent_names_by_tail[t_idx])

                active_k_indices = active.nonzero(as_tuple=False).squeeze(-1).tolist()
                active_tail_names = [tail_names_by_k[k] for k in active_k_indices]
                active_parent_names = [sorted(parent_names_by_k[k]) for k in active_k_indices]

                chunk_info_list.append({
                    "hop": hop,
                    "relation": relation,
                    "global_start": global_offset,
                    "K_act": k_act,
                    "mean_z_c": mean_z_c_act.detach(),
                    "logodds_norm": logodds_norm_act.detach(),
                    "p_c": p_c_raw_act.detach(),
                    "p_s": p_s_raw_act.detach(),
                    "tail_names_per_chunk": active_tail_names,
                    "parent_names_per_chunk": active_parent_names,
                })
                global_offset += k_act

            if not chunk_info_list:
                continue

            state0 = build_chunk_graph_state(chunk_info_list, device, q_emb=z_q.detach())
            if state0 is None or state0.K_active == 0:
                continue

            lambda_non = _lambda_non_at_step(args, global_step)
            collect_hinge_pairs = not bool(args.disable_hinge_loss)
            new_trajs = []
            for _ in range(args.n_rollouts):
                traj = sample_pruning_trajectory_with_hinge_pairs(
                    state0, policy_model,
                    answer_texts=answer_texts,
                    device=device,
                    temperature=args.gflow_temp,
                    min_chunks=args.min_chunks,
                    alpha_logodds=args.gflow_alpha,
                    beta_gnn=args.gflow_beta,
                    rand_explore_prob=args.rand_explore_prob,
                    pair_topk=args.hinge_pair_topk,
                    pair_mode=args.eval_pair_mode,
                    pair_gap=args.hinge_pair_gap,
                    max_pairs_per_state=args.hinge_max_pairs_per_state,
                    pair_horizon=args.hinge_pair_horizon,
                    pcau_floor=args.pcau_floor,
                    pcau_weight=args.pcau_weight,
                    lambda_non=lambda_non,
                    collect_hinge_pairs=collect_hinge_pairs,
                )
                if traj is not None:
                    new_trajs.append(traj)

            policy_model.train()
            batch_trajs = new_trajs
            l_subtb = subtb_loss(
                batch_trajs, policy_model, device=device,
                alpha_logodds=args.gflow_alpha, beta_gnn=args.gflow_beta,
                temperature=args.gflow_temp, lambda_non=lambda_non,
                pcau_floor=args.pcau_floor, pcau_weight=args.pcau_weight,
                rand_explore_prob=args.rand_explore_prob,
            )
            if collect_hinge_pairs:
                l_hinge = evaluate_hinge_loss(
                    batch_trajs, policy_model, device=device,
                    alpha_logodds=args.gflow_alpha, beta_gnn=args.gflow_beta,
                    hinge_margin=args.hinge_margin,
                )
                hinge_weight = _hinge_weight_at_step(args, global_step)
            else:
                l_hinge = torch.tensor(0.0, device=device)
                hinge_weight = 0.0
            loss = args.w_subtb * l_subtb + hinge_weight * l_hinge

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                grad_params = list(policy_model.parameters())
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
                optimizer.step()
                global_step += 1

            if args.checkpoint_interval > 0 and (q_idx + 1) % args.checkpoint_interval == 0:
                ckpt_path = args.save_path.replace(".pth", f"_ckpt{q_idx + 1}.pth")
                _save(ckpt_path, policy_model, hidden, profile_dim, emb_prefix, args)
                print(f"[CaseFlow-Train] Checkpoint: {ckpt_path}", flush=True)

    _save(args.save_path, policy_model, hidden, profile_dim, emb_prefix, args)
    print(f"[CaseFlow-Train] Done. Saved: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
