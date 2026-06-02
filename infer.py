from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_ORIG_SYS_PATH = list(sys.path)
_CWD = Path.cwd().resolve()
sys.path = [
    p for p in sys.path
    if p != "" and Path(p or ".").resolve() != _CWD
]
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.path = _ORIG_SYS_PATH

PROJECT_ROOT = Path(__file__).resolve().parent
THIS_DIR     = Path(__file__).resolve().parent
ENCODER_DIR  = PROJECT_ROOT / "CausalEncoder"

for p in [str(PROJECT_ROOT), str(ENCODER_DIR), str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import heapq
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import count

from model import CausalEncoder
from opts import parse_caseflow_infer_args, setup_seed

from chunking import adaptive_spherical_kmeans_assign_with_soft_scores, compute_chunk_logodds
from gflownet import PolicyFlowModel, build_chunk_graph_state, load_policy_model_state

from graph_builder import (
    OfflineEmbeddingStore,
    build_graph_with_llm_sampling,
    extract_graph_payload,
    get_topic_name_from_item,
    load_local_qa_split,
    parse_answers,
    resolve_graph_triples_for_display,
)

from utils import build_chunk_readout, run_llm, save_2_jsonl
from prompts import question_prompt_with_knowledge

from rollout import get_leaf_mask


DEFAULT_SEARCH_BUDGET = 20


def build_profile(z_c: torch.Tensor, z_s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ratio = (z_c / (z_s.abs() + 0.1)).clamp(-10.0, 10.0)
    return torch.cat([z_c, z_s, ratio, z_c - z_s], dim=-1)


def encode_full_graph(graph_data, encoder, device):
    graph_data = graph_data.to(device)
    encoder.eval()
    with torch.no_grad():
        out = encoder(graph_data, return_alpha=True)
        if len(out) < 4:
            raise ValueError(
                f"Unexpected encoder output length: {len(out)} (expected >= 4)"
            )
        z_c, z_s, alpha_c, alpha_s = out[:4]
    return z_c, z_s, alpha_c.squeeze(-1), alpha_s.squeeze(-1)


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


_FREEBASE_ID_RE = re.compile(r"^(?:[mg]\.)[0-9a-z_]+$", re.IGNORECASE)
_RELATION_PATH_RE = re.compile(r"\b[a-z_]+\.[a-z_]+\b")
_JUNK_ANSWER_PREFIXES = (
    "here are the answer strings",
    "here are the possible answer strings",
    "based on the provided facts",
    "based on the information provided",
    "the question asks",
    "these answer strings",
    "the answer strings",
    "note:",
    "explanation:",
)
_JUNK_ANSWER_TOKENS = (
    "answer strings",
    "facts section",
    "provided facts",
    "provided information",
    "please keep in mind",
)


def _extract_answer_from_sentence(text: str) -> str:
    m = re.search(
        r"\banswer is\b\s*(?:the\s+)?[\"'“”]?([^\"'“”,.;:\n]+(?:\s+[^\"'“”,.;:\n]+){0,7})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return ""
    candidate = m.group(1).strip().strip("\"'“”")
    return candidate.strip()


def _clean_answer_item(item: str) -> str:
    item = re.sub(r"^\s*\d+[\.\)]\s*", "", item).strip()
    item = item.strip("-* \t").strip("\"'“”").strip()
    if not item:
        return ""

    lower = item.lower()
    if "answer is" in lower:
        extracted = _extract_answer_from_sentence(item)
        if extracted:
            item = extracted
            lower = item.lower()

    if lower.startswith(_JUNK_ANSWER_PREFIXES):
        return ""
    if any(token in lower for token in _JUNK_ANSWER_TOKENS):
        return ""
    if "->" in item:
        return ""
    if _FREEBASE_ID_RE.fullmatch(item):
        return ""
    if _RELATION_PATH_RE.search(item):
        return ""
    if len(item.split()) > 14 and any(
        token in lower
        for token in ("question", "facts", "provided", "assume", "therefore", "based on")
    ):
        return ""

    return item.strip()


def _get_list_str(text: str) -> list[str]:
    lines = text.strip().split("\n")
    items = []
    for ln in lines:
        cleaned = _clean_answer_item(ln)
        if cleaned:
            items.append(cleaned)
    if items:
        return items

    cleaned_text = _clean_answer_item(text)
    return [cleaned_text] if cleaned_text else []


def hit(answer_list: list[str], result: str) -> bool:
    norm_res = _normalize(result)
    return any(_normalize(a) in norm_res for a in answer_list)


def reverse_hit(answer_list: list[str], result: str) -> bool:
    norm_answers = [_normalize(a) for a in answer_list]
    for item in _get_list_str(result):
        ni = _normalize(item)
        if ni and any(ni in na for na in norm_answers):
            return True
    return False


def compute_answer_f1(answer_list: list[str], result: str) -> float:
    def _unique_normalized(items: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            norm_item = _normalize(item)
            if not norm_item or norm_item in seen:
                continue
            seen.add(norm_item)
            unique.append(norm_item)
        return unique

    gold_items = _unique_normalized(answer_list)
    pred_items = _unique_normalized(_get_list_str(result))
    if not gold_items or not pred_items:
        return 0.0

    matched_gold: set[int] = set()
    true_positive = 0
    for pred in pred_items:
        match_idx = None
        for idx, gold in enumerate(gold_items):
            if idx in matched_gold:
                continue
            if pred in gold or gold in pred:
                match_idx = idx
                break
        if match_idx is not None:
            matched_gold.add(match_idx)
            true_positive += 1

    if true_positive == 0:
        return 0.0

    precision = true_positive / len(pred_items)
    recall = true_positive / len(gold_items)
    return 2.0 * precision * recall / (precision + recall)


def _embedding_prefix(args, ckpt: dict) -> str:
    if args.emb_prefix.strip():
        return args.emb_prefix.strip()
    if ckpt.get("emb_prefix"):
        return str(ckpt["emb_prefix"]).strip()
    return args.dataset_name


def _score_final_state(state, policy_model, device, args) -> float:
    active_idx = state.get_active_indices()
    if active_idx.numel() == 0:
        return float("-inf")
    p_c = state.p_c[active_idx]
    p_s = state.p_s[active_idx]
    mean_pcau = max(float(p_c.mean().item()), float(args.pcau_floor))
    pnon_norm = float(p_s.sum().item()) / max(float(state.node_feats.size(0)), 1.0)

    x, edge_index_local, active_idx_local = state.to_pyg_input()
    if x is None or active_idx_local is None or x.size(0) == 0:
        return float("-inf")
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    logodds = state.logodds_norm[active_idx_local].to(device)
    with torch.no_grad():
        _, _, log_value = policy_model(
            x, edge_index_local, batch, logodds,
            q_emb=state.q_emb,
            alpha_logodds=args.gflow_alpha,
            beta_gnn=args.gflow_beta,
        )
    return float(log_value.item()) + args.pcau_weight * math.log(mean_pcau) - args.lam * pnon_norm


def _state_key(state) -> tuple[int, ...]:
    active_idx = state.get_active_indices()
    return tuple(int(x) for x in active_idx.tolist())


def _top_action_candidates(state, policy_model, device, args) -> list[tuple[int | str, float]]:
    x, edge_index_local, active_idx_local = state.to_pyg_input()
    if x is None or active_idx_local is None or x.size(0) == 0:
        return []

    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    logodds = state.logodds_norm[active_idx_local].to(device)
    with torch.no_grad():
        action_logits, stop_logit, _ = policy_model(
            x, edge_index_local, batch, logodds,
            q_emb=state.q_emb,
            alpha_logodds=args.gflow_alpha,
            beta_gnn=args.gflow_beta,
        )

    leaf_mask_local = get_leaf_mask(edge_index_local, x.size(0))
    leaf_indices = leaf_mask_local.nonzero(as_tuple=True)[0]
    if leaf_indices.numel() == 0:
        return [("STOP", 0.0)]

    leaf_logits = (action_logits[leaf_indices] / args.gflow_temp).reshape(-1)
    stop_scaled = (stop_logit / args.gflow_temp).reshape(-1)[:1]
    all_logits = torch.cat([leaf_logits, stop_scaled], dim=0)
    probs = F.softmax(all_logits, dim=0)

    top_k = min(int(args.top_b2), int(probs.numel()))
    top_probs, top_idx = torch.topk(probs, k=top_k, largest=True, sorted=True)

    candidates: list[tuple[int | str, float]] = []
    stop_pos = int(leaf_indices.numel())
    for prob_t, idx_t in zip(top_probs, top_idx):
        prob = float(prob_t.item())
        idx = int(idx_t.item())
        log_pi = math.log(max(prob, 1e-12))
        if idx == stop_pos:
            candidates.append(("STOP", log_pi))
        else:
            local_leaf_idx = int(leaf_indices[idx].item())
            global_idx = int(active_idx_local[local_leaf_idx].item())
            candidates.append((global_idx, log_pi))
    return candidates


def _best_first_prune_search(state0, policy_model, device, args):
    init_reward = _score_final_state(state0, policy_model, device, args)
    frontier: list[tuple[float, int, object]] = []
    counter = count()
    heapq.heappush(frontier, (-init_reward, next(counter), state0))

    best_seen: dict[tuple[int, ...], float] = {_state_key(state0): init_reward}
    best_terminal_score = float("-inf")
    best_terminal_state = None
    best_frontier_score = init_reward
    best_frontier_state = state0
    expansions = 0

    while frontier and expansions < DEFAULT_SEARCH_BUDGET:
        neg_priority, _, state = heapq.heappop(frontier)
        current_priority = -neg_priority
        state_key = _state_key(state)
        if current_priority < best_seen.get(state_key, float("-inf")):
            continue

        state_reward = current_priority

        if state.K_active <= args.min_chunks:
            continue

        if current_priority > best_frontier_score:
            best_frontier_score = current_priority
            best_frontier_state = state

        expansions += 1
        candidates = _top_action_candidates(state, policy_model, device, args)
        if not candidates:
            if current_priority > best_terminal_score:
                best_terminal_score = current_priority
                best_terminal_state = state
            continue

        for action, _ in candidates:
            if action == "STOP":
                if state_reward > best_terminal_score:
                    best_terminal_score = state_reward
                    best_terminal_state = state
                continue

            child_state = state.remove(int(action))
            child_reward = _score_final_state(child_state, policy_model, device, args)
            priority = child_reward

            child_key = _state_key(child_state)
            if priority <= best_seen.get(child_key, float("-inf")):
                continue
            best_seen[child_key] = priority

            if child_state.K_active > args.min_chunks and priority > best_frontier_score:
                best_frontier_score = priority
                best_frontier_state = child_state

            heapq.heappush(frontier, (-priority, next(counter), child_state))

    if best_terminal_state is not None:
        return best_terminal_state
    if best_frontier_state is not None:
        return best_frontier_state
    return state0

def main():
    args = parse_caseflow_infer_args()
    setup_seed(args.seed)
    os.chdir(PROJECT_ROOT)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[CaseFlow-Infer] Device: {device}", flush=True)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    hidden      = ckpt["hidden"]
    profile_dim = ckpt["profile_dim"]
    cluster_feature_dim = ckpt.get("cluster_feature_dim", profile_dim)
    K           = ckpt["K"]
    tau         = ckpt["tau"]
    kmeans_iters = int(ckpt.get("kmeans_iters", 5))
    if args.sim_one is None:
        args.sim_one = float(ckpt.get("sim_one", 0.85))
    if args.k1_quantile is None:
        args.k1_quantile = float(ckpt.get("k1_quantile", 0.10))
    clusterer = str(ckpt.get("clusterer", "unknown"))
    print(f"[CaseFlow-Infer] Checkpoint: {args.checkpoint}", flush=True)
    print(
        f"[CaseFlow-Infer] clusterer={clusterer} K={K}, D={cluster_feature_dim}, "
        f"hidden={hidden}, iters={kmeans_iters}, "
        f"sim_one={args.sim_one:.3f}, k1_quantile={args.k1_quantile:.2f}",
        flush=True,
    )

    sbert = SentenceTransformer(args.sbert_model, device=str(device))
    embed_dim = sbert.get_sentence_embedding_dimension()
    print(f"[CaseFlow-Infer] SBERT: {args.sbert_model}  dim={embed_dim}", flush=True)

    emb_prefix = _embedding_prefix(args, ckpt)
    store = OfflineEmbeddingStore(emb_dir=args.emb_dir, device=device, prefix=emb_prefix)
    print(f"[CaseFlow-Infer] Embedding prefix: {emb_prefix}", flush=True)

    enc_path = Path(args.encoder_path)
    if enc_path.is_file():
        enc_ckpt = torch.load(str(enc_path), map_location=device)
        enc_args = enc_ckpt.get("args", {})
        enc_hidden = enc_ckpt.get("embed_dim", hidden)
        encoder  = CausalEncoder(
            in_channels=enc_hidden, hidden=enc_hidden,
            num_layers=enc_args.get("layers", 3), dropout=0.0,
        ).to(device)
        encoder.load_state_dict(enc_ckpt["model_state"], strict=False)
        print(f"[CaseFlow-Infer] Encoder: {args.encoder_path}", flush=True)
    else:
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder_path}")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    policy_model = PolicyFlowModel(
        hidden=hidden,
        gnn_layers=int(ckpt.get("gnn_layers", args.gnn_layers)),
        dropout=0.0,
    ).to(device)
    if "policy_model_state" not in ckpt:
        raise KeyError("Checkpoint missing policy_model_state")
    load_info = load_policy_model_state(policy_model, ckpt["policy_model_state"])
    if load_info["missing"] or load_info["skipped"]:
        print(
            "[CaseFlow-Infer] Policy checkpoint load: "
            f"missing={len(load_info['missing'])} "
            f"skipped={len(load_info['skipped'])}",
            flush=True,
        )
    policy_model.eval()
    datas = load_local_qa_split(args.dataset_name, args.split)
    print(f"[CaseFlow-Infer] {len(datas)} questions", flush=True)
    print(
        f"[CaseFlow-Infer] BestFirst: top_b2={args.top_b2}, "
        f"search_budget={DEFAULT_SEARCH_BUDGET}, min_chunks={args.min_chunks}",
        flush=True,
    )

    if args.max_questions > 0:
        datas = datas[:args.max_questions]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total_hit = 0
    total_f1 = 0.0
    n_done    = 0

    for data_item in tqdm(datas, desc="CaseFlow Infer"):
        question      = data_item.get("question", "")
        answer_texts  = parse_answers(data_item)
        topic_name    = get_topic_name_from_item(data_item)
        graph_triples, graph_label_map = extract_graph_payload(data_item)

        if not topic_name or not graph_triples:
            continue
        topic_display, display_graph_triples, _ = resolve_graph_triples_for_display(
            graph_triples, topic_name,
            label_map=graph_label_map,
            resolve_online=False,
        )

        z_q = sbert.encode([question], convert_to_tensor=True).to(device)[0]
        z_a = (
            sbert.encode(answer_texts, convert_to_tensor=True).to(device).mean(dim=0)
            if answer_texts
            else sbert.encode([topic_name], convert_to_tensor=True).to(device)[0]
        )

        graph_data, rel_groups, node_names = build_graph_with_llm_sampling(
            graph_triples, topic_name, store, z_q, z_a, device,
            question=question, args=args, max_hops=args.depth,
            resolve_display=False,
            label_map=graph_label_map,
        )
        if graph_data.x.size(0) <= 1 or not rel_groups:
            continue

        with torch.no_grad():
            z_c_all, z_s_all, alpha_c_all, alpha_s_all = encode_full_graph(
                graph_data, encoder, device
            )

        chunk_info_list: list = []
        global_offset:   int  = 0

        for hop, relation, tail_indices, parent_names_by_tail in rel_groups:
            if not tail_indices:
                continue

            idx_t     = torch.tensor(tail_indices, dtype=torch.long, device=device)
            z_c_t     = z_c_all[idx_t]
            z_s_t     = z_s_all[idx_t]
            alpha_c_t = alpha_c_all[idx_t]
            alpha_s_t = alpha_s_all[idx_t]
            tail_names_all = [node_names[i] for i in tail_indices]

            profile = build_profile(z_c_t, z_s_t, args.eps)

            with torch.no_grad():
                features = profile
                _centers, pi_soft, k_star = adaptive_spherical_kmeans_assign_with_soft_scores(
                    features,
                    max_k=K,
                    tau=tau,
                    n_iters=kmeans_iters,
                    eps=args.eps,
                    sim_one=args.sim_one,
                    k1_quantile=args.k1_quantile,
                )
                pi_hard = F.one_hot(k_star, num_classes=pi_soft.size(1)).float()

                logodds, logodds_norm, mean_z_c, active, p_c_raw, p_s_raw = compute_chunk_logodds(
                    pi_soft, pi_hard, alpha_c_t, alpha_s_t, z_c_t, eps=args.eps
                )

            K_act = int(active.sum().item())
            if K_act == 0:
                continue

            logodds_norm_act = logodds_norm[active]
            mean_z_c_act     = mean_z_c[active]
            p_c_raw_act      = p_c_raw[active]
            p_s_raw_act      = p_s_raw[active]

            tail_names_by_k: list = [[] for _ in range(pi_soft.size(1))]
            parent_names_by_k: list[set[str]] = [set() for _ in range(pi_soft.size(1))]
            for t_idx, k in enumerate(k_star.tolist()):
                if t_idx < len(tail_names_all):
                    tail_names_by_k[k].append(tail_names_all[t_idx])
                if t_idx < len(parent_names_by_tail):
                    parent_names_by_k[k].update(parent_names_by_tail[t_idx])

            active_k_indices  = active.nonzero(as_tuple=False).squeeze(-1).tolist()
            active_tail_names = [tail_names_by_k[k] for k in active_k_indices]
            active_parent_names = [
                sorted(parent_names_by_k[k]) for k in active_k_indices
            ]

            chunk_info_list.append({
                "hop":                  hop,
                "relation":             relation,
                "global_start":         global_offset,
                "K_act":                K_act,
                "mean_z_c":             mean_z_c_act.detach(),
                "logodds_norm":         logodds_norm_act.detach(),
                "p_c":                  p_c_raw_act.detach(),
                "p_s":                  p_s_raw_act.detach(),
                "tail_names_per_chunk": active_tail_names,
                "parent_names_per_chunk": active_parent_names,
            })
            global_offset += K_act

        if not chunk_info_list:
            continue

        state0 = build_chunk_graph_state(
            chunk_info_list,
            device,
            q_emb=z_q.detach(),
        )
        if state0 is None or state0.K_active == 0:
            continue

        best_state = _best_first_prune_search(state0, policy_model, device, args)

        full_facts = build_chunk_readout(
            chunk_info_list,
            best_state,
            topic_display,
            display_graph_triples,
            args.limit_llm_in,
        )
        prompt = question_prompt_with_knowledge.format(full_facts, question)
        response = run_llm(prompt, args).strip()

        is_hit = hit(answer_texts, response) or reverse_hit(answer_texts, response)
        f1 = compute_answer_f1(answer_texts, response)
        total_hit += int(is_hit)
        total_f1 += float(f1)
        n_done    += 1

        record = {
            "question": question,
            "topic":    topic_name,
            "result":   response,
            "gold":     answer_texts,
            "hit":      int(is_hit),
            "f1":       float(f1),
            "status":   "ok",
        }
        save_2_jsonl(args.output, record)

    if n_done > 0:
        acc = total_hit / n_done * 100
        avg_f1 = total_f1 / n_done * 100
        print(f"\n[CaseFlow-Infer] ===== Results =====", flush=True)
        print(f"  Dataset   : {args.dataset_name} / {args.split}", flush=True)
        print(f"  Questions : {n_done}", flush=True)
        print(f"  Hits      : {total_hit}", flush=True)
        print(f"  Accuracy  : {acc:.2f}%", flush=True)
        print(f"  F1        : {avg_f1:.2f}%", flush=True)
    else:
        print("[CaseFlow-Infer] No questions processed.", flush=True)

    print(f"[CaseFlow-Infer] Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
