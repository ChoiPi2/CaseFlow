from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from torch.optim import Adam
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from torch_geometric.data import Data

from model import CausalEncoder
from losses import pretrain_step_Ls_total, pretrain_step_mgda_Ls_total
from opts import parse_args, print_args, setup_seed

from freebase import sample_relations, sample_relations_distant
from graph_builder import OfflineEmbeddingStore

def _normalize_entity_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _unique_answer_texts(answer_entities: list[str]) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for answer in answer_entities:
        text = str(answer).strip()
        key = _normalize_entity_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(text)
    return uniq


def _resolve_answer_indices(
    node_to_idx:     dict,
    answer_entities: list,
    entity_text_map: dict | None = None,
) -> set[int]:
    ans_norms = {
        _normalize_entity_key(a)
        for a in answer_entities
        if _normalize_entity_key(a)
    }
    if not ans_norms:
        return set()

    answer_indices: set[int] = set()
    for node_name, idx in node_to_idx.items():
        raw_key = _normalize_entity_key(node_name)
        disp_key = _normalize_entity_key(entity_text_map.get(node_name, "")) if entity_text_map else ""
        if raw_key in ans_norms or disp_key in ans_norms:
            answer_indices.add(idx)

    if answer_indices or not entity_text_map:
        return answer_indices

    text_to_mid = {}
    for mid, text in entity_text_map.items():
        key = _normalize_entity_key(text)
        if key and key not in text_to_mid:
            text_to_mid[key] = mid

    for ans_key in ans_norms:
        mid = text_to_mid.get(ans_key)
        if mid and mid in node_to_idx:
            answer_indices.add(node_to_idx[mid])

    return answer_indices


def _find_answer_path_masks(
    node_to_idx:     dict,
    edges:           list,
    topic_idx:       int,
    answer_entities: list,
    num_nodes:       int,
    entity_text_map: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from collections import deque

    zero = torch.zeros(num_nodes, dtype=torch.bool)

    answer_indices = _resolve_answer_indices(
        node_to_idx,
        answer_entities,
        entity_text_map=entity_text_map,
    )
    if not answer_indices:
        return zero.clone(), zero.clone(), zero.clone()


    fwd: dict[int, list[int]] = defaultdict(list)
    rev: dict[int, list[int]] = defaultdict(list)
    for src, dst in edges:
        fwd[src].append(dst)
        rev[dst].append(src)

    reachable_from_topic: set[int] = {topic_idx}
    queue = deque([topic_idx])
    while queue:
        node = queue.popleft()
        for nb in fwd[node]:
            if nb not in reachable_from_topic:
                reachable_from_topic.add(nb)
                queue.append(nb)

    can_reach_answer: set[int] = set()
    queue = deque(answer_indices)
    while queue:
        node = queue.popleft()
        if node in can_reach_answer:
            continue
        can_reach_answer.add(node)
        for nb in rev[node]:
            if nb not in can_reach_answer:
                queue.append(nb)

    support_nodes = reachable_from_topic & can_reach_answer
    if not support_nodes:
        return zero.clone(), zero.clone(), zero.clone()

    answer_nodes = {idx for idx in answer_indices if idx in support_nodes}

    support_mask = torch.zeros(num_nodes, dtype=torch.bool)
    answer_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for idx in support_nodes:
        support_mask[idx] = True
    for idx in answer_nodes:
        answer_mask[idx] = True

    anchor_mask = support_mask.clone()
    anchor_mask[topic_idx] = False
    anchor_mask &= ~answer_mask

    return support_mask, anchor_mask, answer_mask


def _filter_rejected_entities(
    rejected_entities: list[str],
    selected_entities: set[str],
    support_entities: set[str],
    answer_entities: list[str],
    entity_text_map: dict[str, str] | None = None,
) -> list[str]:
    entity_text_map = entity_text_map or {}
    protected_norms: set[str] = set()

    def _protect(name: str) -> None:
        if not name:
            return
        protected_norms.add(_normalize_entity_key(name))
        display = entity_text_map.get(name)
        if display:
            protected_norms.add(_normalize_entity_key(display))

    for name in selected_entities:
        _protect(name)
    for name in support_entities:
        _protect(name)
    for answer in answer_entities:
        _protect(answer)

    text_to_mid = {}
    for mid, text in entity_text_map.items():
        key = _normalize_entity_key(text)
        if key and key not in text_to_mid:
            text_to_mid[key] = mid
    for answer in answer_entities:
        mid = text_to_mid.get(_normalize_entity_key(answer))
        if mid:
            _protect(mid)

    filtered: list[str] = []
    seen: set[str] = set()
    for name in rejected_entities:
        display = entity_text_map.get(name, name)
        norm_name = _normalize_entity_key(name)
        norm_display = _normalize_entity_key(display)
        dedupe_key = norm_name or norm_display
        if not dedupe_key or dedupe_key in seen:
            continue
        if name in selected_entities:
            continue
        if norm_name in protected_norms or norm_display in protected_norms:
            continue
        seen.add(dedupe_key)
        filtered.append(name)

    return filtered


def build_subgraph_with_negatives(
    graph_triples:   list,
    topic_id:        str,
    topic_name:      str,
    question:        str,
    store:           OfflineEmbeddingStore,
    z_q:             torch.Tensor,
    z_a:             torch.Tensor,
    device:          torch.device,
    args,
    entity_text_map: dict[str, str] | None = None,
    answer_entities: list[str]      | None = None,
) -> tuple[Data | None, torch.Tensor]:
    if entity_text_map is None:
        entity_text_map = {}

    def _display(entity_id: str) -> str:
        return entity_text_map.get(entity_id, entity_id)

    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for triple in graph_triples:
        h, r, t = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
        if h and r and t:
            adj[h].append((r, t))

    hop1_rel_tails: dict[str, list[str]] = defaultdict(list)
    for rel, tail in adj.get(topic_id, []):
        hop1_rel_tails[rel].append(tail)

    hop1_all_rels = list(hop1_rel_tails.keys())
    if not hop1_all_rels:
        return None, torch.zeros(0, store.emb_dim, device=device)

    if len(hop1_all_rels) > args.width:
        selected_hop1 = sample_relations(question, topic_name, hop1_all_rels, args)
    else:
        selected_hop1 = hop1_all_rels

    selected_hop1_set = set(selected_hop1)
    rejected_entities: list[str] = []

    for rel, tails in hop1_rel_tails.items():
        if rel not in selected_hop1_set:
            rejected_entities.extend(tails)

    node_names:     list[str]             = []
    node_to_idx:    dict[str, int]        = {}
    edges:          list[tuple[int, int]] = []
    edge_relations: list[str]             = []

    def _add_node(name: str) -> int:
        if name not in node_to_idx:
            idx = len(node_names)
            node_to_idx[name] = idx
            node_names.append(name)
        return node_to_idx[name]

    topic_idx = _add_node(topic_id)

    frontier_by_rel: dict[str, list[str]] = {}
    graphs_state:    dict[str, dict]      = {}

    for rel in selected_hop1:
        tails = hop1_rel_tails.get(rel, [])
        for tail in tails:
            tail_idx = _add_node(tail)
            edges.append((topic_idx, tail_idx))
            edge_relations.append(rel)
        frontier_by_rel[rel] = tails
        preview = "; ".join(_display(t) for t in tails[:10])
        graphs_state[rel] = {
            "fact": f"The entity {topic_name} has relation {rel} with following entities: {preview}",
        }

    for hop in range(2, args.depth + 1):
        next_relations: dict[str, dict] = {}
        for prev_rel in list(frontier_by_rel.keys()):
            hop2_rels: set[str] = set()
            for tail_name in frontier_by_rel[prev_rel]:
                for r2, _ in adj.get(tail_name, []):
                    hop2_rels.add(r2)
            fact = graphs_state.get(prev_rel, {}).get("fact", "").strip()
            if hop2_rels and fact:
                next_relations[prev_rel] = {"relation": list(hop2_rels), "fact": fact}

        if not next_relations:
            break

        total = sum(len(v["relation"]) for v in next_relations.values())
        if total > args.width:
            selected_chained = sample_relations_distant(question, topic_name,
                                                        next_relations, args)
        else:
            selected_chained = [
                f"{r1}->{r2}"
                for r1, v in next_relations.items()
                for r2 in v["relation"]
            ]

        selected_chained_set = set(selected_chained)

        for prev_rel, v in next_relations.items():
            for r2 in v["relation"]:
                chained = f"{prev_rel}->{r2}"
                if chained not in selected_chained_set:
                    for src_name in frontier_by_rel.get(prev_rel, []):
                        for rel2, tail_name in adj.get(src_name, []):
                            if rel2 == r2:
                                rejected_entities.append(tail_name)

        new_frontier: dict[str, list[str]] = {}
        for chained_rel in selected_chained:
            prev_rel, next_rel = chained_rel.rsplit("->", 1)
            new_tails: list[str] = []
            for src_name in frontier_by_rel.get(prev_rel, []):
                src_idx = node_to_idx.get(src_name)
                if src_idx is None:
                    continue
                for r2, tail_name in adj.get(src_name, []):
                    if r2 == next_rel:
                        tail_idx = _add_node(tail_name)
                        edges.append((src_idx, tail_idx))
                        edge_relations.append(next_rel)
                        new_tails.append(tail_name)
            if new_tails:
                new_frontier[chained_rel] = new_tails
                preview = "; ".join(_display(t) for t in new_tails[:10])
                graphs_state[chained_rel] = {
                    "fact": f"The entity {topic_name} has relation {chained_rel} with following entities: {preview}",
                }

        frontier_by_rel = new_frontier

    if len(node_names) < 2:
        return None, torch.zeros(0, store.emb_dim, device=device)

    N = len(node_names)
    x = store.lookup_entities(node_names)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_z_r = store.lookup_relations(edge_relations)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_z_r = torch.zeros((0, store.emb_dim), device=device)

    _z_q = z_q.unsqueeze(0) if z_q.dim() == 1 else z_q
    _z_a = z_a.unsqueeze(0) if z_a.dim() == 1 else z_a

    data = Data(x=x, edge_index=edge_index, edge_z_r=edge_z_r,
                z_q=_z_q.to(device), z_a=_z_a.to(device))
    data.batch = torch.zeros(N, dtype=torch.long, device=device)
    data.answer_count = torch.tensor([_z_a.size(0)], dtype=torch.long, device=device)

    support_mask, anchor_mask, answer_mask = _find_answer_path_masks(
        node_to_idx, edges, topic_idx,
        answer_entities or [], N,
        entity_text_map=entity_text_map,
    )
    data.path_mask = support_mask.to(device)
    data.support_mask = support_mask.to(device)
    data.anchor_mask = anchor_mask.to(device)
    data.answer_mask = answer_mask.to(device)

    selected_entities = set(node_names)
    support_entities = {
        node_names[idx] for idx, keep in enumerate(support_mask.tolist()) if keep
    }
    rejected_entities = _filter_rejected_entities(
        rejected_entities,
        selected_entities=selected_entities,
        support_entities=support_entities,
        answer_entities=answer_entities or [],
        entity_text_map=entity_text_map,
    )
    if rejected_entities:
        z_neg = store.lookup_entities(rejected_entities)
    else:
        z_neg = torch.zeros(0, store.emb_dim, device=device)

    return data, z_neg


      


def main():
    args = parse_args()
    print_args(args)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        memory_device = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(0.9, memory_device)
    device_label = f"cuda:{args.gpu_id}" if device.type == "cuda" else "cpu"
    print(f"[CaseFlow Encoder] Device: {device_label}", flush=True)

    store = OfflineEmbeddingStore(emb_dir=args.emb_dir, device=device,
                                  prefix=getattr(args, "emb_prefix", "unified"))
    embed_dim = store.emb_dim

    print(f"[CaseFlow Encoder] Loading embedder: {args.sbert_model}", flush=True)
    embedder = SentenceTransformer(args.sbert_model, device=str(device))

    def text_to_tensor(texts: list[str]) -> torch.Tensor:
        return embedder.encode(texts, convert_to_tensor=True).to(device)

    if args.hidden != embed_dim:
        print(f"[CaseFlow Encoder] Overriding --hidden {args.hidden} → {embed_dim}", flush=True)
        args.hidden = embed_dim

    model = CausalEncoder(
        in_channels=embed_dim,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop_out,
        att_temperature=getattr(args, "att_temperature", 1.0),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[CaseFlow Encoder] CausalEncoder loaded (InfoNCE mode).", flush=True)

    start_q = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_q = ckpt.get("question_idx", 0)
        print(f"[CaseFlow Encoder] Resumed from {args.resume}, question {start_q}", flush=True)

    os.chdir(PROJECT_ROOT)
    _ds_name = getattr(args, "pretrain_dataset", "cwq")
    if _ds_name.endswith(".jsonl") or os.path.isfile(_ds_name):
        import json as _json
        print(f"[CaseFlow Encoder] Loading local jsonl: {_ds_name} ...", flush=True)
        with open(_ds_name, encoding="utf-8") as _f:
            datas = [_json.loads(l) for l in _f]
    else:
        _hf_map = {
            "cwq":    os.environ.get("CWQ_HF_DATASET", "").strip(),
            "webqsp": os.environ.get("WEBQSP_HF_DATASET", "").strip(),
        }
        _hf_name = _hf_map.get(_ds_name, _ds_name)
        if not _hf_name:
            raise FileNotFoundError(
                "No local jsonl was provided and no HF dataset id is configured. "
                "Set CWQ_HF_DATASET or WEBQSP_HF_DATASET, or pass a local .jsonl path."
            )
        print(f"[CaseFlow Encoder] Loading {_hf_name} [train] ...", flush=True)
        ds = load_dataset(_hf_name, split="train")
        datas = list(ds)
    if args.max_questions > 0:
        datas = datas[:args.max_questions]
    print(f"[CaseFlow Encoder] Dataset: {_ds_name}, {len(datas)} questions", flush=True)

    print("=== Start CaseFlow Encoder InfoNCE Pre-training ===", flush=True)
    train_fn = pretrain_step_mgda_Ls_total if args.train_model == "infonce_mgda" else pretrain_step_Ls_total

    for epoch in range(args.epochs):
        total_loss_epoch = 0.0
        n_trained = 0
        graph_buffer: list = []

        def _flush(buf: list) -> float:
            if not buf:
                return 0.0
            t, _, _, _, _ = train_fn(model, optimizer, buf, device, args)
            return t

        for q_idx, item in enumerate(tqdm(datas, desc=f"Epoch {epoch+1}")):
            if q_idx < start_q:
                continue

            question = item.get("question", "")
            graph_triples = item.get("graph_mid") or item.get("graph", [])
            entity_text_map = item.get("entity_mid_map", {}) if item.get("graph_mid") else {}

            topic_entities = (
                item.get("topic_entity_mid")
                or item.get("q_entity_mid")
                or item.get("topic_entity")
                or item.get("q_entity", [])
            )
            topic_display_entities = item.get("topic_entity") or item.get("q_entity", [])
            raw_answers = item.get("answer") or item.get("a_entity", [])

            if not graph_triples or not topic_entities:
                continue

            answer_texts = _unique_answer_texts(raw_answers)
            if not answer_texts:
                continue
            z_a = text_to_tensor(answer_texts)
            z_q = text_to_tensor([question])[0]

            topic_id = topic_entities[0] if isinstance(topic_entities, list) \
                else str(topic_entities)
            topic_name = (
                topic_display_entities[0]
                if isinstance(topic_display_entities, list) and topic_display_entities
                else entity_text_map.get(topic_id, str(topic_id))
            )

            try:
                data, z_neg = build_subgraph_with_negatives(
                    graph_triples, topic_id, topic_name, question,
                    store, z_q, z_a, device, args,
                    entity_text_map=entity_text_map,
                    answer_entities=raw_answers,
                )
            except Exception:
                continue

            if data is None:
                continue

            graph_buffer.append((data, z_neg))

            if len(graph_buffer) >= args.batch_size:
                t_loss = _flush(graph_buffer)
                total_loss_epoch += t_loss
                n_trained += 1
                graph_buffer = []

            if (args.checkpoint_interval > 0
                    and (q_idx + 1) % args.checkpoint_interval == 0):
                ckpt_path = args.save_path.replace(".pth", f"_ckpt{q_idx+1}.pth")
                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "question_idx": q_idx + 1}, ckpt_path)
                print(f"[CaseFlow Encoder] Checkpoint: {ckpt_path}", flush=True)

        if graph_buffer:
            t_loss = _flush(graph_buffer)
            total_loss_epoch += t_loss
            n_trained += 1

        avg = total_loss_epoch / max(n_trained, 1)
        print(f"[CaseFlow Encoder] Epoch {epoch+1} | Avg Loss: {avg:.4f} | "
              f"Batches: {n_trained}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "embed_dim": embed_dim}, args.save_path)
    print(f"[CaseFlow Encoder] Saved → {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
