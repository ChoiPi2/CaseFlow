
from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch_geometric.data import Data

from freebase import execute_sparql

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMB_DIR = str(PROJECT_ROOT / "embeddings")
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"

_HF_DATASET_NAMES = {
    "webqsp": os.environ.get("WEBQSP_HF_DATASET", "").strip(),
    "cwq":    os.environ.get("CWQ_HF_DATASET", "").strip(),
}

_LOCAL_DATASET_DIRS = {
    "webqsp": [
        DEFAULT_DATA_DIR / "webqsp",
    ],
    "cwq": [
        DEFAULT_DATA_DIR / "cwq",
    ],
    "metaqa3hop": [
        DEFAULT_DATA_DIR / "metaqa3hop",
    ],
}

_FB_ID_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
_FB_NAME_QUERY = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity ?name
WHERE {
  VALUES ?entity { %s }
  OPTIONAL { ?entity ns:type.object.name ?name . FILTER(lang(?name)='en') }
}
"""
_NAME_CACHE: dict[str, str] = {}


def _is_freebase_id(value: str) -> bool:
    return bool(_FB_ID_RE.match(str(value).strip()))


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _resolve_type_object_names(entity_values: list[str], batch_size: int = 100) -> dict[str, str]:
    result: dict[str, str] = {}
    pending = []

    for value in entity_values:
        raw = str(value).strip()
        if not raw:
            continue
        if not _is_freebase_id(raw):
            result[raw] = raw
            continue
        if raw in _NAME_CACHE:
            result[raw] = _NAME_CACHE[raw]
            continue
        pending.append(raw)

    for batch in _chunks(sorted(set(pending)), batch_size):
        bindings = " ".join(f"ns:{mid}" for mid in batch)
        rows = execute_sparql(_FB_NAME_QUERY % bindings)
        seen: set[str] = set()
        for row in rows:
            ent = row.get("entity", {}).get("value", "")
            ent = ent.replace("http://rdf.freebase.com/ns/", "").strip()
            if not ent:
                continue
            name = row.get("name", {}).get("value", "").strip()
            _NAME_CACHE[ent] = name if name else ent
            seen.add(ent)
        for mid in batch:
            if mid not in seen:
                _NAME_CACHE[mid] = mid

    for value in entity_values:
        raw = str(value).strip()
        if not raw:
            continue
        if raw in result:
            continue
        result[raw] = _NAME_CACHE.get(raw, raw)

    return result


def resolve_graph_triples_for_display(
    graph_triples: list,
    topic_name: str,
    label_map: Optional[dict[str, str]] = None,
    resolve_online: bool = True,
) -> tuple[str, list[list[str]], dict[str, str]]:
    entity_values = [topic_name]
    for triple in graph_triples:
        if len(triple) < 3:
            continue
        entity_values.extend([str(triple[0]).strip(), str(triple[2]).strip()])

    display_map = build_display_map(
        entity_values,
        label_map=label_map,
        resolve_online=resolve_online,
    )
    topic_display = display_map.get(topic_name, topic_name)

    display_triples: list[list[str]] = []
    for triple in graph_triples:
        if len(triple) < 3:
            continue
        h = str(triple[0]).strip()
        r = str(triple[1]).strip()
        t = str(triple[2]).strip()
        display_triples.append([
            display_map.get(h, h),
            r,
            display_map.get(t, t),
        ])

    return topic_display, display_triples, display_map


def extract_graph_payload(data_item: dict) -> tuple[list[list[str]], dict[str, str]]:
    graph_field = data_item.get("graph", [])
    if isinstance(graph_field, dict):
        triples = graph_field.get("triples", [])
        raw_map = graph_field.get("label_map", {})
        label_map = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                key = str(k).strip()
                if not key:
                    continue
                val = str(v).strip() if v is not None else ""
                label_map[key] = val or key
        return triples, label_map

    return graph_field, {}


def build_display_map(
    entity_values: list[str],
    label_map: Optional[dict[str, str]] = None,
    resolve_online: bool = True,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    label_map = label_map or {}

    for value in entity_values:
        raw = str(value).strip()
        if not raw or raw in merged:
            continue
        if raw in label_map and str(label_map[raw]).strip():
            merged[raw] = str(label_map[raw]).strip()
        elif not _is_freebase_id(raw):
            merged[raw] = raw

    if resolve_online:
        pending = [str(v).strip() for v in entity_values if str(v).strip() and str(v).strip() not in merged]
        if pending:
            merged.update(_resolve_type_object_names(pending))

    for value in entity_values:
        raw = str(value).strip()
        if raw and raw not in merged:
            merged[raw] = raw

    return merged


def load_local_qa_split(dataset_name: str, split: str) -> list[dict]:
    if dataset_name not in _LOCAL_DATASET_DIRS:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. "
            f"Available: {sorted(_LOCAL_DATASET_DIRS)}"
        )

    for local_dir in _LOCAL_DATASET_DIRS[dataset_name]:
        if not local_dir.is_dir():
            continue

        dataset_dict_file = local_dir / "dataset_dict.json"
        if dataset_dict_file.is_file():
            print(
                f"[Data] Loading {dataset_name} [{split}] from {local_dir} ...",
                flush=True,
            )
            ds_dict = load_from_disk(str(local_dir))
            ds = ds_dict[split]
            print(f"[Data] {len(ds)} items loaded.", flush=True)
            return list(ds)

        prefix = f"{dataset_name}"
        paths = sorted(local_dir.glob(f"{prefix}-{split}*.arrow"))
        paths = [p for p in paths if p.is_file()]
        if paths:
            datasets = [Dataset.from_file(str(p)) for p in paths]
            ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
            print(
                f"[Data] Loading {dataset_name} [{split}] from {local_dir} ...",
                flush=True,
            )
            print(f"[Data] {len(ds)} items loaded.", flush=True)
            return list(ds)

    hf_name = _HF_DATASET_NAMES.get(dataset_name)
    if not hf_name:
        searched = ", ".join(str(path) for path in _LOCAL_DATASET_DIRS[dataset_name])
        raise FileNotFoundError(
            f"[Data] Local split missing for {dataset_name} [{split}]. "
            f"Searched: {searched}. Set CWQ_HF_DATASET or WEBQSP_HF_DATASET to enable HF fallback."
        )
    print(f"[Data] Local split missing — fallback to {hf_name} [{split}] ...", flush=True)
    ds = load_dataset(hf_name, split=split)
    items = list(ds)
    print(f"[Data] {len(items)} items loaded from HF dataset.", flush=True)
    return items


                                                                               
                                 
                                                                               

class OfflineEmbeddingStore:

    def __init__(
        self,
        emb_dir: str = DEFAULT_EMB_DIR,
        device: Optional[torch.device] = None,
        prefix: str = "unified",
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        emb_dir = Path(emb_dir)

        print(f"[OfflineEmbeddingStore] Loading embeddings ({prefix}) → {device} ...", flush=True)

        entity_t   = torch.load(emb_dir / f"{prefix}_entity_embeddings.pt",   map_location=device)
        relation_t = torch.load(emb_dir / f"{prefix}_relation_embeddings.pt", map_location=device)

        self.entity_emb   = nn.Embedding.from_pretrained(entity_t,   freeze=True).to(device)
        self.relation_emb = nn.Embedding.from_pretrained(relation_t, freeze=True).to(device)

        with open(emb_dir / f"{prefix}_entity2idx.json",   encoding="utf-8") as f:
            self.entity2idx: dict[str, int] = json.load(f)
        with open(emb_dir / f"{prefix}_relation2idx.json", encoding="utf-8") as f:
            self.relation2idx: dict[str, int] = json.load(f)

        self.emb_dim    = entity_t.shape[1]        
        self._oov_total = 0

        print(
            f"[OfflineEmbeddingStore] Entities={entity_t.shape[0]:,}, "
            f"Relations={relation_t.shape[0]:,}, dim={self.emb_dim}",
            flush=True,
        )

    def lookup_entities(self, names: list[str]) -> torch.Tensor:
        result = torch.zeros(len(names), self.emb_dim, device=self.device)
        valid_pos, valid_idx = [], []
        for pos, name in enumerate(names):
            idx = self.entity2idx.get(name, -1)
            if idx >= 0:
                valid_pos.append(pos)
                valid_idx.append(idx)
            else:
                self._oov_total += 1
        if valid_idx:
            t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)
            result[valid_pos] = self.entity_emb(t)
        return result

    def lookup_relations(self, names: list[str]) -> torch.Tensor:
        result = torch.zeros(len(names), self.emb_dim, device=self.device)
        valid_pos, valid_idx = [], []
        for pos, name in enumerate(names):
            idx = self.relation2idx.get(name, -1)
            if idx >= 0:
                valid_pos.append(pos)
                valid_idx.append(idx)
            else:
                self._oov_total += 1
        if valid_idx:
            t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)
            result[valid_pos] = self.relation_emb(t)
        return result

                                                                               
                                               
                                                                               

def build_graph_with_llm_sampling(
    graph_triples: list,
    topic_name:    str,
    store:         OfflineEmbeddingStore,
    z_q:           torch.Tensor,
    z_a,
    device:        torch.device,
    question:      str,
    args,
    max_hops:      int = 2,
    resolve_display: bool = True,
    label_map:     Optional[dict[str, str]] = None,
) -> tuple[Data, list[tuple[int, str, list[int], list[list[str]]]], list[str]]:
    from freebase import sample_relations, sample_relations_distant

    if getattr(args, "width", 0) <= 0:
        raise ValueError("build_graph_with_llm_sampling requires args.width >= 1.")

    entity_values = [topic_name]
    for triple in graph_triples:
        if len(triple) < 3:
            continue
        entity_values.extend([str(triple[0]).strip(), str(triple[2]).strip()])
    display_map = build_display_map(
        entity_values,
        label_map=label_map,
        resolve_online=resolve_display,
    )
    topic_display = display_map.get(topic_name, topic_name)

    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for triple in graph_triples:
        h = str(triple[0]).strip()
        r = str(triple[1]).strip()
        t = str(triple[2]).strip()
        if h and r and t:
            adj[h].append((r, t))

    node_keys:   list[str]              = []
    node_names:  list[str]              = []
    node_hops:   list[int]              = []
    node_to_idx: dict[str, int]         = {}
    edges:       list[tuple[int, int]]  = []
    edge_relations: list[str]           = []
    rel_groups_map: dict[tuple[int, str], list[int]] = defaultdict(list)
    rel_parent_map: dict[tuple[int, str], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    def _add_node(name: str, hop: int) -> int:
        if name not in node_to_idx:
            idx = len(node_names)
            node_to_idx[name] = idx
            node_keys.append(name)
            node_names.append(name)
            node_names[idx] = display_map.get(name, name)
            node_hops.append(hop)
        return node_to_idx[name]

    topic_idx = _add_node(topic_name, 0)

    hop1_rel_tails: dict[str, list[str]] = defaultdict(list)
    for rel, tail in adj.get(topic_name, []):
        hop1_rel_tails[rel].append(tail)

    hop1_all_rels = list(hop1_rel_tails.keys())
    if len(hop1_all_rels) > args.width:
        selected_hop1 = sample_relations(question, topic_display, hop1_all_rels, args)
    else:
        selected_hop1 = hop1_all_rels

    frontier_by_rel: dict[str, list[str]] = {}
    graphs_state:    dict[str, dict]      = {}

    for rel in selected_hop1:
        tails = hop1_rel_tails.get(rel, [])
        for tail in tails:
            tail_idx = _add_node(tail, 1)
            edges.append((topic_idx, tail_idx))
            edge_relations.append(rel)
            rel_groups_map[(1, rel)].append(tail_idx)
        frontier_by_rel[rel] = tails
        tail_preview = "; ".join(display_map.get(t, t) for t in tails[:10])
        graphs_state[rel] = {
            "fact": f"The entity {topic_display} has relation {rel} with following entities: {tail_preview}",
        }

    for hop in range(2, max_hops + 1):
        next_relations: dict[str, dict] = {}

        for prev_rel in list(frontier_by_rel.keys()):
            hop2_rel_set: set[str] = set()
            for tail_name in frontier_by_rel[prev_rel]:
                for r2, _ in adj.get(tail_name, []):
                    hop2_rel_set.add(r2)
            hop2_rels = list(hop2_rel_set)
            fact = graphs_state.get(prev_rel, {}).get("fact", "").strip()
            if hop2_rels and fact:
                next_relations[prev_rel] = {"relation": hop2_rels, "fact": fact}

        if not next_relations:
            break

        total = sum(len(v["relation"]) for v in next_relations.values())
        if total > args.width:
            selected_chained = sample_relations_distant(question, topic_display, next_relations, args)
        else:
            selected_chained = [
                f"{r1}->{r2}"
                for r1, v in next_relations.items()
                for r2 in v["relation"]
            ]

        new_frontier: dict[str, list[str]] = {}
        new_graphs_state: dict[str, dict]  = {}

        for chained_rel in selected_chained:
            prev_rel, next_rel = chained_rel.rsplit("->", 1)
            new_tails: list[str] = []
            for src_name in frontier_by_rel.get(prev_rel, []):
                src_idx = node_to_idx.get(src_name)
                if src_idx is None:
                    continue
                for r2, tail_name in adj.get(src_name, []):
                    if r2 == next_rel:
                        tail_idx = _add_node(tail_name, hop)
                        edges.append((src_idx, tail_idx))
                        edge_relations.append(next_rel)
                        rel_groups_map[(hop, chained_rel)].append(tail_idx)
                        rel_parent_map[(hop, chained_rel)][tail_idx].add(src_name)
                        new_tails.append(tail_name)
            if new_tails:
                new_frontier[chained_rel] = new_tails
                preview = "; ".join(display_map.get(t, t) for t in new_tails[:10])
                new_graphs_state[chained_rel] = {
                    "fact": f"The entity {topic_display} has relation {chained_rel} with following entities: {preview}",
                }

        frontier_by_rel = new_frontier
        graphs_state    = new_graphs_state

    N   = len(node_names)
    x = store.lookup_entities(node_keys)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_z_r = store.lookup_relations(edge_relations)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_z_r = torch.zeros((0, store.emb_dim), device=device)

    if z_q.dim() == 1:
        z_q = z_q.unsqueeze(0)
    z_q = z_q.to(device)

    data       = Data(x=x, edge_index=edge_index, edge_z_r=edge_z_r, z_q=z_q)
    data.batch = torch.zeros(N, dtype=torch.long, device=device)

    if z_a is not None:
        if z_a.dim() == 1:
            z_a = z_a.unsqueeze(0)
        data.z_a = z_a.to(device)

    rel_groups: list[tuple[int, str, list[int], list[list[str]]]] = []
    for (hop, rel), tail_indices in rel_groups_map.items():
        seen:   set[int]  = set()
        unique: list[int] = []
        for idx in tail_indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        if unique:
            parent_names_by_tail = [
                sorted(rel_parent_map[(hop, rel)].get(idx, set()))
                for idx in unique
            ]
            rel_groups.append((hop, rel, unique, parent_names_by_tail))

    return data, rel_groups, node_names


def get_topic_name_from_item(data_item: dict) -> str | None:
    raw = data_item.get("q_entity", data_item.get("topic_entity", None))
    if raw is None:
        return None
    if isinstance(raw, dict):
        vals = [v for v in raw.values() if isinstance(v, str) and v.strip()]
        return vals[0].strip() if vals else None
    if isinstance(raw, list):
        for v in raw:
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                names = [x for x in v.values() if isinstance(x, str) and x.strip()]
                if names:
                    return names[0].strip()
        return None
    return str(raw).strip() or None


def parse_answers(data_item: dict) -> list[str]:
    raw = data_item.get("answer", data_item.get("answers", []))
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    texts = []
    for a in raw:
        if isinstance(a, dict):
            name = (a.get("entity_name") or a.get("answer_argument") or "").strip()
            if name:
                texts.append(name)
        elif str(a).strip():
            texts.append(str(a).strip())
    return texts
