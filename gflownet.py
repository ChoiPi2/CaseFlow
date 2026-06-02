from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from rollout import STOP_ACTION
from rollout import get_leaf_mask, reconstruct_graph, _forward_single, _mix_with_uniform


                                                                               

class PolicyFlowModel(nn.Module):

    def __init__(
        self,
        hidden: int,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden

        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(GCNConv(hidden, hidden))

        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden * 2, 1)
        self.stop_head = nn.Linear(hidden * 2, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        nn.init.zeros_(self.stop_head.weight)
        nn.init.zeros_(self.stop_head.bias)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        h_chunk: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        logodds: torch.Tensor,
        q_emb: torch.Tensor | None = None,
        alpha_logodds: float = 1.0,
        beta_gnn: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = h_chunk
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if q_emb is None:
            q_graph = torch.zeros(
                (n_graphs, self.hidden),
                dtype=h_chunk.dtype,
                device=h_chunk.device,
            )
        else:
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            if q_emb.size(0) == 1 and n_graphs > 1:
                q_emb = q_emb.expand(n_graphs, -1)
            if q_emb.size(0) != n_graphs:
                raise ValueError(
                    f"q_emb batch ({q_emb.size(0)}) must match graph batch ({n_graphs})."
                )
            q_graph = q_emb.to(device=h_chunk.device, dtype=h_chunk.dtype)

        q_graph = self.query_proj(q_graph)

        for conv in self.gnn_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        q_node = q_graph[batch]
        x_q = torch.cat([x, q_node], dim=-1)
        structural_score = self.policy_head(x_q).squeeze(-1)
        action_logits = alpha_logodds * logodds + beta_gnn * structural_score

        graph_repr = global_mean_pool(x, batch)
        graph_q = torch.cat([graph_repr, q_graph], dim=-1)
        stop_logit = self.stop_head(graph_q).squeeze(-1)
        log_value = self.value_head(graph_q).squeeze(-1)

        return action_logits, stop_logit, log_value


def load_policy_model_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, object]:
    own_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}

    for key, value in state_dict.items():
        if key not in own_state:
            skipped[key] = "unexpected"
            continue

        target = own_state[key]
        if target.shape == value.shape:
            filtered[key] = value
            continue

        skipped[key] = f"shape {tuple(value.shape)} -> {tuple(target.shape)}"

    missing = [key for key in own_state.keys() if key not in filtered]
    model.load_state_dict(filtered, strict=False)
    return {
        "missing": missing,
        "skipped": skipped,
    }


                                                                                

class ChunkGraphState:

    def __init__(
        self,
        node_feats:   torch.Tensor,
        logodds_norm: torch.Tensor,
        p_c:          torch.Tensor,
        p_s:          torch.Tensor,
        hop_ids:      torch.Tensor | None,
        edge_index:   torch.Tensor,
        tail_names:   list,
        q_emb:        torch.Tensor | None,
        device:       torch.device,
    ):
        self.node_feats   = node_feats
        self.logodds_norm = logodds_norm
        self.p_c          = p_c
        self.p_s          = p_s
        self.hop_ids      = hop_ids
        self.edge_index   = edge_index
        self.tail_names   = tail_names
        self.q_emb        = q_emb
        self.device       = device
        self.active = torch.ones(
            node_feats.size(0), dtype=torch.bool, device=device
        )

    def clone(self) -> "ChunkGraphState":
        s = ChunkGraphState.__new__(ChunkGraphState)
        s.node_feats   = self.node_feats
        s.logodds_norm = self.logodds_norm
        s.p_c          = self.p_c
        s.p_s          = self.p_s
        s.hop_ids      = self.hop_ids
        s.edge_index   = self.edge_index
        s.tail_names   = self.tail_names
        s.q_emb        = self.q_emb
        s.device       = self.device
        s.active       = self.active.clone()
        return s

    @property
    def K_active(self) -> int:
        return int(self.active.sum().item())

    def get_active_indices(self) -> torch.Tensor:
        return self.active.nonzero(as_tuple=False).squeeze(-1)

    def get_leaves(self) -> torch.Tensor:
        if self.edge_index.size(1) == 0:
            return self.get_active_indices()

        src, dst = self.edge_index[0], self.edge_index[1]
        active_edge = self.active[src] & self.active[dst]
        has_child   = torch.zeros(
            self.active.size(0), dtype=torch.bool, device=self.device
        )
        active_src = src[active_edge]
        if active_src.numel() > 0:
            has_child.scatter_(
                0, active_src,
                torch.ones_like(active_src, dtype=torch.bool)
            )
        return (self.active & ~has_child).nonzero(as_tuple=False).squeeze(-1)

    def remove(self, global_idx: int) -> "ChunkGraphState":
        s = self.clone()
        s.active[global_idx] = False
        return s

    def to_pyg_input(self):
        active_idx = self.get_active_indices()
        K = active_idx.numel()
        if K == 0:
            return None, None, None

        x = self.node_feats[active_idx]                         

        if self.edge_index.size(1) > 0:
            src, dst  = self.edge_index[0], self.edge_index[1]
            mask      = self.active[src] & self.active[dst]
            filtered  = self.edge_index[:, mask]
            remap     = torch.full(
                (self.active.size(0),), -1, dtype=torch.long, device=self.device
            )
            remap[active_idx] = torch.arange(K, device=self.device)
            edge_index_local  = remap[filtered]
        else:
            edge_index_local = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )

        return x, edge_index_local, active_idx


                                                                                

def build_chunk_graph_state(
    chunk_info_list: list,
    device:          torch.device,
    q_emb:           torch.Tensor | None = None,
) -> "ChunkGraphState | None":
    if not chunk_info_list:
        return None

    all_mean_z_c     = []
    all_logodds_norm = []
    all_p_c          = []
    all_p_s          = []
    all_hop_ids      = []
    all_tail_names   = []

    for info in chunk_info_list:
        all_mean_z_c.append(info["mean_z_c"])
        all_logodds_norm.append(info["logodds_norm"])
        all_p_c.append(info["p_c"])
        all_p_s.append(info["p_s"])
        all_hop_ids.append(
            torch.full(
                (int(info["K_act"]),),
                int(info["hop"]),
                dtype=torch.long,
            )
        )
        all_tail_names.extend(info["tail_names_per_chunk"])

    node_feats   = torch.cat(all_mean_z_c, dim=0).to(device)
    logodds_norm = torch.cat(all_logodds_norm, dim=0).to(device)
    p_c          = torch.cat(all_p_c, dim=0).to(device)
    p_s          = torch.cat(all_p_s, dim=0).to(device)
    hop_ids      = torch.cat(all_hop_ids, dim=0).to(device)

    def _norm_name(name: str) -> str:
        return str(name).strip().lower()

    entity_to_chunk_indices: dict[str, set[int]] = {}
    chunk_hop_by_idx: dict[int, int] = {}

    for info in chunk_info_list:
        hop = info["hop"]
        for local_idx in range(info["K_act"]):
            global_idx = info["global_start"] + local_idx
            chunk_hop_by_idx[global_idx] = hop
            for tail_name in info["tail_names_per_chunk"][local_idx]:
                norm_name = _norm_name(tail_name)
                if not norm_name:
                    continue
                entity_to_chunk_indices.setdefault(norm_name, set()).add(global_idx)

    edge_set: set[tuple[int, int]] = set()
    has_parent_names = any(
        "parent_names_per_chunk" in info for info in chunk_info_list
    )

    if has_parent_names:
        for info in chunk_info_list:
            child_hop = info["hop"]
            parent_names_per_chunk = info.get(
                "parent_names_per_chunk",
                [[] for _ in range(info["K_act"])],
            )
            for local_idx in range(info["K_act"]):
                child_idx = info["global_start"] + local_idx
                for parent_name in parent_names_per_chunk[local_idx]:
                    norm_parent = _norm_name(parent_name)
                    if not norm_parent:
                        continue
                    for parent_idx in entity_to_chunk_indices.get(norm_parent, ()):
                        if chunk_hop_by_idx.get(parent_idx) == child_hop - 1 and parent_idx != child_idx:
                            edge_set.add((parent_idx, child_idx))
    else:
                                                                          
                                              
        for info_i in chunk_info_list:
            if info_i["hop"] != 1:
                continue
            prefix = info_i["relation"] + "->"
            for info_j in chunk_info_list:
                if info_j["hop"] != 2:
                    continue
                if not info_j["relation"].startswith(prefix):
                    continue
                for k1 in range(info_i["K_act"]):
                    for k2 in range(info_j["K_act"]):
                        edge_set.add((
                            info_i["global_start"] + k1,
                            info_j["global_start"] + k2,
                        ))

    if edge_set:
        edges_src, edges_dst = zip(*sorted(edge_set))
        edge_index = torch.tensor(
            [edges_src, edges_dst], dtype=torch.long, device=device
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return ChunkGraphState(
        node_feats   = node_feats,
        logodds_norm = logodds_norm,
        p_c          = p_c,
        p_s          = p_s,
        hop_ids      = hop_ids,
        edge_index   = edge_index,
        tail_names   = all_tail_names,
        q_emb        = q_emb.detach().to(device) if q_emb is not None else None,
        device       = device,
    )

                                                                                

def _build_log_pi(
    action_logits: torch.Tensor,            
    stop_logit:    torch.Tensor,           
    leaf_mask_sub: torch.Tensor,                 
    chosen_sub_idx: int | None,                             
    temperature: float = 1.0,
    rand_explore_prob: float = 0.0,
) -> torch.Tensor:
    leaf_indices = leaf_mask_sub.nonzero(as_tuple=True)[0]
    if leaf_indices.numel() == 0:
        return torch.tensor(0.0, device=action_logits.device)

    leaf_logits = action_logits[leaf_indices] / temperature
    stop_scaled = stop_logit / temperature
    all_logits = torch.cat([leaf_logits, stop_scaled.unsqueeze(0)])
    base_probs = F.softmax(all_logits, dim=0)
    probs = _mix_with_uniform(base_probs, rand_explore_prob)
    log_probs = torch.log(probs.clamp_min(1e-12))

    if chosen_sub_idx is None:
        return log_probs[-1]

    pos = (leaf_indices == chosen_sub_idx).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return torch.tensor(0.0, device=action_logits.device)
    return log_probs[int(pos[0].item())]


def subtb_loss(
    batch_trajs:   list[dict],
    policy_model,
    device:        torch.device,
    alpha_logodds: float = 1.0,
    beta_gnn:      float = 1.0,
    temperature:   float = 1.0,
    lambda_non:    float = 0.1,
    pcau_floor:    float = 0.05,
    pcau_weight:   float = 1.0,
    rand_explore_prob: float = 0.0,
) -> torch.Tensor:
    def _shape_terms(p_c_sub: torch.Tensor, p_s_sub: torch.Tensor, initial_num_chunks: int):
        if p_c_sub.numel() == 0 or p_s_sub.numel() == 0:
            mean_pcau = torch.tensor(pcau_floor, device=device)
            pnon_norm = torch.tensor(0.0, device=device)
            return mean_pcau, pnon_norm

        mean_pcau = p_c_sub.mean().clamp_min(pcau_floor)
        denom = max(float(initial_num_chunks), 1.0)
        pnon_norm = p_s_sub.sum() / denom
        return mean_pcau, pnon_norm

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for traj in batch_trajs:
        h_all = traj["h_chunk_all"]
        ei_init = traj["edge_index_init"]
        logodds_all = traj["logodds_all"]
        p_c_all = traj["p_c_all"]
        p_s_all = traj["p_s_all"]
        actions = traj["actions"]
        terminal_vllm = float(traj.get("terminal_vllm", 0.1))
        initial_num_chunks = int(traj.get("initial_num_chunks", h_all.size(0)))
        traj_rand_explore = float(traj.get("rand_explore_prob", rand_explore_prob))
        q_emb = traj.get("q_emb")

        if not actions:
            continue

        num_nodes = h_all.size(0)
        active_mask = torch.ones(num_nodes, dtype=torch.bool)
        log_rewards: list[torch.Tensor] = []
        log_pis: list[torch.Tensor] = []

        for action in actions:
            h_sub, ei_sub, lo_sub, index_map = reconstruct_graph(
                h_all, ei_init, active_mask, logodds_all, device
            )

            if h_sub.size(0) == 0:
                break

            active_idx = active_mask.nonzero(as_tuple=True)[0]
            p_c_sub = p_c_all[active_idx].to(device)
            p_s_sub = p_s_all[active_idx].to(device)

            action_logits, stop_logit, log_v_pred = _forward_single(
                policy_model, h_sub, ei_sub, lo_sub, alpha_logodds, device, beta_gnn, q_emb=q_emb
            )
            mean_pcau, pnon_norm = _shape_terms(p_c_sub, p_s_sub, initial_num_chunks)
            nonterminal_log_r = (
                log_v_pred
                + pcau_weight * torch.log(mean_pcau)
                - lambda_non * pnon_norm
            )
            log_rewards.append(nonterminal_log_r)

            leaf_mask_sub = get_leaf_mask(ei_sub, h_sub.size(0))
            chosen_sub = None if action == STOP_ACTION else index_map.get(action)
            log_pi = _build_log_pi(
                action_logits,
                stop_logit,
                leaf_mask_sub,
                chosen_sub,
                temperature,
                traj_rand_explore,
            )
            log_pis.append(log_pi)

            if action == STOP_ACTION:
                terminal_log_r = torch.log(
                    torch.tensor(terminal_vllm, device=device, dtype=log_v_pred.dtype)
                ) + pcau_weight * torch.log(mean_pcau) - lambda_non * pnon_norm
                log_rewards.append(terminal_log_r)
                break

            active_mask[action] = False

        n_states = len(log_rewards)
        if n_states < 2:
            continue

        for m in range(n_states - 1):
            cum_log_pi = torch.tensor(0.0, device=device)
            for n in range(m + 1, n_states):
                cum_log_pi = cum_log_pi + log_pis[n - 1]
                residual = log_rewards[n] - log_rewards[m] - cum_log_pi
                total_loss = total_loss + residual.pow(2)
                n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / n_terms


def _successor_log_value(
    h_all: torch.Tensor,
    ei_init: torch.Tensor,
    logodds_all: torch.Tensor,
    active_mask: torch.Tensor,
    action_global_idx: int,
    policy_model,
    device: torch.device,
    alpha_logodds: float = 1.0,
    beta_gnn: float = 1.0,
    q_emb: torch.Tensor | None = None,
) -> torch.Tensor | None:
    next_active = active_mask.clone()
    next_active[int(action_global_idx)] = False

    h_sub, ei_sub, lo_sub, _ = reconstruct_graph(
        h_all,
        ei_init,
        next_active,
        logodds_all,
        device,
    )
    if h_sub.size(0) == 0:
        return None

    _, _, log_value = _forward_single(
        policy_model,
        h_sub,
        ei_sub,
        lo_sub,
        alpha_logodds,
        device,
        beta_gnn,
        q_emb=q_emb,
    )
    return log_value


def evaluate_hinge_loss(
    batch_trajs: list[dict],
    policy_model,
    device: torch.device,
    alpha_logodds: float = 1.0,
    beta_gnn: float = 1.0,
    hinge_margin: float = 0.1,
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0
    margin = torch.tensor(float(hinge_margin), device=device)

    for traj in batch_trajs:
        hinge_pairs = traj.get("hinge_pairs", [])
        if not hinge_pairs:
            continue

        h_all = traj["h_chunk_all"]
        ei_init = traj["edge_index_init"]
        logodds_all = traj["logodds_all"]
        q_emb = traj.get("q_emb")

        for pair in hinge_pairs:
            active_mask = pair["active_mask"]
            if not isinstance(active_mask, torch.Tensor):
                active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
            active_mask = active_mask.to(dtype=torch.bool)

            winner_action = int(pair["winner_action"])
            loser_action = int(pair["loser_action"])

            winner_log_value = _successor_log_value(
                h_all=h_all,
                ei_init=ei_init,
                logodds_all=logodds_all,
                active_mask=active_mask,
                action_global_idx=winner_action,
                policy_model=policy_model,
                device=device,
                alpha_logodds=alpha_logodds,
                beta_gnn=beta_gnn,
                q_emb=q_emb,
            )
            loser_log_value = _successor_log_value(
                h_all=h_all,
                ei_init=ei_init,
                logodds_all=logodds_all,
                active_mask=active_mask,
                action_global_idx=loser_action,
                policy_model=policy_model,
                device=device,
                alpha_logodds=alpha_logodds,
                beta_gnn=beta_gnn,
                q_emb=q_emb,
            )

            if winner_log_value is None or loser_log_value is None:
                continue

            total_loss = total_loss + F.relu(margin - (winner_log_value - loser_log_value))
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / n_pairs
