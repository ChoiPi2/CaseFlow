from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MinNormSolver:
    @staticmethod
    def gradient_normalizers(grads, losses, mode="loss+"):
        normalizers = {}
        for name, grad_list in grads.items():
            if not grad_list:
                normalizers[name] = torch.tensor(1.0)
                continue

            if mode == "l2":
                total = torch.tensor(0.0, device=grad_list[0].device)
                for grad in grad_list:
                    total = total + grad.pow(2).sum()
                normalizers[name] = total.sqrt().clamp_min(1e-6)
            elif mode in {"loss", "loss+"}:
                loss_value = losses[name]
                if not torch.is_tensor(loss_value):
                    loss_value = torch.tensor(float(loss_value), device=grad_list[0].device)
                normalizers[name] = loss_value.abs().clamp_min(1e-6)
            else:
                normalizers[name] = torch.tensor(1.0, device=grad_list[0].device)
        return normalizers

    @staticmethod
    def _dot(grad_list_a, grad_list_b):
        total = torch.tensor(0.0, device=grad_list_a[0].device)
        for grad_a, grad_b in zip(grad_list_a, grad_list_b):
            total = total + (grad_a * grad_b).sum()
        return total

    @staticmethod
    def _two_task_solution(grad_list_a, grad_list_b):
        g11 = MinNormSolver._dot(grad_list_a, grad_list_a)
        g22 = MinNormSolver._dot(grad_list_b, grad_list_b)
        g12 = MinNormSolver._dot(grad_list_a, grad_list_b)

        denom = g11 + g22 - 2 * g12
        if torch.abs(denom) < 1e-12:
            gamma = 0.5
        else:
            gamma = ((g22 - g12) / denom).clamp(0.0, 1.0).item()
        return [gamma, 1.0 - gamma], None

    @staticmethod
    def find_min_norm_element_FW(grad_list_collection):
        n_tasks = len(grad_list_collection)
        if n_tasks == 0:
            return [], None
        if n_tasks == 1:
            return [1.0], None
        if n_tasks == 2:
            return MinNormSolver._two_task_solution(
                grad_list_collection[0], grad_list_collection[1]
            )
        weight = 1.0 / n_tasks
        return [weight for _ in range(n_tasks)], None


def num_graphs(data) -> int:
    if hasattr(data, "num_graphs"):
        return data.num_graphs
    return int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0




 
     





def infonce_losses(
    z_c:         torch.Tensor,                                          
    z_s:         torch.Tensor,                                            
    z_a:         torch.Tensor,                               
    z_neg:       torch.Tensor,                             
    anchor_mask: torch.Tensor | None = None,                                  
    support_mask: torch.Tensor | None = None,                                   
    tau_infonce: float = 0.2,
    eps:         float = 1e-6,
):
    if z_a.dim() == 1:
        z_a = z_a.unsqueeze(0)
    z_a_n   = F.normalize(z_a, p=2, dim=-1)                                     
    z_neg_n = F.normalize(z_neg, p=2, dim=-1) if z_neg.numel() > 0 else \
        torch.zeros(0, z_c.size(-1), device=z_c.device)                       
    anchor_mask_b = anchor_mask.view(-1).bool() if anchor_mask is not None else None
    support_mask_b = support_mask.view(-1).bool() if support_mask is not None else None


    z_c_n  = F.normalize(z_c,       p=2, dim=-1)                                
    z_cy_n = F.normalize(z_c + z_s, p=2, dim=-1)                                

    def _infonce_mean(z_n, mask=None):
        if mask is not None:
            z_n = z_n[mask]
        if z_n.size(0) == 0:
            return _zero_loss(z_c)
        e_pos = (z_n @ z_a_n.T) / tau_infonce                                     
        logits = [e_pos]
        if z_neg_n.size(0) > 0:
            logits.append((z_n @ z_neg_n.T) / tau_infonce)                      
        all_e = torch.cat(logits, dim=1)
        log_num = torch.logsumexp(e_pos, dim=1)                              
        log_denom = torch.logsumexp(all_e, dim=1)                            
        per_node  = log_denom - log_num                                      
        return per_node.mean()                                                 

    has_causal = anchor_mask_b is not None and bool(anchor_mask_b.any().item())
    l_c = _infonce_mean(z_c_n, anchor_mask_b) if has_causal else _zero_loss(z_c)


    l_y = _infonce_mean(z_cy_n)


    z_s_n   = F.normalize(z_s, p=2, dim=-1)                                     
    cos_sq  = (z_s_n @ z_a_n.T).pow(2).mean(dim=-1)                          
    if support_mask_b is not None:
        if not bool(support_mask_b.any().item()):
            return l_c, _zero_loss(z_s), l_y, has_causal, False
        spurious_mask = ~support_mask_b
        has_spurious = bool(spurious_mask.any().item())
        if has_spurious:
            cos_sq = cos_sq[spurious_mask]
    else:
        has_spurious = cos_sq.numel() > 0
    l_s_nec = cos_sq.mean() if has_spurious else _zero_loss(z_s)              

    return l_c, l_s_nec, l_y, has_causal, has_spurious






def vicreg_variance_loss(
    X_s:       torch.Tensor,
    x_h:       torch.Tensor,
    batch:     torch.Tensor,
    alpha:     float = 0.3,
    min_nodes: int   = 8,
) -> torch.Tensor:
    eps = 1e-4
    n_graphs    = int(batch.max().item()) + 1
    loss_sum    = torch.tensor(0.0, device=X_s.device)
    weight_sum  = 0.0

    for g in range(n_graphs):
        mask = (batch == g)
        n_g  = int(mask.sum().item())
        if n_g < min_nodes:
            continue

        x_h_g = x_h[mask].detach()
        X_s_g = X_s[mask]

        target  = alpha * torch.sqrt(x_h_g.var(dim=0) + eps)
        std_xs  = torch.sqrt(X_s_g.var(dim=0) + eps)

        loss_sum   = loss_sum + n_g * F.softplus(target - std_xs).mean()
        weight_sum += n_g

    if weight_sum == 0.0:
        return torch.tensor(0.0, device=X_s.device, requires_grad=False)

    return loss_sum / weight_sum






def pretrain_step_Ls_total(model, optimizer, graph_buf, device, args):
    from torch_geometric.data import Batch as PyGBatch

    model.train()
    if not graph_buf:
        return 0.0, 0.0, 0.0, 0.0, torch.tensor([])

    data_list, z_neg_list = zip(*graph_buf)
    data  = PyGBatch.from_data_list(list(data_list)).to(device)
    z_neg = torch.cat([z.to(device) for z in z_neg_list], dim=0)

    if not hasattr(data, "z_a") or data.z_a is None:
        return 0.0, 0.0, 0.0, 0.0, torch.tensor([])

    optimizer.zero_grad()



    z_c, z_s, ac_node, as_node, ac_edge, as_edge, z_a_proj, X_s, x_h = model(
        data, return_xs=True, return_alpha=True
    )

    z_neg_proj = model.proj_a(z_neg) if z_neg.size(0) > 0 else \
        torch.zeros(0, z_c.size(-1), device=device)

    n_graphs        = data.num_graphs
    l_c_total       = _zero_loss(z_c)
    l_s_nec_total   = _zero_loss(z_s)
    l_y_total       = _zero_loss(z_c + z_s)
    neg_offset      = 0
    answer_offset   = 0
    n_causal        = 0
    n_spurious      = 0
    answer_counts = (
        data.answer_count.view(-1).tolist()
        if hasattr(data, "answer_count")
        else [1] * n_graphs
    )

    for g in range(n_graphs):
        mask   = (data.batch == g)
        zc_g   = z_c[mask]                                  
        zs_g   = z_s[mask]                                  
        count_g = int(answer_counts[g]) if g < len(answer_counts) else 1
        za_g   = z_a_proj[answer_offset:answer_offset + count_g]
        answer_offset += count_g
        anchor_g = data.anchor_mask[mask] if hasattr(data, "anchor_mask") else None
        support_g = data.support_mask[mask] if hasattr(data, "support_mask") else \
            (data.path_mask[mask] if hasattr(data, "path_mask") else None)
        M_g    = z_neg_list[g].size(0)
        zneg_g = z_neg_proj[neg_offset:neg_offset + M_g]
        neg_offset += M_g

        lc_g, ls_nec_g, ly_g, has_causal, has_spurious = infonce_losses(
            zc_g, zs_g, za_g, zneg_g,
            anchor_mask=anchor_g,
            support_mask=support_g,
            tau_infonce=args.tau_infonce,
        )
        if has_causal:
            l_c_total += lc_g
            n_causal += 1
        if has_spurious:
            l_s_nec_total += ls_nec_g
            n_spurious += 1
        l_y_total     += ly_g

    l_c     = l_c_total     / n_causal if n_causal > 0 else _zero_loss(z_c)
    l_s_nec = l_s_nec_total / n_spurious if n_spurious > 0 else _zero_loss(z_s)
    l_y     = l_y_total     / n_graphs

    lambda_var = getattr(args, "lambda_var", 0.3)
    l_s_var    = vicreg_variance_loss(X_s, x_h, data.batch)
    l_s        = l_s_nec + lambda_var * l_s_var

    w_c  = getattr(args, "c",  1.0)
    w_s  = getattr(args, "o",  1.0)
    w_y  = getattr(args, "co", 1.0)
    loss = w_c * l_c + w_s * l_s + w_y * l_y


    if getattr(args, "att_loss", False):
        eps = 1e-6



        att_loss = 1.0 / (F.smooth_l1_loss(as_node, ac_node) + eps)
        if att_loss > getattr(args, "att_loss_threshold", 5.0):
            loss = loss + att_loss
    loss.backward()
    optimizer.step()

    return loss.item(), l_c.item(), l_s.item(), l_y.item(), torch.tensor([])







def pretrain_step_mgda_Ls_total(model, optimizer, graph_buf, device, args):
    from torch_geometric.data import Batch as PyGBatch

    model.train()
    if not graph_buf:
        return 0.0, 0.0, 0.0, 0.0, torch.tensor([])

    data_list, z_neg_list = zip(*graph_buf)
    data  = PyGBatch.from_data_list(list(data_list)).to(device)
    z_neg = torch.cat([z.to(device) for z in z_neg_list], dim=0)

    if not hasattr(data, "z_a") or data.z_a is None:
        return 0.0, 0.0, 0.0, 0.0, torch.tensor([])

    optimizer.zero_grad()

    z_c, z_s, ac_node, as_node, ac_edge, as_edge, z_a_proj, X_s, x_h = model(
        data, return_xs=True, return_alpha=True
    )

    z_neg_proj = model.proj_a(z_neg) if z_neg.size(0) > 0 else \
        torch.zeros(0, z_c.size(-1), device=device)

    n_graphs        = data.num_graphs
    l_c_total       = _zero_loss(z_c)
    l_s_nec_total   = _zero_loss(z_s)
    l_y_total       = _zero_loss(z_c + z_s)
    neg_offset      = 0
    answer_offset   = 0
    n_causal        = 0
    n_spurious      = 0
    answer_counts = (
        data.answer_count.view(-1).tolist()
        if hasattr(data, "answer_count")
        else [1] * n_graphs
    )

    for g in range(n_graphs):
        mask   = (data.batch == g)
        zc_g   = z_c[mask]
        zs_g   = z_s[mask]
        count_g = int(answer_counts[g]) if g < len(answer_counts) else 1
        za_g   = z_a_proj[answer_offset:answer_offset + count_g]
        answer_offset += count_g
        anchor_g = data.anchor_mask[mask] if hasattr(data, "anchor_mask") else None
        support_g = data.support_mask[mask] if hasattr(data, "support_mask") else \
            (data.path_mask[mask] if hasattr(data, "path_mask") else None)
        M_g    = z_neg_list[g].size(0)
        zneg_g = z_neg_proj[neg_offset:neg_offset + M_g]
        neg_offset += M_g

        lc_g, ls_nec_g, ly_g, has_causal, has_spurious = infonce_losses(
            zc_g, zs_g, za_g, zneg_g,
            anchor_mask=anchor_g,
            support_mask=support_g,
            tau_infonce=args.tau_infonce,
        )
        if has_causal:
            l_c_total += lc_g
            n_causal += 1
        if has_spurious:
            l_s_nec_total += ls_nec_g
            n_spurious += 1
        l_y_total     += ly_g

    l_c     = l_c_total     / n_causal if n_causal > 0 else _zero_loss(z_c)
    l_s_nec = l_s_nec_total / n_spurious if n_spurious > 0 else _zero_loss(z_s)
    l_y     = l_y_total     / n_graphs


    lambda_var = getattr(args, "lambda_var", 0.3)
    l_s_var    = vicreg_variance_loss(X_s, x_h, data.batch)
    l_s        = l_s_nec + lambda_var * l_s_var

    if n_causal == 0:
        loss = l_s + l_y
        loss.backward()
        optimizer.step()
        return loss.item(), l_c.item(), l_s.item(), l_y.item(), torch.tensor([[0.0, 1.0]])


    def _get_grad_big():
        return [
            Variable(p.grad.data.clone(), requires_grad=False)
            if p.grad is not None
            else Variable(torch.zeros_like(p.data), requires_grad=False)
            for p in model.big.parameters()
        ]

    l_c.backward(retain_graph=True)
    grads_big_c = _get_grad_big()
    model.zero_grad()

    l_s.backward(retain_graph=True)
    grads_big_s = _get_grad_big()
    model.zero_grad()

    grads     = {"c": grads_big_c, "s": grads_big_s}
    loss_data = {"c": l_c.data.clone(), "s": l_s.data.clone()}

    gn = MinNormSolver.gradient_normalizers(grads, loss_data, args.mgda_model)
    for name in ["c", "s"]:
        if gn[name] < 1e-3:
            gn[name] = torch.tensor(1e-3, device=device)
    for name in grads:
        for i in range(len(grads[name])):
            grads[name][i] = grads[name][i] / gn[name].to(grads[name][i].device)

    sol, _ = MinNormSolver.find_min_norm_element_FW([grads["c"], grads["s"]])
    w_c, w_s = float(sol[0]), float(sol[1])


    loss = w_c * l_c + w_s * l_s + l_y


    if getattr(args, "att_loss", False):
        eps = 1e-6



        att_loss = 1.0 / (F.smooth_l1_loss(as_node, ac_node) + eps)
        if att_loss > getattr(args, "att_loss_threshold", 5.0):
            loss = loss + att_loss
    loss.backward()
    optimizer.step()

    return loss.item(), l_c.item(), l_s.item(), l_y.item(), torch.tensor([[w_c, w_s]])
