from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from TREE_GRPO.tree_structs import TreeGroup, TreeNode


def _clip(x: float, lim: float) -> float:
    if lim <= 0:
        return float(x)
    if x > lim:
        return float(lim)
    if x < -lim:
        return float(-lim)
    return float(x)


def assign_tree_values_and_advantages(
    *,
    nodes: List[TreeNode],
    groups: Dict[int, TreeGroup],
    retry_token_id: int,
    c_retry: float,
    gamma: float,
    c_branch: float,
    advantage_clip: float,
) -> None:
    """
    Shallow Tree-GRPO credit assignment.

    Rules:
    - Q_F = q
    - Q_R = -c_retry + gamma * backup(children V) when retry is available
    - Forced-finalize nodes (no verify action) use U=Q_F and V=Q_F
    - Z/ANSWER tokens use U and local sibling mean-centering
    - VERIFY token uses local 2-action comparison advantage
    """
    node_by_id = {int(n.node_id): n for n in nodes}

    # Children first (depth=1 in v1), then roots.
    by_depth: Dict[int, List[TreeNode]] = defaultdict(list)
    for n in nodes:
        by_depth[int(n.retry_depth)].append(n)
    depths = sorted(by_depth.keys(), reverse=True)

    for depth in depths:
        for n in by_depth[depth]:
            n.Q_F = float(n.q)

            has_children = len(n.child_node_ids) > 0
            chosen_retry = int(n.verify_token_id) == int(retry_token_id) if n.verify_token_id is not None else False
            has_verify_action = bool(getattr(n, "verify_action_present", n.verify_token_id is not None))

            if not has_verify_action:
                # Retry is not a legal action on forced-finalize rounds.
                n.Q_R = float(n.Q_F)
                n.U = float(n.Q_F)
                n.V = float(n.Q_F)
                n.A_V = 0.0
                continue

            if has_children:
                child_vals = [float(node_by_id[cid].V) for cid in n.child_node_ids if cid in node_by_id]
                if len(child_vals) == 0:
                    raise RuntimeError(f"Retry node {n.node_id} has no child values; this should be exceptional")
                if len(child_vals) == 1:
                    backup = float(child_vals[0])
                elif len(child_vals) == 2:
                    backup = float(sum(child_vals) / 2.0)
                elif len(child_vals) == 4:
                    best = sorted(child_vals, reverse=True)[:2]
                    backup = float(sum(best) / 2.0)
                else:
                    backup = float(sum(child_vals) / float(len(child_vals)))
                n.Q_R = -float(c_retry) - float(c_branch) + float(gamma) * float(backup)
            else:
                # Finalize-chosen nodes may not have sampled retry descendants.
                # Keep counterfactual conservative-neutral in v1 (no truncation semantics).
                n.Q_R = float(n.Q_F)

            n.U = float(max(n.Q_F, n.Q_R))
            n.V = float(n.Q_R if chosen_retry else n.Q_F)

            q_chosen = float(n.Q_R if chosen_retry else n.Q_F)
            n.A_V = float(q_chosen - 0.5 * (n.Q_F + n.Q_R))

    # Group-local mean-centering for Z/ANSWER credit.
    for g in groups.values():
        member_ids = [int(x) for x in g.member_node_ids if int(x) in node_by_id]
        if len(member_ids) == 0:
            continue
        mean_u = sum(float(node_by_id[nid].U) for nid in member_ids) / float(len(member_ids))
        for nid in member_ids:
            node = node_by_id[nid]
            node.A_Z = float(node.U - mean_u)

    if float(advantage_clip) > 0.0:
        for n in nodes:
            n.A_Z = _clip(float(n.A_Z), float(advantage_clip))
            n.A_V = _clip(float(n.A_V), float(advantage_clip))


def tree_summary(nodes: Iterable[TreeNode], retry_token_id: int) -> Dict[str, float]:
    rows = list(nodes)
    if not rows:
        return {
            "num_nodes": 0.0,
            "num_retry_nodes": 0.0,
            "mean_q": 0.0,
            "mean_u": 0.0,
            "mean_v": 0.0,
            "mean_az": 0.0,
            "mean_av": 0.0,
        }
    n = float(len(rows))
    retry_n = float(sum(1 for r in rows if int(r.verify_token_id or -1) == int(retry_token_id)))
    return {
        "num_nodes": n,
        "num_retry_nodes": retry_n,
        "mean_q": float(sum(float(r.q) for r in rows) / n),
        "mean_u": float(sum(float(r.U) for r in rows) / n),
        "mean_v": float(sum(float(r.V) for r in rows) / n),
        "mean_az": float(sum(float(r.A_Z) for r in rows) / n),
        "mean_av": float(sum(float(r.A_V) for r in rows) / n),
    }
