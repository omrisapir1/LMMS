from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from PPO.train import Trajectory, _action_stats_tensors, _extract_z_phase_from_vllm_row_with_budget
from TREE_GRPO.conf import Config
from TREE_GRPO.credit import assign_tree_values_and_advantages, tree_summary
from TREE_GRPO.tree_structs import ExpandRequest, SegmentResult, TreeGroup, TreeNode


class _WaveState:
    def __init__(self, req: ExpandRequest, remaining_budget: int) -> None:
        self.req = req
        self.remaining_budget = int(max(0, remaining_budget))
        self.z_ids: List[int] = []
        self.has_answer: bool = False
        self.digit_ids: List[int] = []
        self.pred_digits: Optional[List[int]] = None
        self.verify_token_id: Optional[int] = None
        self.actions: List[int] = []
        self.action_types: List[str] = []
        self.full_generated_ids: List[int] = []
        self.terminated_reason: str = "max_new_tokens"


def _run_segment_wave(
    *,
    requests: Sequence[ExpandRequest],
    tokenizer,
    vllm_engine: Any,
    max_new_tokens_global: int,
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    digit_token_ids: Sequence[int],
    temperature: float,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    digit_temperature: float,
    digit_top_p: float,
    digit_greedy: bool,
) -> List[SegmentResult]:
    if len(requests) == 0:
        return []

    supports_token_prompts = bool(vllm_engine.supports_prompt_token_ids())
    decode_cache: Dict[Tuple[int, ...], str] = {}

    def _decode_cached(ids: Sequence[int]) -> str:
        key = tuple(int(x) for x in ids)
        if key in decode_cache:
            return decode_cache[key]
        txt = tokenizer.decode(
            list(key),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decode_cache[key] = str(txt)
        return str(txt)

    states: List[_WaveState] = []
    active_idx: List[int] = []
    for i, req in enumerate(requests):
        rem = int(max_new_tokens_global - int(req.path_generated_len))
        st = _WaveState(req=req, remaining_budget=rem)
        if rem > 0:
            active_idx.append(i)
        states.append(st)

    if active_idx:
        max_z_budget = max(states[i].remaining_budget for i in active_idx)
        z_prompt_ids = [requests[i].prefix_ids for i in active_idx]
        if supports_token_prompts:
            z_rows = vllm_engine.generate_z(
                prompt_token_ids=z_prompt_ids,
                num_samples_per_prompt=1,
                max_new_tokens=max_z_budget,
                temperature=float(temperature),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        else:
            z_texts = [_decode_cached(x) for x in z_prompt_ids]
            z_rows = vllm_engine.generate_z(
                prompts=z_texts,
                num_samples_per_prompt=1,
                max_new_tokens=max_z_budget,
                temperature=float(temperature),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        if len(z_rows) != len(active_idx):
            raise RuntimeError("Tree wave Z-phase row count mismatch")

        need_digits: List[int] = []
        for j, req_idx in enumerate(active_idx):
            st = states[req_idx]
            row = z_rows[j]
            z_prefix, has_answer = _extract_z_phase_from_vllm_row_with_budget(
                row=row,
                answer_token_id=int(answer_token_id),
                budget=int(st.remaining_budget),
            )
            st.z_ids = [int(x) for x in z_prefix]
            st.actions.extend(st.z_ids)
            st.action_types.extend(["z"] * len(st.z_ids))
            st.full_generated_ids.extend(st.z_ids)
            st.has_answer = bool(has_answer)

            if not st.has_answer:
                st.terminated_reason = "max_new_tokens"
                continue

            if len(st.full_generated_ids) >= st.remaining_budget:
                st.has_answer = False
                st.terminated_reason = "max_new_tokens"
                continue

            st.actions.append(int(answer_token_id))
            st.action_types.append("answer")
            st.full_generated_ids.append(int(answer_token_id))
            need_digits.append(req_idx)

        if need_digits:
            digits_group: Dict[int, List[int]] = defaultdict(list)
            for req_idx in need_digits:
                st = states[req_idx]
                rem_after_answer = int(st.remaining_budget - len(st.full_generated_ids))
                if rem_after_answer <= 0:
                    st.terminated_reason = "max_new_tokens"
                    continue
                k = min(5, rem_after_answer)
                digits_group[int(k)].append(req_idx)

            digit_allowed_set = set(int(x) for x in digit_token_ids)
            id2d = {int(tok): i for i, tok in enumerate(digit_token_ids)}
            need_verify: List[int] = []

            for k, idxs in digits_group.items():
                prompt_ids_batch = [requests[idx].prefix_ids + states[idx].full_generated_ids for idx in idxs]
                if supports_token_prompts:
                    digit_rows = vllm_engine.generate_digits(
                        prompt_token_ids=prompt_ids_batch,
                        num_digits=int(k),
                        temperature=float(digit_temperature),
                        top_p=float(digit_top_p),
                        greedy=bool(digit_greedy),
                        min_p=0.0,
                        repetition_penalty=1.0,
                    )
                else:
                    digit_texts = [_decode_cached(x) for x in prompt_ids_batch]
                    digit_rows = vllm_engine.generate_digits(
                        prompts=digit_texts,
                        num_digits=int(k),
                        temperature=float(digit_temperature),
                        top_p=float(digit_top_p),
                        greedy=bool(digit_greedy),
                        min_p=0.0,
                        repetition_penalty=1.0,
                    )
                if len(digit_rows) != len(idxs):
                    raise RuntimeError("Tree wave digit-phase row count mismatch")

                for row_i, req_idx in enumerate(idxs):
                    st = states[req_idx]
                    digits = [int(x) for x in list(digit_rows[row_i])]
                    if len(digits) != int(k):
                        raise RuntimeError(f"Digit phase must return exactly {k} tokens, got {len(digits)}")
                    bad = [d for d in digits if d not in digit_allowed_set]
                    if bad:
                        raise RuntimeError(f"Digit rollout contains tokens outside digit set: {bad}")
                    st.digit_ids = list(digits)
                    st.full_generated_ids.extend(st.digit_ids)

                    if int(k) < 5:
                        st.terminated_reason = "max_new_tokens"
                        continue
                    st.pred_digits = [int(id2d[x]) for x in st.digit_ids]
                    need_verify.append(req_idx)

            if need_verify:
                verify_prompt_ids = []
                verify_owner: List[int] = []
                for req_idx in need_verify:
                    st = states[req_idx]
                    rem_after_digits = int(st.remaining_budget - len(st.full_generated_ids))
                    if rem_after_digits <= 0:
                        st.terminated_reason = "max_new_tokens"
                        continue
                    verify_prompt_ids.append(requests[req_idx].prefix_ids + st.full_generated_ids)
                    verify_owner.append(req_idx)

                if verify_owner:
                    if supports_token_prompts:
                        verify_rows = vllm_engine.generate_verify(
                            prompt_token_ids=verify_prompt_ids,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            greedy=True,
                            min_p=float(min_p),
                            repetition_penalty=float(repetition_penalty),
                        )
                    else:
                        verify_texts = [_decode_cached(x) for x in verify_prompt_ids]
                        verify_rows = vllm_engine.generate_verify(
                            prompts=verify_texts,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            greedy=True,
                            min_p=float(min_p),
                            repetition_penalty=float(repetition_penalty),
                        )
                    if len(verify_rows) != len(verify_owner):
                        raise RuntimeError("Tree wave verify-phase row count mismatch")

                    for row_i, req_idx in enumerate(verify_owner):
                        st = states[req_idx]
                        row = [int(x) for x in list(verify_rows[row_i])]
                        if len(row) != 1:
                            raise RuntimeError(f"Verify phase must return exactly 1 token, got {len(row)}")
                        tok = int(row[0])
                        if tok not in (int(finalize_token_id), int(retry_token_id)):
                            raise RuntimeError("Verify phase emitted token outside {<FINALIZE>, <RETRY>}")
                        st.verify_token_id = int(tok)
                        st.actions.append(int(tok))
                        st.action_types.append("verify")
                        st.full_generated_ids.append(int(tok))
                        st.terminated_reason = "finalize" if tok == int(finalize_token_id) else "retry"

    out: List[SegmentResult] = []
    for st in states:
        next_prefix_ids = list(st.req.prefix_ids) + list(st.full_generated_ids)
        next_attn = list(st.req.prefix_attention_mask) + [1] * len(st.full_generated_ids)
        out.append(
            SegmentResult(
                z_token_ids=list(st.z_ids),
                has_answer=bool(st.has_answer),
                digit_token_ids=list(st.digit_ids),
                pred_digits=(None if st.pred_digits is None else list(st.pred_digits)),
                verify_token_id=(None if st.verify_token_id is None else int(st.verify_token_id)),
                actions=list(st.actions),
                action_types=list(st.action_types),
                full_generated_ids=list(st.full_generated_ids),
                next_prefix_ids=next_prefix_ids,
                next_prefix_attention_mask=next_attn,
                next_path_generated_len=int(st.req.path_generated_len + len(st.full_generated_ids)),
                terminated_reason=str(st.terminated_reason),
            )
        )
    return out


def collect_tree_grpo_v1_batch(
    *,
    model,
    value_head,
    tokenizer,
    vllm_engine: Any,
    prepared: Sequence[Dict[str, object]],
    cfg: Config,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    digit_token_ids: Sequence[int],
) -> Tuple[List[Trajectory], Dict[str, float]]:
    if len(prepared) == 0:
        return [], {}

    root_k = int(cfg.tree.root_siblings)
    if root_k != 4:
        raise RuntimeError(f"This v1 implementation expects tree.root_siblings=4, got {root_k}")

    group_id_next = 0
    node_id_next = 0
    groups: Dict[int, TreeGroup] = {}
    nodes: List[TreeNode] = []

    prompt_meta: Dict[int, Dict[str, object]] = {}

    root_requests: List[ExpandRequest] = []
    prompt_root_group: Dict[int, int] = {}

    for item in prepared:
        prompt_id = int(item["prompt_id"])
        prompt_meta[prompt_id] = {
            "question": str(item["question"]),
            "true_digits": [int(x) for x in list(item["true_digits"])],
            "sample_id_base": str(item["sample_id_base"]),
        }
        gid = int(group_id_next)
        group_id_next += 1
        groups[gid] = TreeGroup(
            group_id=gid,
            prompt_id=prompt_id,
            group_type="root_siblings",
            parent_node_id=None,
            member_node_ids=[],
        )
        prompt_root_group[prompt_id] = gid

        for branch_slot in range(root_k):
            root_requests.append(
                ExpandRequest(
                    prompt_id=prompt_id,
                    true_digits=[int(x) for x in list(item["true_digits"])],
                    prefix_ids=list(map(int, item["prompt_ids"])),
                    prefix_attention_mask=list(map(int, item["prompt_attention_mask"])),
                    path_generated_len=0,
                    retry_depth=0,
                    parent_node_id=None,
                    group_id=gid,
                    branch_slot=int(branch_slot),
                )
            )

    root_segments = _run_segment_wave(
        requests=root_requests,
        tokenizer=tokenizer,
        vllm_engine=vllm_engine,
        max_new_tokens_global=int(cfg.rollout.max_new_tokens),
        answer_token_id=int(answer_token_id),
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
        digit_token_ids=digit_token_ids,
        temperature=float(cfg.rollout.temperature),
        top_p=float(cfg.rollout.top_p),
        min_p=float(cfg.rollout.min_p),
        repetition_penalty=float(cfg.rollout.repetition_penalty),
        digit_temperature=float(cfg.rollout.digit_temperature),
        digit_top_p=float(cfg.rollout.digit_top_p),
        digit_greedy=bool(cfg.rollout.digit_greedy),
    )

    root_nodes_by_prompt: Dict[int, List[int]] = defaultdict(list)
    for req, seg in zip(root_requests, root_segments):
        nid = int(node_id_next)
        node_id_next += 1
        q = 0.0
        if seg.pred_digits is not None:
            q = 1.0 if list(seg.pred_digits) == list(req.true_digits) else 0.0
        n = TreeNode(
            node_id=nid,
            prompt_id=int(req.prompt_id),
            parent_node_id=None,
            retry_depth=0,
            group_id=int(req.group_id),
            group_type="root_siblings",
            branch_slot=int(req.branch_slot),
            path_generated_len_before=int(req.path_generated_len),
            path_generated_len_after=int(seg.next_path_generated_len),
            prompt_ids=list(req.prefix_ids),
            prompt_attention_mask=list(req.prefix_attention_mask),
            z_token_ids=list(seg.z_token_ids),
            digit_token_ids=list(seg.digit_token_ids),
            pred_digits=(None if seg.pred_digits is None else list(seg.pred_digits)),
            verify_token_id=(None if seg.verify_token_id is None else int(seg.verify_token_id)),
            actions=list(seg.actions),
            action_types=list(seg.action_types),
            full_generated_ids=list(seg.full_generated_ids),
            q=float(q),
            terminated_reason=str(seg.terminated_reason),
        )
        nodes.append(n)
        groups[int(req.group_id)].member_node_ids.append(int(nid))
        root_nodes_by_prompt[int(req.prompt_id)].append(int(nid))

    # Expand only retry nodes from root, up to 2 parents per prompt, each with k=2 children.
    child_requests: List[ExpandRequest] = []
    selected_retry_parents: set[int] = set()

    for prompt_id, node_ids in root_nodes_by_prompt.items():
        retry_roots = [
            nid
            for nid in node_ids
            if int(nodes[nid].verify_token_id or -1) == int(retry_token_id)
        ]
        retry_roots.sort(key=lambda nid: int(nodes[nid].branch_slot))
        keep = retry_roots[: int(cfg.tree.max_retry_parents_from_root)]
        selected_retry_parents.update(int(x) for x in keep)
        dropped = retry_roots[int(cfg.tree.max_retry_parents_from_root):]
        for nid in dropped:
            nodes[nid].retry_block_reason = "root_retry_parent_cap"

        for parent_nid in keep:
            parent = nodes[parent_nid]
            gid = int(group_id_next)
            group_id_next += 1
            groups[gid] = TreeGroup(
                group_id=gid,
                prompt_id=int(prompt_id),
                group_type="retry_children",
                parent_node_id=int(parent_nid),
                member_node_ids=[],
            )
            for branch_slot in range(int(cfg.tree.retry_children_per_parent)):
                child_requests.append(
                    ExpandRequest(
                        prompt_id=int(prompt_id),
                        true_digits=list(prompt_meta[int(prompt_id)]["true_digits"]),
                        prefix_ids=list(parent.prompt_ids) + list(parent.full_generated_ids),
                        prefix_attention_mask=list(parent.prompt_attention_mask) + [1] * len(parent.full_generated_ids),
                        path_generated_len=int(parent.path_generated_len_after),
                        retry_depth=1,
                        parent_node_id=int(parent_nid),
                        group_id=int(gid),
                        branch_slot=int(branch_slot),
                    )
                )

    child_segments = _run_segment_wave(
        requests=child_requests,
        tokenizer=tokenizer,
        vllm_engine=vllm_engine,
        max_new_tokens_global=int(cfg.rollout.max_new_tokens),
        answer_token_id=int(answer_token_id),
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
        digit_token_ids=digit_token_ids,
        temperature=float(cfg.rollout.temperature),
        top_p=float(cfg.rollout.top_p),
        min_p=float(cfg.rollout.min_p),
        repetition_penalty=float(cfg.rollout.repetition_penalty),
        digit_temperature=float(cfg.rollout.digit_temperature),
        digit_top_p=float(cfg.rollout.digit_top_p),
        digit_greedy=bool(cfg.rollout.digit_greedy),
    )

    for req, seg in zip(child_requests, child_segments):
        nid = int(node_id_next)
        node_id_next += 1
        q = 0.0
        if seg.pred_digits is not None:
            q = 1.0 if list(seg.pred_digits) == list(req.true_digits) else 0.0
        n = TreeNode(
            node_id=nid,
            prompt_id=int(req.prompt_id),
            parent_node_id=(None if req.parent_node_id is None else int(req.parent_node_id)),
            retry_depth=1,
            group_id=int(req.group_id),
            group_type="retry_children",
            branch_slot=int(req.branch_slot),
            path_generated_len_before=int(req.path_generated_len),
            path_generated_len_after=int(seg.next_path_generated_len),
            prompt_ids=list(req.prefix_ids),
            prompt_attention_mask=list(req.prefix_attention_mask),
            z_token_ids=list(seg.z_token_ids),
            digit_token_ids=list(seg.digit_token_ids),
            pred_digits=(None if seg.pred_digits is None else list(seg.pred_digits)),
            verify_token_id=(None if seg.verify_token_id is None else int(seg.verify_token_id)),
            actions=list(seg.actions),
            action_types=list(seg.action_types),
            full_generated_ids=list(seg.full_generated_ids),
            q=float(q),
            terminated_reason=str(seg.terminated_reason),
        )
        nodes.append(n)
        groups[int(req.group_id)].member_node_ids.append(int(nid))
        if req.parent_node_id is not None:
            nodes[int(req.parent_node_id)].child_node_ids.append(int(nid))

    for root_nid in selected_retry_parents:
        if len(nodes[root_nid].child_node_ids) == 0:
            nodes[root_nid].retry_block_reason = "budget_truncated"

    assign_tree_values_and_advantages(
        nodes=nodes,
        groups=groups,
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
        c_retry=float(cfg.tree.c_retry),
        c_trunc=float(cfg.tree.c_trunc),
        gamma=float(cfg.tree.gamma),
        c_branch=float(cfg.tree.c_branch),
        advantage_clip=float(cfg.tree.advantage_clip),
    )

    trajectories: List[Trajectory] = []
    for n in nodes:
        if len(n.actions) == 0:
            continue

        logp_t, values_t, entropy_t = _action_stats_tensors(
            model=model,
            value_head=value_head,
            prompt_ids=n.prompt_ids,
            prompt_attention_mask=n.prompt_attention_mask,
            actions=n.actions,
            action_types=n.action_types,
            z_allowed_t=z_allowed_t,
            digit_allowed_t=digit_allowed_t,
            verify_allowed_t=verify_allowed_t,
            temperature=float(cfg.rollout.temperature),
        )

        returns: List[float] = []
        advantages: List[float] = []
        for t in n.action_types:
            if str(t) == "verify":
                returns.append(float(n.V))
                advantages.append(float(n.A_V))
            else:
                returns.append(float(n.U))
                advantages.append(float(n.A_Z))

        if len(returns) != len(n.actions):
            raise RuntimeError("Tree returns length mismatch")
        if len(advantages) != len(n.actions):
            raise RuntimeError("Tree advantages length mismatch")

        prompt_info = prompt_meta[int(n.prompt_id)]
        reward_info: Dict[str, object] = {
            "reward_full": int(1 if float(n.q) >= 1.0 else 0),
            "reward_partial": float(n.q),
            "reward": float(n.q),
            "reward_final": float(n.V),
            "exact_match": bool(float(n.q) >= 1.0),
            "q": float(n.q),
            "Q_F": float(n.Q_F),
            "Q_R": float(n.Q_R),
            "U": float(n.U),
            "V": float(n.V),
            "A_Z": float(n.A_Z),
            "A_V": float(n.A_V),
            "group_id": int(n.group_id),
            "group_type": str(n.group_type),
            "retry_depth": int(n.retry_depth),
            "parent_node_id": int(n.parent_node_id) if n.parent_node_id is not None else None,
            "child_node_ids": [int(x) for x in n.child_node_ids],
            "retry_block_reason": str(n.retry_block_reason),
            "terminated_reason": str(n.terminated_reason),
            "tree_mode": "shallow_v1",
        }

        traj = Trajectory(
            prompt_id=int(n.prompt_id),
            sample_id=f"{str(prompt_info['sample_id_base'])}_n{int(n.node_id)}",
            question=str(prompt_info["question"]),
            prompt_ids=list(n.prompt_ids),
            prompt_attention_mask=list(n.prompt_attention_mask),
            actions=list(n.actions),
            action_types=list(n.action_types),
            logp_old=logp_t.float().cpu().tolist(),
            values_old=values_t.float().cpu().tolist(),
            entropy_old=entropy_t.float().cpu().tolist(),
            terminated_by=str(n.terminated_reason),
            generated_z_ids=list(n.z_token_ids),
            generated_digit_ids=list(n.digit_token_ids),
            digit_logits=None,
            digit_probs=None,
            digit_pred=(None if n.pred_digits is None else list(n.pred_digits)),
            digit_true=[int(x) for x in list(prompt_info["true_digits"])],
            reward_info=reward_info,
            num_generated_total=int(len(n.full_generated_ids)),
            num_digits_generated=int(len(n.digit_token_ids)),
            generated_verify_ids=([] if n.verify_token_id is None else [int(n.verify_token_id)]),
            rounds_meta=[
                {
                    "tree_node_id": int(n.node_id),
                    "group_id": int(n.group_id),
                    "group_type": str(n.group_type),
                    "retry_depth": int(n.retry_depth),
                    "parent_node_id": int(n.parent_node_id) if n.parent_node_id is not None else None,
                    "z_token_ids": list(n.z_token_ids),
                    "digit_token_ids": list(n.digit_token_ids),
                    "pred_digits": (None if n.pred_digits is None else list(n.pred_digits)),
                    "verify_token_id": (None if n.verify_token_id is None else int(n.verify_token_id)),
                    "q": float(n.q),
                    "Q_F": float(n.Q_F),
                    "Q_R": float(n.Q_R),
                    "U": float(n.U),
                    "V": float(n.V),
                    "A_Z": float(n.A_Z),
                    "A_V": float(n.A_V),
                    "retry_block_reason": str(n.retry_block_reason),
                }
            ],
            full_generated_ids=list(n.full_generated_ids),
            termination_reason=str(n.terminated_reason),
        )

        traj.returns = list(returns)
        traj.advantages = list(advantages)
        traj.advantages_norm_global = []
        traj.advantages_norm_prompt = []
        traj.advantages_norm = list(advantages)
        trajectories.append(traj)

    stats = tree_summary(nodes=nodes, retry_token_id=int(retry_token_id))
    stats["num_groups"] = float(len(groups))
    stats["num_root_requests"] = float(len(root_requests))
    stats["num_child_requests"] = float(len(child_requests))
    stats["num_trajectories"] = float(len(trajectories))
    return trajectories, stats
