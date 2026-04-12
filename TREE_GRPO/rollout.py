from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from PPO.train import Trajectory, _extract_z_phase_from_vllm_row_with_budget
from TREE_GRPO.conf import Config
from TREE_GRPO.credit import assign_tree_values_and_advantages, tree_summary
from TREE_GRPO.tree_structs import ExpandRequest, SegmentResult, TreeGroup, TreeNode


class _WaveState:
    def __init__(self, req: ExpandRequest) -> None:
        self.req = req
        self.z_ids: List[int] = []
        self.has_answer: bool = False
        self.digit_ids: List[int] = []
        self.pred_digits: Optional[List[int]] = None
        self.verify_token_id: Optional[int] = None
        self.actions: List[int] = []
        self.action_types: List[str] = []
        self.full_generated_ids: List[int] = []
        self.terminated_reason: str = "non_terminal_retry"
        self.was_forced_finalize: bool = False
        self.verify_action_present: bool = False
        self.leaf_end_type: str = "non_terminal_retry"


def _action_logp_entropy_tensors(
    *,
    model,
    prompt_ids: Sequence[int],
    prompt_attention_mask: Sequence[int],
    actions: Sequence[int],
    action_types: Sequence[str],
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    if len(actions) == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return empty, empty

    seq_ids = torch.tensor(list(prompt_ids) + list(actions), dtype=torch.long, device=device).unsqueeze(0)
    full_attn = list(prompt_attention_mask) + [1] * len(actions)
    attn = torch.tensor(full_attn, dtype=torch.long, device=device).unsqueeze(0)

    out = model(
        input_ids=seq_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )

    p_len = len(prompt_ids)
    t_steps = len(actions)
    state_positions = torch.arange(
        p_len - 1,
        p_len - 1 + t_steps,
        device=device,
        dtype=torch.long,
    )
    logits_all = out.logits[0]

    logp_list: List[torch.Tensor] = []
    entropy_list: List[torch.Tensor] = []
    for i in range(t_steps):
        pos = int(state_positions[i].item())
        aid = int(actions[i])
        t = str(action_types[i])
        if t == "digit":
            allowed_t = digit_allowed_t
        elif t == "verify":
            allowed_t = verify_allowed_t
        else:
            allowed_t = z_allowed_t

        allowed_logits = logits_all[pos].index_select(0, allowed_t) / float(temperature)
        log_probs_allowed = torch.log_softmax(allowed_logits, dim=-1)
        probs_allowed = log_probs_allowed.exp()

        local_matches = torch.nonzero(allowed_t == aid, as_tuple=False)
        if local_matches.numel() == 0:
            raise RuntimeError(f"Action id {aid} not in allowed set for type={t}")
        local_idx = int(local_matches[0].item())
        logp_list.append(log_probs_allowed[local_idx])
        entropy_list.append((-(probs_allowed * log_probs_allowed).sum()))

    return torch.stack(logp_list, dim=0), torch.stack(entropy_list, dim=0)


def _run_segment_wave(
    *,
    requests: Sequence[ExpandRequest],
    tokenizer,
    vllm_engine: Any,
    max_new_tokens_round: int,
    max_retry_depth: int,
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    digit_token_ids: Sequence[int],
    temperature: float,
    top_p: float,
    verify_temperature: float,
    verify_p: float,
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

    states: List[_WaveState] = [_WaveState(req=req) for req in requests]
    active_idx = list(range(len(requests)))

    max_z_budget = max(int(max_new_tokens_round), 1)
    row_by_req_idx: Dict[int, Dict[str, object]] = {}

    # Group by identical prefix to reuse prefill and request sibling samples
    # via num_samples_per_prompt=k (important for retry-child efficiency).
    prefix_to_req_idxs: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for req_idx in active_idx:
        key = tuple(int(x) for x in requests[req_idx].prefix_ids)
        prefix_to_req_idxs[key].append(int(req_idx))

    for key, req_idxs in prefix_to_req_idxs.items():
        ordered_req_idxs = sorted(req_idxs, key=lambda i: int(requests[i].branch_slot))
        k = int(len(ordered_req_idxs))
        if k <= 0:
            continue
        if supports_token_prompts:
            z_rows_group = vllm_engine.generate_z(
                prompt_token_ids=[list(key)],
                num_samples_per_prompt=k,
                max_new_tokens=max_z_budget,
                temperature=float(temperature),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        else:
            z_rows_group = vllm_engine.generate_z(
                prompts=[_decode_cached(key)],
                num_samples_per_prompt=k,
                max_new_tokens=max_z_budget,
                temperature=float(temperature),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        if len(z_rows_group) != k:
            raise RuntimeError(
                f"Tree wave grouped Z-phase row count mismatch: got={len(z_rows_group)} expected={k}"
            )
        for local_i, req_idx in enumerate(ordered_req_idxs):
            row_by_req_idx[int(req_idx)] = dict(z_rows_group[local_i])

    # Z + <ANSWER> (strict: each round must reach an answer token).
    for req_idx in active_idx:
        st = states[req_idx]
        row = row_by_req_idx.get(int(req_idx))
        if row is None:
            raise RuntimeError(f"Missing grouped Z-phase row for request index {req_idx}")
        z_prefix, has_answer = _extract_z_phase_from_vllm_row_with_budget(
            row=row,
            answer_token_id=int(answer_token_id),
            budget=max_z_budget,
        )
        st.z_ids = [int(x) for x in z_prefix]
        st.actions.extend(st.z_ids)
        st.action_types.extend(["z"] * len(st.z_ids))
        st.full_generated_ids.extend(st.z_ids)
        st.has_answer = bool(has_answer)
        if not st.has_answer:
            raise RuntimeError(
                f"Round failed to emit <ANSWER> within max_new_tokens={max_z_budget}; "
                "this violates terminal-leaf semantics."
            )
        st.actions.append(int(answer_token_id))
        st.action_types.append("answer")
        st.full_generated_ids.append(int(answer_token_id))

    # Always generate exactly 5 digits.
    digit_allowed_set = set(int(x) for x in digit_token_ids)
    id2d = {int(tok): i for i, tok in enumerate(digit_token_ids)}
    digit_prompt_ids = [requests[idx].prefix_ids + states[idx].full_generated_ids for idx in active_idx]
    if supports_token_prompts:
        digit_rows = vllm_engine.generate_digits(
            prompt_token_ids=digit_prompt_ids,
            num_digits=5,
            temperature=float(digit_temperature),
            top_p=float(digit_top_p),
            greedy=bool(digit_greedy),
            min_p=0.0,
            repetition_penalty=1.0,
        )
    else:
        digit_rows = vllm_engine.generate_digits(
            prompts=[_decode_cached(x) for x in digit_prompt_ids],
            num_digits=5,
            temperature=float(digit_temperature),
            top_p=float(digit_top_p),
            greedy=bool(digit_greedy),
            min_p=0.0,
            repetition_penalty=1.0,
        )
    if len(digit_rows) != len(active_idx):
        raise RuntimeError("Tree wave digit-phase row count mismatch")
    for j, req_idx in enumerate(active_idx):
        st = states[req_idx]
        digits = [int(x) for x in list(digit_rows[j])]
        if len(digits) != 5:
            raise RuntimeError(f"Digit phase must return exactly 5 tokens, got {len(digits)}")
        bad = [d for d in digits if d not in digit_allowed_set]
        if bad:
            raise RuntimeError(f"Digit rollout contains tokens outside digit set: {bad}")
        st.digit_ids = list(digits)
        st.pred_digits = [int(id2d[x]) for x in st.digit_ids]
        st.full_generated_ids.extend(st.digit_ids)

    # Verify only when retry is legal at this depth.
    need_verify: List[int] = []
    verify_prompt_ids: List[List[int]] = []
    for req_idx in active_idx:
        st = states[req_idx]
        req = requests[req_idx]
        force_finalize = int(req.retry_depth) >= int(max_retry_depth)
        if force_finalize:
            st.was_forced_finalize = True
            st.verify_action_present = False
            st.verify_token_id = None
            st.terminated_reason = "forced_finalize_max_retry"
            st.leaf_end_type = "forced_finalize_max_retry"
            continue
        need_verify.append(int(req_idx))
        verify_prompt_ids.append(list(req.prefix_ids) + list(st.full_generated_ids))

    if need_verify:
        if supports_token_prompts:
            verify_rows = vllm_engine.generate_verify(
                prompt_token_ids=verify_prompt_ids,
                temperature=float(verify_temperature),
                top_p=float(verify_p),
                greedy=False,
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        else:
            verify_rows = vllm_engine.generate_verify(
                prompts=[_decode_cached(x) for x in verify_prompt_ids],
                temperature=float(verify_temperature),
                top_p=float(verify_p),
                greedy=False,
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
            )
        if len(verify_rows) != len(need_verify):
            raise RuntimeError("Tree wave verify-phase row count mismatch")
        for row_i, req_idx in enumerate(need_verify):
            st = states[req_idx]
            row = [int(x) for x in list(verify_rows[row_i])]
            if len(row) != 1:
                raise RuntimeError(f"Verify phase must return exactly 1 token, got {len(row)}")
            tok = int(row[0])
            if tok not in (int(finalize_token_id), int(retry_token_id)):
                raise RuntimeError("Verify phase emitted token outside {<FINALIZE>, <RETRY>}")
            st.verify_action_present = True
            st.verify_token_id = int(tok)
            st.actions.append(int(tok))
            st.action_types.append("verify")
            st.full_generated_ids.append(int(tok))
            if tok == int(finalize_token_id):
                st.terminated_reason = "model_finalize"
                st.leaf_end_type = "model_finalize"
            else:
                st.terminated_reason = "retry"
                st.leaf_end_type = "non_terminal_retry"

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
                was_forced_finalize=bool(st.was_forced_finalize),
                verify_action_present=bool(st.verify_action_present),
                leaf_end_type=str(st.leaf_end_type),
            )
        )
    return out


def collect_tree_grpo_v1_batch(
    *,
    model,
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

    # v1 split policy is intentionally shallow and fixed-shape.
    root_k = int(cfg.tree.root_siblings)
    if root_k != 4:
        raise RuntimeError(f"This v1 implementation expects tree.root_siblings=4, got {root_k}")

    group_id_next = 0
    node_id_next = 0
    groups: Dict[int, TreeGroup] = {}
    nodes: List[TreeNode] = []
    node_by_id: Dict[int, TreeNode] = {}

    prompt_meta: Dict[int, Dict[str, object]] = {}

    root_requests: List[ExpandRequest] = []
    nonroot_request_count = 0

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
        max_new_tokens_round=int(cfg.rollout.max_new_tokens),
        max_retry_depth=int(cfg.tree.max_retry_depth),
        answer_token_id=int(answer_token_id),
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
        digit_token_ids=digit_token_ids,
        temperature=float(cfg.rollout.temperature),
        top_p=float(cfg.rollout.top_p),
        verify_temperature=float(cfg.rollout.verify_temperature),
        verify_p=float(cfg.rollout.verify_p),
        min_p=float(cfg.rollout.min_p),
        repetition_penalty=float(cfg.rollout.repetition_penalty),
        digit_temperature=float(cfg.rollout.digit_temperature),
        digit_top_p=float(cfg.rollout.digit_top_p),
        digit_greedy=bool(cfg.rollout.digit_greedy),
    )

    def _append_node(req: ExpandRequest, seg: SegmentResult, *, group_type: str) -> int:
        nonlocal node_id_next
        nid = int(node_id_next)
        node_id_next += 1
        q = 0.0
        if seg.pred_digits is not None:
            q = 1.0 if list(seg.pred_digits) == list(req.true_digits) else 0.0
        n = TreeNode(
            node_id=nid,
            prompt_id=int(req.prompt_id),
            parent_node_id=(None if req.parent_node_id is None else int(req.parent_node_id)),
            retry_depth=int(req.retry_depth),
            group_id=int(req.group_id),
            group_type=str(group_type),
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
            was_forced_finalize=bool(seg.was_forced_finalize),
            verify_action_present=bool(seg.verify_action_present),
            leaf_end_type=str(seg.leaf_end_type),
        )
        nodes.append(n)
        node_by_id[int(nid)] = n
        groups[int(req.group_id)].member_node_ids.append(int(nid))
        if req.parent_node_id is not None:
            parent = node_by_id.get(int(req.parent_node_id))
            if parent is None:
                raise RuntimeError(f"Missing parent node_id={int(req.parent_node_id)} for child node_id={nid}")
            parent.child_node_ids.append(int(nid))
        return int(nid)

    root_nodes_by_prompt: Dict[int, List[int]] = defaultdict(list)
    for req, seg in zip(root_requests, root_segments):
        nid = _append_node(req=req, seg=seg, group_type="root_siblings")
        root_nodes_by_prompt[int(req.prompt_id)].append(int(nid))

    # Root split policy:
    # - Expand up to K retry parents with k=2 children.
    # - Remaining retry roots continue as k=1 (no unresolved retry leaves).
    pending_requests: List[ExpandRequest] = []

    for prompt_id, node_ids in root_nodes_by_prompt.items():
        retry_roots = [
            nid
            for nid in node_ids
            if bool(node_by_id[int(nid)].verify_action_present)
            and int(node_by_id[int(nid)].verify_token_id or -1) == int(retry_token_id)
        ]
        retry_roots.sort(key=lambda nid: int(node_by_id[int(nid)].branch_slot))
        keep = retry_roots[: int(cfg.tree.max_retry_parents_from_root)]
        dropped = retry_roots[int(cfg.tree.max_retry_parents_from_root):]

        for parent_nid in keep:
            parent = node_by_id[int(parent_nid)]
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
                pending_requests.append(
                    ExpandRequest(
                        prompt_id=int(prompt_id),
                        true_digits=list(prompt_meta[int(prompt_id)]["true_digits"]),
                        prefix_ids=list(parent.prompt_ids) + list(parent.full_generated_ids),
                        prefix_attention_mask=list(parent.prompt_attention_mask) + [1] * len(parent.full_generated_ids),
                        path_generated_len=int(parent.path_generated_len_after),
                        retry_depth=int(parent.retry_depth + 1),
                        parent_node_id=int(parent_nid),
                        group_id=int(gid),
                        branch_slot=int(branch_slot),
                    )
                )
                nonroot_request_count += 1

        # Not split because of parent cap: continue with a single route.
        for parent_nid in dropped:
            parent = node_by_id[int(parent_nid)]
            parent.retry_block_reason = "continued_single_after_parent_cap"
            gid = int(group_id_next)
            group_id_next += 1
            groups[gid] = TreeGroup(
                group_id=gid,
                prompt_id=int(prompt_id),
                group_type="retry_single_continue",
                parent_node_id=int(parent_nid),
                member_node_ids=[],
            )
            pending_requests.append(
                ExpandRequest(
                    prompt_id=int(prompt_id),
                    true_digits=list(prompt_meta[int(prompt_id)]["true_digits"]),
                    prefix_ids=list(parent.prompt_ids) + list(parent.full_generated_ids),
                    prefix_attention_mask=list(parent.prompt_attention_mask) + [1] * len(parent.full_generated_ids),
                    path_generated_len=int(parent.path_generated_len_after),
                    retry_depth=int(parent.retry_depth + 1),
                    parent_node_id=int(parent_nid),
                    group_id=int(gid),
                    branch_slot=0,
                )
            )
            nonroot_request_count += 1

    # After root split phase, all retry continuations are k=1 until terminal.
    while pending_requests:
        segs = _run_segment_wave(
            requests=pending_requests,
            tokenizer=tokenizer,
            vllm_engine=vllm_engine,
            max_new_tokens_round=int(cfg.rollout.max_new_tokens),
            max_retry_depth=int(cfg.tree.max_retry_depth),
            answer_token_id=int(answer_token_id),
            finalize_token_id=int(finalize_token_id),
            retry_token_id=int(retry_token_id),
            digit_token_ids=digit_token_ids,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            verify_temperature=float(cfg.rollout.verify_temperature),
            verify_p=float(cfg.rollout.verify_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            digit_temperature=float(cfg.rollout.digit_temperature),
            digit_top_p=float(cfg.rollout.digit_top_p),
            digit_greedy=bool(cfg.rollout.digit_greedy),
        )
        next_pending: List[ExpandRequest] = []
        for req, seg in zip(pending_requests, segs):
            gid = groups[int(req.group_id)].group_type
            nid = _append_node(req=req, seg=seg, group_type=str(gid))
            node = node_by_id[int(nid)]
            if bool(node.verify_action_present) and int(node.verify_token_id or -1) == int(retry_token_id):
                cgid = int(group_id_next)
                group_id_next += 1
                groups[cgid] = TreeGroup(
                    group_id=cgid,
                    prompt_id=int(node.prompt_id),
                    group_type="retry_single_continue",
                    parent_node_id=int(nid),
                    member_node_ids=[],
                )
                next_pending.append(
                    ExpandRequest(
                        prompt_id=int(node.prompt_id),
                        true_digits=list(prompt_meta[int(node.prompt_id)]["true_digits"]),
                        prefix_ids=list(node.prompt_ids) + list(node.full_generated_ids),
                        prefix_attention_mask=list(node.prompt_attention_mask) + [1] * len(node.full_generated_ids),
                        path_generated_len=int(node.path_generated_len_after),
                        retry_depth=int(node.retry_depth + 1),
                        parent_node_id=int(nid),
                        group_id=int(cgid),
                        branch_slot=0,
                    )
                )
                nonroot_request_count += 1
        pending_requests = next_pending

    # Strict invariant: no unresolved retry leaves in normal algorithm.
    for n in nodes:
        if bool(n.verify_action_present) and int(n.verify_token_id or -1) == int(retry_token_id):
            if len(n.child_node_ids) == 0:
                n.retry_block_reason = "exception_missing_retry_child"
                raise RuntimeError(
                    f"Retry node {n.node_id} has no child. This should only happen as an exceptional system failure."
                )

    assign_tree_values_and_advantages(
        nodes=nodes,
        groups=groups,
        retry_token_id=int(retry_token_id),
        c_retry=float(cfg.tree.c_retry),
        gamma=float(cfg.tree.gamma),
        c_branch=float(cfg.tree.c_branch),
        advantage_clip=float(cfg.tree.advantage_clip),
    )

    trajectories: List[Trajectory] = []
    for n in nodes:
        if len(n.actions) == 0:
            continue

        logp_t, entropy_t = _action_logp_entropy_tensors(
            model=model,
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
            "leaf_end_type": str(n.leaf_end_type),
            "was_forced_finalize": bool(n.was_forced_finalize),
            "retry_depth_at_leaf": (
                int(n.retry_depth)
                if str(n.leaf_end_type) in ("model_finalize", "forced_finalize_max_retry")
                else None
            ),
            "verify_action_present": bool(n.verify_action_present),
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
            values_old=[0.0] * len(n.actions),
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
                    "verify_action_present": bool(n.verify_action_present),
                    "leaf_end_type": str(n.leaf_end_type),
                    "was_forced_finalize": bool(n.was_forced_finalize),
                    "retry_depth_at_leaf": (
                        int(n.retry_depth)
                        if str(n.leaf_end_type) in ("model_finalize", "forced_finalize_max_retry")
                        else None
                    ),
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
    stats["num_child_requests"] = float(nonroot_request_count)
    stats["num_trajectories"] = float(len(trajectories))
    return trajectories, stats
