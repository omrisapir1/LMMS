from __future__ import annotations

import random
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


def _depth_prob(values: Sequence[float], depth: int) -> float:
    if len(values) == 0:
        raise RuntimeError("Branching probability list must be non-empty")
    idx = int(depth)
    if idx < 0:
        idx = 0
    if idx >= len(values):
        idx = len(values) - 1
    return float(values[idx])


def _sample_branch_k(cfg: Config, retry_depth: int) -> Tuple[int, str]:
    p4 = _depth_prob(cfg.tree.tree_p4_by_depth, retry_depth)
    p2 = _depth_prob(cfg.tree.tree_p2_by_depth, retry_depth)
    p1 = _depth_prob(cfg.tree.tree_p1_by_depth, retry_depth)
    total = float(p4 + p2 + p1)
    if total <= 0.0:
        raise RuntimeError(f"Invalid branching probabilities at depth={retry_depth}: sum={total}")
    p4n = p4 / total
    p2n = p2 / total
    r = random.random()
    if r < p4n:
        return 4, "p4"
    if r < (p4n + p2n):
        return 2, "p2"
    return 1, "p1"


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
    # Prefix-aware grouped digit sampling (reuses prefill for identical prefixes).
    digit_row_by_req_idx: Dict[int, List[int]] = {}
    digit_prefix_to_req_idxs: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for req_idx in active_idx:
        prompt_ids_key = tuple(int(x) for x in (requests[req_idx].prefix_ids + states[req_idx].full_generated_ids))
        digit_prefix_to_req_idxs[prompt_ids_key].append(int(req_idx))
    for key, req_idxs in digit_prefix_to_req_idxs.items():
        ordered_req_idxs = sorted(req_idxs, key=lambda i: int(requests[i].branch_slot))
        k = int(len(ordered_req_idxs))
        if supports_token_prompts:
            rows_group = vllm_engine.generate_digits(
                prompt_token_ids=[list(key)],
                num_samples_per_prompt=k,
                num_digits=5,
                temperature=float(digit_temperature),
                top_p=float(digit_top_p),
                greedy=bool(digit_greedy),
                min_p=0.0,
                repetition_penalty=1.0,
            )
        else:
            rows_group = vllm_engine.generate_digits(
                prompts=[_decode_cached(key)],
                num_samples_per_prompt=k,
                num_digits=5,
                temperature=float(digit_temperature),
                top_p=float(digit_top_p),
                greedy=bool(digit_greedy),
                min_p=0.0,
                repetition_penalty=1.0,
            )
        if len(rows_group) != k:
            raise RuntimeError(
                f"Tree wave grouped digit row count mismatch: got={len(rows_group)} expected={k}"
            )
        for local_i, req_idx in enumerate(ordered_req_idxs):
            digit_row_by_req_idx[int(req_idx)] = [int(x) for x in list(rows_group[local_i])]

    for req_idx in active_idx:
        st = states[req_idx]
        digits = list(digit_row_by_req_idx.get(int(req_idx), []))
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
        verify_row_by_req_idx: Dict[int, List[int]] = {}
        verify_prefix_to_req_idxs: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for local_i, req_idx in enumerate(need_verify):
            verify_prefix_to_req_idxs[tuple(int(x) for x in verify_prompt_ids[local_i])].append(int(req_idx))

        for key, req_idxs in verify_prefix_to_req_idxs.items():
            ordered_req_idxs = sorted(req_idxs, key=lambda i: int(requests[i].branch_slot))
            k = int(len(ordered_req_idxs))
            if supports_token_prompts:
                rows_group = vllm_engine.generate_verify(
                    prompt_token_ids=[list(key)],
                    num_samples_per_prompt=k,
                    temperature=float(verify_temperature),
                    top_p=float(verify_p),
                    greedy=False,
                    min_p=float(min_p),
                    repetition_penalty=float(repetition_penalty),
                )
            else:
                rows_group = vllm_engine.generate_verify(
                    prompts=[_decode_cached(key)],
                    num_samples_per_prompt=k,
                    temperature=float(verify_temperature),
                    top_p=float(verify_p),
                    greedy=False,
                    min_p=float(min_p),
                    repetition_penalty=float(repetition_penalty),
                )
            if len(rows_group) != k:
                raise RuntimeError(
                    f"Tree wave grouped verify row count mismatch: got={len(rows_group)} expected={k}"
                )
            for local_i, req_idx in enumerate(ordered_req_idxs):
                verify_row_by_req_idx[int(req_idx)] = [int(x) for x in list(rows_group[local_i])]

        for req_idx in need_verify:
            st = states[req_idx]
            row = list(verify_row_by_req_idx.get(int(req_idx), []))
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


def _run_forced_retry_probes(
    *,
    nodes: Sequence[TreeNode],
    prompt_meta: Dict[int, Dict[str, object]],
    tokenizer,
    vllm_engine: Any,
    cfg: Config,
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    digit_token_ids: Sequence[int],
    probe_node_id_start: int,
) -> Tuple[Dict[int, Dict[str, object]], List[Dict[str, object]], int, Dict[str, int], set[int]]:
    """
    For finalize-chosen nodes (where retry is still legal), run a single
    forced-retry linear probe to terminal and return per-source probe results.
    Probe nodes are rollout-only and must not be added to PPO training rows.
    """
    eligible_sources: List[TreeNode] = []
    for n in nodes:
        if not bool(n.verify_action_present):
            continue
        if int(n.verify_token_id or -1) != int(finalize_token_id):
            continue
        if int(n.retry_depth) >= int(cfg.tree.max_retry_depth):
            continue
        eligible_sources.append(n)

    if len(eligible_sources) == 0:
        return {}, [], int(probe_node_id_start), {"candidates": 0, "launched": 0, "skipped_by_cap": 0}, set()

    selected_source_ids: set[int] = set()
    skipped_by_cap: set[int] = set()
    by_prompt: Dict[int, List[TreeNode]] = defaultdict(list)
    for n in eligible_sources:
        by_prompt[int(n.prompt_id)].append(n)
    cap = int(getattr(cfg.tree, "max_probes_per_prompt", 0))
    selected_sources: List[TreeNode] = []
    for _pid, rows in by_prompt.items():
        if cap > 0 and len(rows) > cap:
            random.shuffle(rows)
            keep = rows[:cap]
            drop = rows[cap:]
            selected_sources.extend(keep)
            selected_source_ids.update(int(x.node_id) for x in keep)
            skipped_by_cap.update(int(x.node_id) for x in drop)
        else:
            selected_sources.extend(rows)
            selected_source_ids.update(int(x.node_id) for x in rows)

    source_state: Dict[int, Dict[str, object]] = {}
    pending_pairs: List[Tuple[int, ExpandRequest]] = []
    for src in selected_sources:
        src_id = int(src.node_id)
        source_state[src_id] = {
            "start_retry_depth": int(src.retry_depth),
            "rounds": 0,
            "terminal_value": None,
            "terminal_node_id": None,
            "terminal_leaf_end_type": None,
            "probe_nodes": [],
        }
        if len(src.full_generated_ids) == 0:
            raise RuntimeError(f"Source finalize node {src_id} has empty full_generated_ids")
        if int(src.full_generated_ids[-1]) != int(finalize_token_id):
            raise RuntimeError(f"Source node {src_id} is finalize-chosen but last token is not <FINALIZE>")

        src_prefix_add = list(src.full_generated_ids[:-1])  # omit sampled verify token
        next_prefix_ids = list(src.prompt_ids) + src_prefix_add
        next_prefix_attention_mask = list(src.prompt_attention_mask) + [1] * len(src_prefix_add)
        pending_pairs.append(
            (
                src_id,
                ExpandRequest(
                    prompt_id=int(src.prompt_id),
                    true_digits=list(prompt_meta[int(src.prompt_id)]["true_digits"]),
                    prefix_ids=next_prefix_ids,
                    prefix_attention_mask=next_prefix_attention_mask,
                    path_generated_len=max(int(src.path_generated_len_after) - 1, 0),
                    retry_depth=int(src.retry_depth + 1),  # forced retry transition
                    parent_node_id=None,
                    group_id=-1,
                    branch_slot=0,
                ),
            )
        )

    probe_rows: List[Dict[str, object]] = []
    probe_node_id_next = int(probe_node_id_start)
    while pending_pairs:
        reqs = [req for _, req in pending_pairs]
        segs = _run_segment_wave(
            requests=reqs,
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
        if len(segs) != len(pending_pairs):
            raise RuntimeError("Forced-retry probe wave size mismatch")

        next_pairs: List[Tuple[int, ExpandRequest]] = []
        for (src_id, req), seg in zip(pending_pairs, segs):
            st = source_state[int(src_id)]
            probe_node_id = int(probe_node_id_next)
            probe_node_id_next += 1
            q = 1.0 if (seg.pred_digits is not None and list(seg.pred_digits) == list(req.true_digits)) else 0.0
            probe_row = {
                "probe_node_id": probe_node_id,
                "is_forced_retry_probe": True,
                "probe_source_node_id": int(src_id),
                "prompt_id": int(req.prompt_id),
                "retry_depth": int(req.retry_depth),
                "verify_action_present": bool(seg.verify_action_present),
                "verify_token_id": (None if seg.verify_token_id is None else int(seg.verify_token_id)),
                "leaf_end_type": str(seg.leaf_end_type),
                "was_forced_finalize": bool(seg.was_forced_finalize),
                "pred_digits": (None if seg.pred_digits is None else list(seg.pred_digits)),
                "q": float(q),
            }
            st["probe_nodes"].append(probe_row)
            probe_rows.append(probe_row)
            st["rounds"] = int(st["rounds"]) + 1

            is_retry = bool(seg.verify_action_present) and int(seg.verify_token_id or -1) == int(retry_token_id)
            if is_retry:
                next_pairs.append(
                    (
                        int(src_id),
                        ExpandRequest(
                            prompt_id=int(req.prompt_id),
                            true_digits=list(req.true_digits),
                            prefix_ids=list(seg.next_prefix_ids),
                            prefix_attention_mask=list(seg.next_prefix_attention_mask),
                            path_generated_len=int(seg.next_path_generated_len),
                            retry_depth=int(req.retry_depth + 1),
                            parent_node_id=None,
                            group_id=-1,
                            branch_slot=0,
                        ),
                    )
                )
            else:
                st["terminal_value"] = float(q)
                st["terminal_node_id"] = int(probe_node_id)
                st["terminal_leaf_end_type"] = str(seg.leaf_end_type)
        pending_pairs = next_pairs

    out: Dict[int, Dict[str, object]] = {}
    for src_id, st in source_state.items():
        if st["terminal_value"] is None:
            raise RuntimeError(f"Forced-retry probe for node {src_id} did not terminate")
        out[int(src_id)] = {
            "has_forced_retry_probe": True,
            "probe_terminal_value": float(st["terminal_value"]),
            "probe_terminal_node_id": int(st["terminal_node_id"]),
            "probe_length_rounds": int(st["rounds"]),
            "probe_leaf_end_type": str(st["terminal_leaf_end_type"]),
            "probe_start_retry_depth": int(st["start_retry_depth"]),
            "probe_nodes": list(st["probe_nodes"]),
        }
    stats = {
        "candidates": int(len(eligible_sources)),
        "launched": int(len(selected_sources)),
        "skipped_by_cap": int(len(skipped_by_cap)),
    }
    return out, probe_rows, int(probe_node_id_next), stats, skipped_by_cap


def collect_tree_grpo_v1_batch(
    *,
    model=None,
    tokenizer,
    vllm_engine: Any,
    prepared: Sequence[Dict[str, object]],
    cfg: Config,
    z_allowed_t: Optional[torch.Tensor] = None,
    digit_allowed_t: Optional[torch.Tensor] = None,
    verify_allowed_t: Optional[torch.Tensor] = None,
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

    def _append_node(
        req: ExpandRequest,
        seg: SegmentResult,
        *,
        group_type: str,
        k_used: int = 0,
        branching_decision: str = "not_retry",
    ) -> int:
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
            k_used=int(k_used),
            branching_decision=str(branching_decision),
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

    prompt_total_nodes: Dict[int, int] = defaultdict(int)
    prompt_reserved_nodes: Dict[int, int] = defaultdict(int)
    prompt_live_paths: Dict[int, int] = defaultdict(int)
    prompt_terminal_leaves: Dict[int, int] = defaultdict(int)
    expanded_split_by_prompt_level: Dict[Tuple[int, int], int] = defaultdict(int)

    for n in nodes:
        pid = int(n.prompt_id)
        prompt_total_nodes[pid] += 1
        is_retry = bool(n.verify_action_present) and int(n.verify_token_id or -1) == int(retry_token_id)
        if is_retry:
            prompt_live_paths[pid] += 1
        else:
            prompt_terminal_leaves[pid] += 1

    pending_requests: List[ExpandRequest] = []

    def _select_k_with_budgets(parent: TreeNode) -> Tuple[int, str]:
        pid = int(parent.prompt_id)
        d = int(parent.retry_depth)
        sampled_k, sampled_decision = _sample_branch_k(cfg, d)
        candidates: List[int]
        if sampled_k == 4:
            candidates = [4, 2, 1]
        elif sampled_k == 2:
            candidates = [2, 1]
        else:
            candidates = [1]

        for cand in candidates:
            if cand > 1 and int(expanded_split_by_prompt_level[(pid, d)]) >= int(
                cfg.tree.max_expanded_retry_nodes_per_level
            ):
                continue
            if cand > 1 and int(prompt_total_nodes[pid] + prompt_reserved_nodes[pid] + cand) > int(
                cfg.tree.max_total_nodes_per_prompt
            ):
                continue
            projected_final_leaves = int(prompt_terminal_leaves[pid] + prompt_live_paths[pid] - 1 + cand)
            if cand > 1 and projected_final_leaves > int(cfg.tree.max_leaves_per_prompt):
                continue

            if cand > 1:
                expanded_split_by_prompt_level[(pid, d)] += 1
            prompt_live_paths[pid] = int(prompt_live_paths[pid] - 1 + cand)
            prompt_reserved_nodes[pid] += int(cand)
            decision = sampled_decision if cand == sampled_k else "downgraded_due_to_budget"
            return int(cand), str(decision)

        # Strict invariant: retry continuation is always at least k=1.
        prompt_live_paths[pid] = int(prompt_live_paths[pid])
        prompt_reserved_nodes[pid] += 1
        return 1, "downgraded_due_to_budget"

    def _enqueue_children(parent: TreeNode) -> None:
        nonlocal group_id_next, nonroot_request_count
        k, decision = _select_k_with_budgets(parent)
        parent.k_used = int(k)
        parent.branching_decision = str(decision)
        if decision == "downgraded_due_to_budget":
            parent.retry_block_reason = "downgraded_due_to_budget"

        gid = int(group_id_next)
        group_id_next += 1
        gtype = "retry_children" if int(k) > 1 else "retry_single_continue"
        groups[gid] = TreeGroup(
            group_id=gid,
            prompt_id=int(parent.prompt_id),
            group_type=gtype,
            parent_node_id=int(parent.node_id),
            member_node_ids=[],
        )
        for branch_slot in range(int(k)):
            pending_requests.append(
                ExpandRequest(
                    prompt_id=int(parent.prompt_id),
                    true_digits=list(prompt_meta[int(parent.prompt_id)]["true_digits"]),
                    prefix_ids=list(parent.prompt_ids) + list(parent.full_generated_ids),
                    prefix_attention_mask=list(parent.prompt_attention_mask) + [1] * len(parent.full_generated_ids),
                    path_generated_len=int(parent.path_generated_len_after),
                    retry_depth=int(parent.retry_depth + 1),
                    parent_node_id=int(parent.node_id),
                    group_id=int(gid),
                    branch_slot=int(branch_slot),
                )
            )
            nonroot_request_count += 1

    # Root is always 4 siblings; branching policy starts at retry children decisions.
    for prompt_id, node_ids in root_nodes_by_prompt.items():
        ordered = sorted(node_ids, key=lambda nid: int(node_by_id[int(nid)].branch_slot))
        for nid in ordered:
            n = node_by_id[int(nid)]
            is_retry = bool(n.verify_action_present) and int(n.verify_token_id or -1) == int(retry_token_id)
            if is_retry:
                _enqueue_children(n)

    max_active_wave = max(int(cfg.tree.max_active_nodes_per_wave), 1)
    while pending_requests:
        pending_requests.sort(
            key=lambda r: (
                int(r.retry_depth),
                -int(len(r.prefix_ids)),
                int(r.prompt_id),
                int(r.parent_node_id) if r.parent_node_id is not None else -1,
                int(r.branch_slot),
            )
        )
        wave_reqs = pending_requests[:max_active_wave]
        pending_requests = pending_requests[max_active_wave:]

        segs = _run_segment_wave(
            requests=wave_reqs,
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
        for req, seg in zip(wave_reqs, segs):
            pid = int(req.prompt_id)
            prompt_reserved_nodes[pid] = max(int(prompt_reserved_nodes[pid] - 1), 0)
            gid = groups[int(req.group_id)].group_type
            nid = _append_node(req=req, seg=seg, group_type=str(gid))
            prompt_total_nodes[pid] += 1
            node = node_by_id[int(nid)]
            is_retry = bool(node.verify_action_present) and int(node.verify_token_id or -1) == int(retry_token_id)
            if is_retry:
                _enqueue_children(node)
            else:
                prompt_live_paths[pid] = int(prompt_live_paths[pid] - 1)
                prompt_terminal_leaves[pid] += 1

    # Strict invariant: no unresolved retry leaves in normal algorithm.
    for n in nodes:
        if bool(n.verify_action_present) and int(n.verify_token_id or -1) == int(retry_token_id):
            if len(n.child_node_ids) == 0:
                n.retry_block_reason = "exception_missing_retry_child"
                raise RuntimeError(
                    f"Retry node {n.node_id} has no child. This should only happen as an exceptional system failure."
                )

    probe_results, _, _, probe_stats, probe_skipped_by_cap = _run_forced_retry_probes(
        nodes=nodes,
        prompt_meta=prompt_meta,
        tokenizer=tokenizer,
        vllm_engine=vllm_engine,
        cfg=cfg,
        answer_token_id=int(answer_token_id),
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
        digit_token_ids=digit_token_ids,
        probe_node_id_start=int(node_id_next),
    )
    for source_id, result in probe_results.items():
        src = node_by_id.get(int(source_id))
        if src is None:
            raise RuntimeError(f"Probe source node missing: {source_id}")
        src.has_forced_retry_probe = bool(result["has_forced_retry_probe"])
        src.probe_terminal_value = float(result["probe_terminal_value"])
        src.probe_terminal_node_id = int(result["probe_terminal_node_id"])
        src.probe_length_rounds = int(result["probe_length_rounds"])
        src.probe_leaf_end_type = str(result["probe_leaf_end_type"])
        src.probe_start_retry_depth = int(result["probe_start_retry_depth"])

    assign_tree_values_and_advantages(
        nodes=nodes,
        groups=groups,
        retry_token_id=int(retry_token_id),
        max_retry_depth=int(cfg.tree.max_retry_depth),
        c_retry=float(cfg.tree.c_retry),
        gamma=float(cfg.tree.gamma),
        c_branch=float(cfg.tree.c_branch),
        advantage_clip=float(cfg.tree.advantage_clip),
    )

    # Compact k=1 retry chains: structural nodes are branch points (k>1) and leaves.
    start_ids: List[int] = []
    for n in nodes:
        pid = int(n.parent_node_id) if n.parent_node_id is not None else None
        if pid is None:
            start_ids.append(int(n.node_id))
            continue
        p = node_by_id.get(int(pid))
        if p is None:
            start_ids.append(int(n.node_id))
            continue
        if int(p.k_used) > 1:
            start_ids.append(int(n.node_id))

    start_ids = sorted(set(start_ids))
    raw_chain_by_start: Dict[int, List[int]] = {}
    start_for_raw: Dict[int, int] = {}
    for sid in start_ids:
        chain: List[int] = []
        cur = int(sid)
        while True:
            if cur in start_for_raw:
                break
            node = node_by_id[int(cur)]
            chain.append(int(cur))
            start_for_raw[int(cur)] = int(sid)
            can_linear_merge = (
                bool(node.verify_action_present)
                and int(node.verify_token_id or -1) == int(retry_token_id)
                and int(node.k_used) == 1
                and len(node.child_node_ids) == 1
            )
            if not can_linear_merge:
                break
            nxt = int(node.child_node_ids[0])
            if nxt not in node_by_id:
                break
            cur = int(nxt)
        raw_chain_by_start[int(sid)] = list(chain)

    compact_ids = {sid: i for i, sid in enumerate(start_ids)}

    trajectories: List[Trajectory] = []
    for sid in start_ids:
        chain_ids = raw_chain_by_start[int(sid)]
        chain_nodes = [node_by_id[int(x)] for x in chain_ids]
        if len(chain_nodes) == 0:
            continue
        first = chain_nodes[0]
        last = chain_nodes[-1]

        actions: List[int] = []
        action_types: List[str] = []
        full_generated_ids: List[int] = []
        returns: List[float] = []
        advantages: List[float] = []
        rounds_meta: List[Dict[str, object]] = []
        for rn in chain_nodes:
            actions.extend(list(rn.actions))
            action_types.extend(list(rn.action_types))
            full_generated_ids.extend(list(rn.full_generated_ids))
            for t in rn.action_types:
                if str(t) == "verify":
                    returns.append(float(rn.V))
                    advantages.append(float(rn.A_V))
                else:
                    returns.append(float(rn.U))
                    advantages.append(float(rn.A_Z))
            rounds_meta.append(
                {
                    "tree_node_id": int(compact_ids[int(sid)]),
                    "round_raw_node_id": int(rn.node_id),
                    "group_id": int(rn.group_id),
                    "group_type": str(rn.group_type),
                    "retry_depth": int(rn.retry_depth),
                    "parent_node_id": int(rn.parent_node_id) if rn.parent_node_id is not None else None,
                    "z_token_ids": list(rn.z_token_ids),
                    "digit_token_ids": list(rn.digit_token_ids),
                    "pred_digits": (None if rn.pred_digits is None else list(rn.pred_digits)),
                    "verify_token_id": (None if rn.verify_token_id is None else int(rn.verify_token_id)),
                    "verify_action_present": bool(rn.verify_action_present),
                    "k_used": int(rn.k_used),
                    "branching_decision": str(rn.branching_decision),
                    "continued_linearly": bool(int(rn.k_used) == 1 and len(rn.child_node_ids) == 1),
                    "leaf_end_type": str(rn.leaf_end_type),
                    "was_forced_finalize": bool(rn.was_forced_finalize),
                    "retry_depth_at_leaf": (
                        int(rn.retry_depth)
                        if str(rn.leaf_end_type) in ("model_finalize", "forced_finalize_max_retry")
                        else None
                    ),
                    "q": float(rn.q),
                    "Q_F": float(rn.Q_F),
                    "Q_R": (None if rn.Q_R is None else float(rn.Q_R)),
                    "U": float(rn.U),
                    "V": float(rn.V),
                    "A_Z": float(rn.A_Z),
                    "A_V": float(rn.A_V),
                    "has_forced_retry_probe": bool(rn.has_forced_retry_probe),
                    "probe_skipped_by_cap": bool(int(rn.node_id) in probe_skipped_by_cap),
                    "probe_terminal_value": (
                        None if rn.probe_terminal_value is None else float(rn.probe_terminal_value)
                    ),
                    "probe_terminal_node_id": (
                        None if rn.probe_terminal_node_id is None else int(rn.probe_terminal_node_id)
                    ),
                    "probe_length_rounds": (
                        None if rn.probe_length_rounds is None else int(rn.probe_length_rounds)
                    ),
                    "probe_leaf_end_type": (
                        None if rn.probe_leaf_end_type is None else str(rn.probe_leaf_end_type)
                    ),
                    "probe_start_retry_depth": (
                        None if rn.probe_start_retry_depth is None else int(rn.probe_start_retry_depth)
                    ),
                    "probe_nodes": (
                        []
                        if int(rn.node_id) not in probe_results
                        else list(probe_results[int(rn.node_id)].get("probe_nodes", []))
                    ),
                    "retry_block_reason": str(rn.retry_block_reason),
                }
            )

        if len(returns) != len(actions):
            raise RuntimeError("Tree returns length mismatch after linear compaction")
        if len(advantages) != len(actions):
            raise RuntimeError("Tree advantages length mismatch after linear compaction")

        compute_old_logp = (
            model is not None
            and z_allowed_t is not None
            and digit_allowed_t is not None
            and verify_allowed_t is not None
        )
        if compute_old_logp:
            logp_t, entropy_t = _action_logp_entropy_tensors(
                model=model,
                prompt_ids=first.prompt_ids,
                prompt_attention_mask=first.prompt_attention_mask,
                actions=actions,
                action_types=action_types,
                z_allowed_t=z_allowed_t,
                digit_allowed_t=digit_allowed_t,
                verify_allowed_t=verify_allowed_t,
                temperature=float(cfg.rollout.temperature),
            )
            logp_old_list = logp_t.float().cpu().tolist()
            entropy_old_list = entropy_t.float().cpu().tolist()
        else:
            logp_old_list = []
            entropy_old_list = []

        compact_child_ids: List[int] = []
        for raw_child in last.child_node_ids:
            sid_child = start_for_raw.get(int(raw_child))
            if sid_child is None:
                continue
            compact_child_ids.append(int(compact_ids[int(sid_child)]))
        compact_child_ids = sorted(set(compact_child_ids))

        compact_parent_id: Optional[int] = None
        raw_parent = first.parent_node_id
        if raw_parent is not None:
            parent_start = start_for_raw.get(int(raw_parent))
            if parent_start is not None and int(parent_start) in compact_ids:
                compact_parent_id = int(compact_ids[int(parent_start)])

        prompt_info = prompt_meta[int(first.prompt_id)]
        reward_info: Dict[str, object] = {
            "reward_full": int(1 if float(last.q) >= 1.0 else 0),
            "reward_partial": float(last.q),
            "reward": float(last.q),
            "reward_final": float(last.V),
            "exact_match": bool(float(last.q) >= 1.0),
            "q": float(last.q),
            "Q_F": float(last.Q_F),
            "Q_R": (None if last.Q_R is None else float(last.Q_R)),
            "U": float(last.U),
            "V": float(last.V),
            "A_Z": float(last.A_Z),
            "A_V": float(last.A_V),
            "group_id": int(first.group_id),
            "group_type": str(first.group_type),
            "retry_depth": int(last.retry_depth),
            "parent_node_id": compact_parent_id,
            "child_node_ids": compact_child_ids,
            "retry_block_reason": str(last.retry_block_reason),
            "terminated_reason": str(last.terminated_reason),
            "leaf_end_type": str(last.leaf_end_type),
            "was_forced_finalize": bool(last.was_forced_finalize),
            "retry_depth_at_leaf": (
                int(last.retry_depth)
                if str(last.leaf_end_type) in ("model_finalize", "forced_finalize_max_retry")
                else None
            ),
            "verify_action_present": bool(last.verify_action_present),
            "k_used": int(last.k_used),
            "branching_decision": str(last.branching_decision),
            "continued_linearly": bool(len(chain_ids) > 1),
            "has_forced_retry_probe": bool(last.has_forced_retry_probe),
            "probe_skipped_by_cap": bool(int(last.node_id) in probe_skipped_by_cap),
            "probe_terminal_value": (None if last.probe_terminal_value is None else float(last.probe_terminal_value)),
            "probe_terminal_node_id": (
                None if last.probe_terminal_node_id is None else int(last.probe_terminal_node_id)
            ),
            "probe_length_rounds": (None if last.probe_length_rounds is None else int(last.probe_length_rounds)),
            "probe_leaf_end_type": (None if last.probe_leaf_end_type is None else str(last.probe_leaf_end_type)),
            "probe_start_retry_depth": (
                None if last.probe_start_retry_depth is None else int(last.probe_start_retry_depth)
            ),
            "probe_nodes": (
                []
                if int(last.node_id) not in probe_results
                else list(probe_results[int(last.node_id)].get("probe_nodes", []))
            ),
            "tree_mode": "depth_prob_structural_nodes",
            "structural_round_count": int(len(chain_ids)),
            "raw_node_chain_ids": list(chain_ids),
            "logp_pending": bool(not compute_old_logp),
        }

        traj = Trajectory(
            prompt_id=int(first.prompt_id),
            sample_id=f"{str(prompt_info['sample_id_base'])}_n{int(compact_ids[int(sid)])}",
            question=str(prompt_info["question"]),
            prompt_ids=list(first.prompt_ids),
            prompt_attention_mask=list(first.prompt_attention_mask),
            actions=list(actions),
            action_types=list(action_types),
            logp_old=list(logp_old_list),
            values_old=[0.0] * len(actions),
            entropy_old=list(entropy_old_list),
            terminated_by=str(last.terminated_reason),
            generated_z_ids=list(first.z_token_ids),
            generated_digit_ids=list(last.digit_token_ids),
            digit_logits=None,
            digit_probs=None,
            digit_pred=(None if last.pred_digits is None else list(last.pred_digits)),
            digit_true=[int(x) for x in list(prompt_info["true_digits"])],
            reward_info=reward_info,
            num_generated_total=int(len(full_generated_ids)),
            num_digits_generated=int(sum(len(x.digit_token_ids) for x in chain_nodes)),
            generated_verify_ids=(
                []
                if last.verify_token_id is None
                else [int(last.verify_token_id)]
            ),
            rounds_meta=rounds_meta,
            full_generated_ids=list(full_generated_ids),
            termination_reason=str(last.terminated_reason),
        )
        traj.returns = list(returns)
        traj.advantages = list(advantages)
        traj.advantages_norm_global = []
        traj.advantages_norm_prompt = []
        traj.advantages_norm = list(advantages)
        trajectories.append(traj)

    stats = tree_summary(nodes=nodes, retry_token_id=int(retry_token_id))
    stats["num_structural_nodes"] = float(len(start_ids))
    stats["num_groups"] = float(len(groups))
    stats["num_root_requests"] = float(len(root_requests))
    stats["num_child_requests"] = float(nonroot_request_count)
    stats["num_trajectories"] = float(len(trajectories))
    stats["num_budget_exhausted_no_k1"] = 0.0
    stats["num_probe_candidates"] = float(probe_stats.get("candidates", 0))
    stats["num_probes_launched"] = float(probe_stats.get("launched", 0))
    stats["num_probes_skipped_by_cap"] = float(probe_stats.get("skipped_by_cap", 0))
    return trajectories, stats
