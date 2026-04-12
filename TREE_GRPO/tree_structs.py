from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TreeGroup:
    group_id: int
    prompt_id: int
    group_type: str  # "root_siblings" | "retry_children"
    parent_node_id: Optional[int]
    member_node_ids: List[int] = field(default_factory=list)


@dataclass
class ExpandRequest:
    prompt_id: int
    true_digits: List[int]
    prefix_ids: List[int]
    prefix_attention_mask: List[int]
    path_generated_len: int
    retry_depth: int
    parent_node_id: Optional[int]
    group_id: int
    branch_slot: int


@dataclass
class SegmentResult:
    z_token_ids: List[int]
    has_answer: bool
    digit_token_ids: List[int]
    pred_digits: Optional[List[int]]
    verify_token_id: Optional[int]
    actions: List[int]
    action_types: List[str]
    full_generated_ids: List[int]
    next_prefix_ids: List[int]
    next_prefix_attention_mask: List[int]
    next_path_generated_len: int
    terminated_reason: str
    was_forced_finalize: bool
    verify_action_present: bool
    leaf_end_type: str


@dataclass
class TreeNode:
    node_id: int
    prompt_id: int
    parent_node_id: Optional[int]
    retry_depth: int
    group_id: int
    group_type: str
    branch_slot: int
    path_generated_len_before: int
    path_generated_len_after: int

    prompt_ids: List[int]
    prompt_attention_mask: List[int]

    z_token_ids: List[int]
    digit_token_ids: List[int]
    pred_digits: Optional[List[int]]
    verify_token_id: Optional[int]
    actions: List[int]
    action_types: List[str]
    full_generated_ids: List[int]

    child_node_ids: List[int] = field(default_factory=list)
    retry_block_reason: str = "none"

    q: float = 0.0
    Q_F: float = 0.0
    Q_R: float = 0.0
    U: float = 0.0
    V: float = 0.0
    A_Z: float = 0.0
    A_V: float = 0.0
    terminated_reason: str = "max_new_tokens"
    was_forced_finalize: bool = False
    verify_action_present: bool = False
    leaf_end_type: str = "model_finalize"
