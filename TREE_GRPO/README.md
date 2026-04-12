# TREE_GRPO (Shallow v1)

Shallow-but-real Tree GRPO trainer that is isolated from `PPO/`.

Current rollout shape:
- Root: 4 siblings per prompt
- Retry expansion: one additional level only
- Expand at most 2 retry parents from root
- Each expanded parent gets 2 children

Credit semantics:
- `Q_F`, `Q_R`, `U=max(Q_F,Q_R)`, `V`
- Z/ANSWER token advantage from group-mean centered `U`
- VERIFY token advantage from local `Q_chosen - 0.5*(Q_F+Q_R)`
- No std normalization

Entry point:
- `python -m TREE_GRPO.train`
