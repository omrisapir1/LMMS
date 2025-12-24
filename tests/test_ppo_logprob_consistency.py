import torch

@torch.no_grad()
def test_logprob_invariance(
    policy,
    rollout_batch,
    tokenizer,
    device,
    vocab_size=64,
    atol=1e-5,
):
    """
    Assert logprob_old == logprob_new when policy has NOT changed.
    """

    policy.eval()

    actions = torch.tensor(rollout_batch["actions"], device=device)
    logprob_old = torch.tensor(rollout_batch["logprob_old"], device=device)
    phases = rollout_batch["phases"]
    input_ids_steps = rollout_batch["input_ids_steps"]
    attention_mask_steps = rollout_batch["attention_mask_steps"]

    N = len(actions)
    logprob_new = []

    for i in range(N):
        input_ids = torch.tensor(input_ids_steps[i], device=device).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask_steps[i], device=device).unsqueeze(0)

        outputs = policy(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # [1, V]

        # rebuild mask EXACTLY like rollout
        mask = torch.full_like(logits, -1e9)

        if phases[i] == "latent":
            z_ids = tokenizer.convert_tokens_to_ids([f"<z{j}>" for j in range(vocab_size)])
            mask[:, z_ids] = 0.0
        else:
            digit_ids = tokenizer.convert_tokens_to_ids([str(j) for j in range(10)])
            mask[:, digit_ids] = 0.0

        masked_logits = logits + mask

        log_probs = torch.log_softmax(masked_logits, dim=-1)
        lp = log_probs[0, actions[i]].item()
        logprob_new.append(lp)

    logprob_new = torch.tensor(logprob_new, device=device)

    diff = (logprob_old - logprob_new).abs()

    print("max |Δ logprob| =", diff.max().item())

    assert torch.allclose(
        logprob_old,
        logprob_new,
        atol=atol,
    ), "❌ logprob mismatch with frozen policy!"
