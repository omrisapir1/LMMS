import json
import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from phase_0.model import Phase0Model
from phase1.model import Phase1CoconutModel

LATENT_TOKEN = "<|latent|>"

def load_phase1_from_hf(repo_id: str, torch_dtype="bfloat16", device="cuda"):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # load Phase-1 config
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # load Phase-0
    phase0 = Phase0Model.from_pretrained(
        cfg["phase0_repo"],
        torch_dtype=torch_dtype,
    )
    phase0.model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )

    # load Phase-1 weights
    weights_path = hf_hub_download(repo_id, "phase1_weights.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval().to(device)
    return tokenizer, model, cfg


import json
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from phase_0.model import Phase0Model
from phase1.model import Phase1CoconutModel

repo = "omrisap/LMMS_phase1"

# 1) Load tokenizer FIRST (this defines vocab size)
tokenizer = AutoTokenizer.from_pretrained(repo)

# 2) Load meta
meta_path = hf_hub_download(repo, "phase1_meta.json")
with open(meta_path) as f:
    meta = json.load(f)


p0_tokenizer = AutoTokenizer.from_pretrained(meta["phase0_repo"])

# 🔥 transplant chat template
tokenizer.chat_template = p0_tokenizer.chat_template

# 3) Load Phase-0
phase0 = Phase0Model.from_pretrained(
    meta["phase0_repo"],
    torch_dtype=torch.bfloat16,
)

# 🔥 CRITICAL STEP — resize embeddings to tokenizer vocab
phase0.model.resize_token_embeddings(len(tokenizer))

# 4) Wrap Phase-1
latent_id = tokenizer.convert_tokens_to_ids(meta["latent_token"])

model = Phase1CoconutModel(
    phase0_model=phase0,
    latent_token_id=latent_id,
)

# 5) Load Phase-1 weights
weights_path = hf_hub_download(repo, "phase1_weights.pt")
state_dict = torch.load(weights_path, map_location="cpu")

model.load_state_dict(state_dict, strict=True)
model.eval()
model.to('cuda')