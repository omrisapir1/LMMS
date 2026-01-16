import os
import json
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from phase_0.model import Phase0Model
from phase1.model import Phase1CoconutModel

LOCAL_DIR = "/workspace/runs/phase1/phase1_meta.json"   # where save_phase1_checkpoint wrote files
HF_REPO = "omrisap/LMMS_phase1"       # target HF repo
TORCH_DTYPE = torch.bfloat16


def push_phase1_checkpoint(local_dir: str, hf_repo: str):
    api = HfApi(token=)

    # ─────────────────────────────────────
    # 1) Load tokenizer (defines vocab!)
    # ─────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # ─────────────────────────────────────
    # 2) Load metadata contract
    # ─────────────────────────────────────
    meta_path = os.path.join(local_dir, "phase1_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # ─────────────────────────────────────
    # 3) Load Phase-0
    # ─────────────────────────────────────
    phase0 = Phase0Model.from_pretrained(
        meta["phase0_repo"],
        torch_dtype=TORCH_DTYPE,
    )

    # 🔥 CRITICAL: resize embeddings to tokenizer
    phase0.model.resize_token_embeddings(len(tokenizer))

    # ─────────────────────────────────────
    # 4) Wrap Phase-1
    # ─────────────────────────────────────
    latent_id = tokenizer.convert_tokens_to_ids(meta["latent_token"])

    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )

    # ─────────────────────────────────────
    # 5) Load Phase-1 weights ONLY
    # ─────────────────────────────────────
    weights_path = os.path.join(local_dir, "phase1_weights.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    # ─────────────────────────────────────
    # 6) Prepare HF repo structure
    # ─────────────────────────────────────
    os.makedirs("hf_tmp", exist_ok=True)

    # tokenizer
    tokenizer.save_pretrained("hf_tmp")

    # phase1 weights
    torch.save(state_dict, "hf_tmp/phase1_weights.pt")

    # metadata (rename stays the same)
    with open("hf_tmp/phase1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # optional: minimal README
    with open("hf_tmp/README.md", "w") as f:
        f.write("# LMMS Phase-1 Coconut Model\n")

    # ─────────────────────────────────────
    # 7) Push to HF
    # ─────────────────────────────────────
    api.create_repo(hf_repo, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path="hf_tmp",
        repo_id=hf_repo,
        repo_type="model",
    )

    print(f"✅ Phase-1 pushed to HF: {hf_repo}")


if __name__ == "__main__":
    push_phase1_checkpoint(LOCAL_DIR, HF_REPO)
