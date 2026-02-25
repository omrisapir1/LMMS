# pip install -U huggingface_hub transformers

import os
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer  # or AutoModelForCausalLM if relevant
import torch

# ====== CONFIG ======
HF_TOKEN = ""   # <- you said you'll provide this
LOCAL_DIR = "runs/phase1/last_checkpoint/"  # folder that contains model + tokenizer files
REPO_ID = "omrisap/phase1_improved"  # e.g. "omrisap/my-awesome-model"
PRIVATE = False  # set False if you want public repo
# ====================

# 1) Login
login(token=HF_TOKEN)

# 2) Create repo if it doesn't exist (safe to call repeatedly)
api = HfApi()
api.create_repo(repo_id=REPO_ID, private=PRIVATE, exist_ok=True)

# 3) Load from local folder (choose the model class that matches your model)
local_dir = Path(LOCAL_DIR)

model = AutoModelForCausalLM.from_pretrained(local_dir,torch_dtype=torch.bfloat16,)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# 4) Push model + tokenizer
model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)

print(f"Done! Pushed model and tokenizer to https://huggingface.co/{REPO_ID}")