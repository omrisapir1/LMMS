import argparse
from phase1.generate_codebook_dataset import run
from pathlib import Path
from huggingface_hub import snapshot_download

# ====== CONFIG ======

REPO_ID = "omrisap/phase1_improved"   # e.g. "omrisap/my-model"
LOCAL_DIR = "model_ckpt/"      # <- you'll provide
# ====================

Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False # copies files into the folder
)

print(f"Downloaded {REPO_ID} to {LOCAL_DIR}")
# args = argparse.Namespace(
#     ckpt_dir=LOCAL_DIR,
#     dataset_name="omrisap/LMMS_numina_250K",
#     split="train",
#     output_dir="res/",
#     k_max=20,
#     batch_size=100,
#     max_rows=100,
#     shard_size=100,
#     eval_rows_limit=100,
#     use_kv_cache=True,
#     debug_parity_checks=False
# )
# run(args)
