from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")
MODEL_ID = "Qwen/Qwen2.5-1.5B"
LOCAL_DIR = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"

LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
)

print(f"Model downloaded to: {path}")
