from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")

print("Downloading nvidia/Daring-Anteater ...")
da = load_dataset("nvidia/Daring-Anteater")
da_out = PROJECT_ROOT / "data" / "raw" / "datasets" / "daring_anteater"
da_out.mkdir(parents=True, exist_ok=True)
da.save_to_disk(str(da_out))
print(f"Saved Daring-Anteater to: {da_out}")

print("Downloading Anthropic/hh-rlhf ...")
hh = load_dataset("Anthropic/hh-rlhf")
hh_out = PROJECT_ROOT / "data" / "raw" / "datasets" / "hh_rlhf"
hh_out.mkdir(parents=True, exist_ok=True)
hh.save_to_disk(str(hh_out))
print(f"Saved HH-RLHF to: {hh_out}")

print("Done.")
