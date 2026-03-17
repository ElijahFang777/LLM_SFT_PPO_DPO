from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path("/mnt/Data2/zfang/rlhf_assignment")

model_path = PROJECT_ROOT / "data" / "raw" / "models" / "Qwen2.5-1.5B"
da_path = PROJECT_ROOT / "data" / "raw" / "datasets" / "daring_anteater"
hh_path = PROJECT_ROOT / "data" / "raw" / "datasets" / "hh_rlhf"

tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
print("Tokenizer loaded.")
print("EOS token:", tok.eos_token)

da = load_from_disk(str(da_path))
print("\nDaring-Anteater:")
print(da)
for split in da.keys():
    print(split, len(da[split]))

hh = load_from_disk(str(hh_path))
print("\nHH-RLHF:")
print(hh)
for split in hh.keys():
    print(split, len(hh[split]))
