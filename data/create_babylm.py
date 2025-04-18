# prepare_babylm.py
import os, pickle
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# 1) データセット取得（言語や設定はお好みで）
ds = load_dataset("google/babylm", split={"train":"train", "val":"validation"})
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

outdir = "data/babylm"
os.makedirs(outdir, exist_ok=True)

for split, key in [("train","train"), ("val","val")]:
    texts = ds[key]["text"]
    # 2) 全文をトークナイズ → flatten
    ids = []
    for line in texts:
        ids.extend(tokenizer.encode(line))
    arr = np.array(ids, dtype=np.uint16)
    # 3) memmap 形式で保存
    path = os.path.join(outdir, f"{split}.bin")
    fp = np.memmap(path, dtype=np.uint16, mode="w+", shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()
    print(f"wrote {path} ({arr.size} tokens)")

# 4) メタ情報
meta = {"vocab_size": tokenizer.vocab_size}
with open(os.path.join(outdir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print("wrote meta.pkl")
