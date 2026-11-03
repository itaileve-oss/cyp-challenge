"""Uni-Mol pretrained molecular embeddings (unimolv1, 84M params, 512-dim output).

Pretrained by DP Technology on a large corpus of molecular 3D conformers via
self-supervised learning (masked atom/coordinate prediction) -- a genuinely
different kind of feature from everything else tried this session: a learned,
data-driven representation from a much larger training corpus, not a hand-crafted
physics or geometric descriptor. Framed as an ENSEMBLE component candidate (see
project memory) rather than an expected standalone win -- the leaderboard's own
"CheMeleon-baseline"/"chemfm-baseline" pretrained-embedding entries score worse
than our current fingerprint-based model, so this is a genuine experiment, not an
assumed upgrade.

Uni-Mol generates its own 3D conformers internally (RDKit ETKDG under the hood),
so this only needs a SMILES list, no pre-built conformer ensemble required.

Run in the `GNINA` conda env (has unimol_tools + a GPU-enabled torch build).
"""

import pickle
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_UNIMOL_FEATURES = 512


def unimol_embeddings(smiles_list: list[str]) -> np.ndarray:
    from unimol_tools import UniMolRepr

    clf = UniMolRepr(data_type="molecule", remove_hs=False)
    reprs = clf.get_repr(smiles_list, return_atomic_reprs=False)
    return np.array(reprs, dtype=np.float32)


def get_cached_embeddings(smiles_list: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached Uni-Mol embeddings from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"Computing Uni-Mol embeddings for {len(smiles_list)} molecules...", flush=True)
    start = time.time()
    emb = unimol_embeddings(smiles_list)
    print(f"  done in {time.time() - start:.1f}s", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(emb, f)
    return emb


if __name__ == "__main__":
    import pandas as pd

    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    cache_path = PROJECT_ROOT / "models" / "features_unimol_train_cache.pkl"
    emb = get_cached_embeddings(smiles, cache_path)
    print(f"Embeddings shape: {emb.shape}")
