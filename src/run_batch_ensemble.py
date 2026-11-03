"""Parallel batch driver for GalaxyDock2-HEME: docks a ligand set (mol2 dir)
against ONE named receptor conformation. Launched once per receptor (run
concurrently as separate processes) via env vars:
  RECEPTOR_NAME, RECEPTOR_PDB, LIGAND_DIR, HEME_RES_NUM, CHAIN,
  BOX_X, BOX_Y, BOX_Z, N_WORKERS

Generalized from the CYP3A4-only version used earlier this project (that one
only ever existed on the remote instance, never saved locally) to take the
ligand directory and box center as env vars instead of hardcoding them, so
the same script covers CYP1A2/CYP2C9/CYP2D6 too.
"""

import multiprocessing as mp
import os

# Must be set before any numpy/BLAS import happens anywhere in this process tree --
# otherwise each of the concurrent docking workers tries to grab all visible
# cores for its own linear-algebra calls, causing massive oversubscription.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import shutil
import subprocess as sp
import time
from pathlib import Path

PROJECT_DIR = Path("/root/project")
GD2_HOME = PROJECT_DIR / "GalaxyDock2_HEME"
CHIMERA_BIN = "/root/tools/UCSF-Chimera64/bin"
PYTHON = "/opt/conda/bin/python3"
SCRIPT = GD2_HOME / "script" / "run_GalaxyDock2_heme.py"

RECEPTOR_NAME = os.environ["RECEPTOR_NAME"]
RECEPTOR = PROJECT_DIR / os.environ["RECEPTOR_PDB"]
LIGAND_DIR = PROJECT_DIR / os.environ["LIGAND_DIR"]
HEME_RES_NUM = os.environ["HEME_RES_NUM"]
CHAIN = os.environ.get("CHAIN", "A")
N_WORKERS = int(os.environ.get("N_WORKERS", "16"))
BOX_CENTER = (float(os.environ["BOX_X"]), float(os.environ["BOX_Y"]), float(os.environ["BOX_Z"]))
TIMEOUT_S = 300

RUN_DIR = Path(f"/root/gd2heme_runs_{RECEPTOR_NAME}")


def dock_one(mol2_path: Path) -> tuple[str, bool, str]:
    name = mol2_path.stem
    out_dir = RUN_DIR / name
    if (out_dir / "GD2_HEME_fb.E.info").exists():
        return (name, True, "already done")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RECEPTOR, out_dir / "receptor.pdb")
    shutil.copy(mol2_path, out_dir / "ligand.mol2")

    env = os.environ.copy()
    env["PATH"] = f"{env.get('PATH', '')}:{CHIMERA_BIN}"

    cmd = [
        PYTHON, str(SCRIPT),
        "-p", "receptor.pdb", "-l", "ligand.mol2",
        "-d", str(GD2_HOME),
        "-x", str(BOX_CENTER[0]), "-y", str(BOX_CENTER[1]), "-z", str(BOX_CENTER[2]),
        "--heme_res_num", HEME_RES_NUM, "--chain", CHAIN,
    ]
    try:
        result = sp.run(
            cmd, cwd=str(out_dir), env=env,
            stdout=sp.PIPE, stderr=sp.STDOUT, timeout=TIMEOUT_S, text=True,
        )
        ok = (out_dir / "GD2_HEME_fb.E.info").exists()
        if not ok:
            (out_dir / "failure.log").write_text(result.stdout[-4000:])
        return (name, ok, "" if ok else "no output file")
    except sp.TimeoutExpired:
        (out_dir / "failure.log").write_text("TIMEOUT")
        return (name, False, "timeout")
    except Exception as e:
        (out_dir / "failure.log").write_text(str(e))
        return (name, False, str(e))


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ligands = sorted(LIGAND_DIR.glob("*.mol2"))
    total = len(ligands)
    print(f"[{RECEPTOR_NAME}] {total} ligands, {N_WORKERS} workers", flush=True)

    start = time.time()
    done = 0
    failed = 0
    with mp.Pool(N_WORKERS) as pool:
        for name, ok, msg in pool.imap_unordered(dock_one, ligands, chunksize=1):
            done += 1
            if not ok:
                failed += 1
                print(f"[{RECEPTOR_NAME}]  FAILED {name}: {msg}", flush=True)
            if done % 100 == 0 or done == total:
                elapsed = (time.time() - start) / 60.0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(
                    f"[{RECEPTOR_NAME}] [{done}/{total}] done={done - failed} failed={failed} "
                    f"elapsed={elapsed:.1f}min eta={eta:.1f}min",
                    flush=True,
                )

    print(f"[{RECEPTOR_NAME}] DONE: {done - failed}/{total} succeeded, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
