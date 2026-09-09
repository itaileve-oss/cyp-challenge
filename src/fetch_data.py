import pandas as pd, pathlib

base = "hf://datasets/openadmet/cyp-challenge-train-test/"
files = {
    "TRAIN_inhibition": "cyp-challenge-TRAIN_inhibition.csv",
    "TEST_BLINDED": "cyp-challenge-TEST-BLINDED.csv",
    "TRAIN_TDI": "cyp-challenge-TRAIN_TDI.csv",
    "single_conc_TRAIN": "cyp-challenge-single-concentration-TRAIN.csv",
    "TRAIN_Emax": "cyp-challenge-TRAIN_Emax.csv",
}
out = pathlib.Path("/home/ileven/Projects/cyp-challenge/data/raw")
out.mkdir(parents=True, exist_ok=True)
for k, v in files.items():
    try:
        df = pd.read_csv(base + v)
        df.to_csv(out / f"{k}.csv", index=False)
        print(f"{k}: {df.shape}")
        print("  cols:", list(df.columns))
    except Exception as e:
        print(f"{k}: FAILED {type(e).__name__}: {e}")
