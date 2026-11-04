# TabPFN as a blend component

A foundation model for tabular data, added to two isoforms. Documented partly as
a worked example of adopting a model that is **worse than what you already have**.

## The result

TabPFN alone underperforms the ensemble at every endpoint. It was adopted anyway,
at two endpoints, because it decorrelates: out-of-fold correlation with the
LightGBM member is only **0.58** for CYP2D6. Its value is independence, not
accuracy.

| isoform | weight | ΔST-RAE (fixed weight) | ΔST-RAE (nested) |
|---|---|---|---|
| CYP2C9 | 0.30 | **−0.0107 ± 0.0010** | −0.0095 |
| CYP2D6 | 0.40 | **−0.0129 ± 0.0013** | −0.0113 |

CYP2C9 0.6344 → 0.6230; CYP2D6 0.8802 → 0.8680. All six metrics improve at both.

## Why we believe it

```bash
python src/tabpfn_confirm_and_predict.py
```

- **Three shuffled cluster-CV repeats** with the ensemble's out-of-fold
  predictions held fixed, so only the blend varies.
- **Nested weight selection** — the weight is re-chosen inside each inner fold,
  so the reported delta carries no selection optimism. It is smaller than the
  fixed-weight delta, as expected, and still clearly negative.
- The gain is **7–10× the repeat standard deviation**, which is the bar a
  0.01-level improvement has to clear to be real.

## Not adopted

- **CYP1A2**: ST-RAE improved but R² and Pearson degraded. Reporting all six
  metrics is what caught this; ST-RAE alone would have accepted it.
- **CYP3A4 / CYP1A2 confirmation** was still running when the project paused —
  24 of 25 CV folds outstanding at ~2.5–3 h. Blending the largest endpoint on one
  fold of evidence was not worth the risk.
- **External-data stacking**: deltas of −0.002 to −0.005, inside the noise.

## Sanity check worth copying

Test predictions were checked against the LightGBM test predictions for
correlation consistency: 0.51 on test against 0.58 out-of-fold for CYP2D6. The
gap is explained by composition — the test set has far fewer basic amines than
training — rather than by a pipeline bug. Verifying that the *direction* of such
a discrepancy has a chemical explanation is cheaper than debugging it later.
