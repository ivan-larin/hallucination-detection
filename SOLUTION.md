*I will make this repository public on May 13. Currently, all of my GitHub accounts are restricted from using private repositories due to U.S. sanctions related to Crimea.*

# Solution

## Reproducibility

The default pipeline is unchanged — just run `solution.py`:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

This produces `results.json` and `predictions.csv`. 

`test_accuracy = 75.76%`, `test_auroc = 74.07%`, `val_accuracy = 76.92%`, `val_auroc = 75.60%`.

## Results

Final solution:


### Aggregation

For each sample, take hidden states at layers 12-16 and concatenate four pooled vectors per layer: the last token and the mean over the last $k$ real tokens ($k \in \{ 3, 8, 16\}$).

$4 \cdot 5 \cdot 896 = 17920$ features in total. 

Several papers suggest mid-top layers carry the strongest hallucination signal, and this is confirmed in this task. Tail means stabilise the high-variance of last token.

### Splitting

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with a stratified validation split equal to 15% of the full dataset carved from each fold's non-test portion (so 65% train / 15% val / 20% test per fold).

### Probe

For each $k$ independently apply scaling+PCA (24 components) for vector $(\text{last}, \text{tail-k})$ and concatenate the results. Per-view PCA — rather than one global PCA on the 17920D vector — gives each tail window an equal voice in the 72D final representation.


Then apply `LogisticRegression(C=0.05)` (`L2` regularization helps prevent overfitting on such a small dataset). During cross-validation, `evaluate.py` additionally tunes the threshold on the fold validation split. For the final `predictions.csv` model, `solution.py` only calls `fit()`, so `fit()` performs an internal 80/20 split to tune the threshold before refitting logistic regression on all labelled rows.

## Failed attempts

- **Different poolings** mean-all-tokens / max-all-tokens. Doing something with all tokens probably didn't work because of prompt tokens. Also changing pooling type for tail decreased the metrics.
- **More layers** (12-24 (or other intervals) instead of 12-16). Layers 17-24 did not improve CV & test metrics in my experiments.
- **MLP probes & gradient boostings** It was overfitting on train. 689 samples is too small for them.
- **L1 instead of L2 in the LR** Sparsifies but loses ~1 pp test_acc.
- **Feature interactions** (polynomial features and other combinations) LR already captures what's separable at this rank.
- **Geometric features** (per-layer norms, inter-layer cosine drift). Even when fed through the same scaling+PCA pipeline and concatenated with the hidden-state blocks, the LR's coefficients on them ended up near zero — they don't add an orthogonal signal.
- **Threshold tuning for accuracy** instead of F1. Slight gains on a few configs but mostly overfits the inner-val sample.
- **Alternative scalers** MinMaxScaler dropped ~2 pp test_acc
- **Skipping threshold tuning** (fit LR on the full input, keep threshold at the default 0.5). Significant drop in metrics — F1-tuning the threshold is worth ~5-6 pp test_acc.

Metrics are omitted here — these experiments span different stages and configurations, so head-to-head comparison would be misleading.

Also due to task limitations (I can modify only 3 files) I can't try response-only features, additional data, features base on final predicted token probabilities.

