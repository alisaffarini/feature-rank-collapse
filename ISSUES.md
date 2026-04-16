# Feature Rank Collapse — Known Issues

This document catalogs verified discrepancies between the paper, code, and results. These must be resolved before submission.

## Critical Issues

### 1. Rank Definition Mismatch (Paper vs Code)

**Paper** defines numerical rank as threshold-based (Eq. 1):
```
rank_τ(A) = |{i : σ_i(A) > τ · σ_1(A)}|   with τ = 0.01
```
Then normalizes: r̃ = rank_τ / min(N, d)

**Code** (`experiment.py`, line 187) computes entropy-based effective rank:
```python
p = s / s.sum()
entropy = -np.sum(p * np.log(p + 1e-12))
return float(np.exp(entropy))
```
This is the Roy & Vetterli (2007) effective rank — a continuous measure based on Shannon entropy of the normalized singular value distribution. It is **not the same metric** as the threshold-based numerical rank described in the paper.

**Impact:** All reported rank values were computed using the entropy-based metric. The paper text, equations, and interpretation must be updated to describe what was actually computed, OR the experiments must be re-run using the threshold-based metric.

### 2. Unverifiable Results

**Results JSON** (`run_080_summary.json`) states: `"source": "extracted from RunPod session, full results.json (24KB) on pod"`. The full per-seed data from RunPod is not available — only a hand-written summary JSON exists.

**Local run data** (in `burn-tokens/research/runs/run_080_feature_collapse/output.log`) only completed:
- ResNet-18 seed 42: test_acc=0.9426, 4 OOD evaluations completed
- VGG-16-BN: crashed immediately after ResNet, never trained

**The local output.log numbers do NOT match the summary JSON:**
| Metric | Local (seed 42) | JSON (claimed 3-seed mean) |
|--------|----------------|---------------------------|
| ResNet test_acc | 0.9426 | 0.9449 |
| gaussian MSP AUROC | 0.9552 | 0.9741 |
| uniform MSP AUROC | 0.5658 | 0.8365 |
| layer4 collapse (uniform) | 0.715 | 0.6248 |

The differences could indicate the JSON was from a separate RunPod session with different random seeds or hyperparameters, but this cannot be verified since the RunPod data was not saved.

### 3. Per-seed Accuracy Variance

The JSON reports `test_acc_std: 0.0003` across 3 seeds for ResNet-18 (94.49% ± 0.03%). While not impossible, this is unusually low variance for CIFAR-10 ResNet-18 training (typical std is ~0.1-0.3%).

### 4. No Rank-Based OOD Detector Evaluated

The paper claims "complementarity" with MSP/Energy scores and discusses the rank collapse profile as a diagnostic tool, but **no rank-based OOD detector is actually implemented or evaluated.** The paper should either:
- Implement a simple rank-based detector and report AUROC
- Or explicitly scope the contribution as "analysis/diagnostic only" and remove claims of "complementarity"

### 5. Table 2 Incomplete

Paper Table 2 has "---" entries for two layers, suggesting missing data.

## Recommended Path Forward

1. **Re-run experiments** from scratch with both ResNet-18 and VGG-16-BN, 3+ seeds, saving full per-seed data
2. **Fix rank definition** — either update paper to describe entropy-based effective rank (what code computes) or update code to use threshold-based rank (what paper describes)
3. **Implement a rank-based OOD detector** or remove "complementarity" claims
4. **Save raw data** — per-seed accuracies, per-layer ranks for all OOD datasets

## What IS Verified

From the local output.log, seed 42 ResNet-18 produced:
- Training completed (50 epochs, test_acc 0.9426)
- Layer-wise rank collapse pattern IS real: deeper layers show more collapse
  - uniform OOD: layer1=-0.059, layer2=0.013, layer3=0.191, layer4=0.715
  - gaussian OOD: layer1=-0.020, layer2=0.117, layer3=0.274, layer4=0.660
- The core thesis (deeper layers collapse more on OOD) appears genuine based on seed 42
- OOD detection scores: gaussian AUROC high (0.95+), uniform AUROC low (0.57) — expected pattern
