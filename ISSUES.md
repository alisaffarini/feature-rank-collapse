# Feature Rank Collapse -- Known Issues

This document catalogs verified discrepancies between the paper, code, and results. These must be resolved before submission.

## Critical Issues

### 1. Rank Definition Mismatch (Paper vs Code) -- FIXED in paper

**Status: RESOLVED.** The paper's Method section (Eq. 1) now correctly describes the entropy-based effective rank (Roy & Vetterli 2007) that the code actually computes. The prior draft had described a threshold-based numerical rank that was never implemented.

### 2. Paper Numbers Were Unverifiable / Fabricated -- RESOLVED

**Status: RESOLVED. Full 3-seed A100 re-run completed; paper updated with real data.**

The prior draft reported fabricated 3-seed results. A full re-run was completed on NVIDIA A100 GPU with 3 seeds (42, 43, 44) for both ResNet-18 and VGG-16-BN. Results are saved in `results/results_a100.json`. The paper has been updated with all real numbers from this JSON (3-seed mean +/- std for all metrics, per-seed appendix tables, etc.). All "preliminary" and "single-seed" caveats have been removed.

### 3. Per-seed Accuracy Variance Was Implausible -- RESOLVED (real data now available)

The prior paper claimed test_acc_std=0.0003 across 3 seeds (fabricated). Real 3-seed data now shows ResNet-18 test_acc_std=0.0012 and VGG-16-BN test_acc_std=0.0015, which are realistic variance levels.

### 4. No Rank-Based OOD Detector Evaluated -- STILL OPEN

The paper discusses rank collapse as a "complementary" signal to MSP/Energy but never implements a rank-based OOD detector. The paper should either:
- Implement a simple rank-based detector and report AUROC
- Or explicitly scope the contribution as "analysis/diagnostic only" and remove "complementarity" claims

### 5. Table 2 Incomplete -- RESOLVED

Table 2 now contains full 3-seed mean +/- std data for all layers of both ResNet-18 and VGG-16-BN, across all 4 OOD datasets. VGG-16-BN data is in a separate Table 3.

### 6. VGG-16-BN Data Missing -- RESOLVED

VGG-16-BN 3-seed data is now available from the A100 re-run. The architecture comparison section in the paper has been fully written with real data. VGG-16-BN shows stronger deep-layer collapse than ResNet-18 (pool4 RC=0.800 vs layer4 RC=0.625 on uniform noise), supporting the residual-connections-mitigate-collapse hypothesis.

### 7. Provenance `results_wideresnet.json` is Misplaced -- NEW

The file `provenance/results_wideresnet.json` contains WideResNet-16-8 dropout/linear-probe experiment data (same_class/wrong_class/random_class perturbations). This appears to be from the bn-calibration repo, not feature-rank-collapse. It should be moved to the correct repo or deleted.

### 8. Provenance `run_081_gradient_starvation/` is Unrelated -- NEW

The `run_081_gradient_starvation/` directory contains a completely different experiment about per-class gradient norms and class-balanced training. It is NOT related to feature rank collapse. This data belongs in a different repo. The logs show it ran on CUDA (RunPod) and MPS (laptop) but the experiments are incomplete (laptop MPS runs have heavy noise from MallocStackLogging and appear truncated).

### 9. experiment.py Provenance vs Current -- VERIFIED MATCH

The provenance copy `provenance/run_080_feature_collapse/experiment.py` is byte-identical to `experiment/experiment.py`. No code drift.

## Recommended Path Forward

1. ~~**Full GPU re-run is MANDATORY.**~~ **DONE.** A100 re-run completed with 3 seeds for both architectures. Results in `results/results_a100.json`.
2. **Implement a rank-based OOD detector** or remove "complementarity" claims from the paper.
3. **Clean up provenance:** Remove or relocate `results_wideresnet.json` and `run_081_gradient_starvation/` to their correct repos.
4. ~~**Save raw data**~~ **DONE.** Full per-seed JSON is saved in `results/results_a100.json`.

## What IS Verified

From the A100 re-run (`results/results_a100.json`), all 6 training runs (2 architectures x 3 seeds) completed successfully in ~111 minutes:

**ResNet-18 (3-seed mean):**
- test_acc: 94.34% +/- 0.12%
- Layer-wise rank collapse pattern confirmed across all 3 seeds:
  - uniform OOD: layer1=-0.053, layer2=0.010, layer3=0.213, layer4=0.625
  - gaussian OOD: layer1=-0.027, layer2=0.098, layer3=0.295, layer4=0.595
  - svhn OOD: layer1=0.262, layer2=0.183, layer3=0.210, layer4=0.360
  - cifar100 OOD: layer1=0.007, layer2=0.003, layer3=-0.016, layer4=-0.063

**VGG-16-BN (3-seed mean):**
- test_acc: 92.92% +/- 0.15%
- Shows even stronger deep-layer collapse than ResNet-18:
  - uniform OOD: pool0=-0.059, pool1=0.042, pool2=0.235, pool3=0.390, pool4=0.800
  - gaussian OOD: pool0=-0.058, pool1=0.114, pool2=0.313, pool3=0.366, pool4=0.784

## Honest Assessment

The paper is now on solid empirical footing with 3-seed A100 data for both architectures. The core phenomenon (monotonically increasing rank collapse with depth on far-OOD data) is confirmed across seeds and architectures. The remaining open issues are: (1) implementing a rank-based OOD detector or scoping the contribution as analysis-only, and (2) cleaning up unrelated provenance files.
