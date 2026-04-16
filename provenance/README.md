# Provenance — Feature Rank Collapse

Raw run data. **See ISSUES.md in the repo root for known data integrity problems.**

## Run History

| Run | Description | Key Output |
|-----|-------------|------------|
| `run_080_feature_collapse` | Main experiment (partial) | experiment.py, output.log (seed 42 ResNet only) |
| `run_081_gradient_starvation` | Related gradient starvation experiment | Multiple output logs from laptop runs |
| `results_wideresnet.json` | WideResNet data from gpu_experiments | From burn-tokens GPU experiment directory |

## Critical Notes

- **Run 080 only completed ResNet-18 seed 42.** VGG-16-BN crashed at startup (MPS memory). See output.log.
- The `run_080_summary.json` in the main results/ directory claims "extracted from RunPod session" with 3-seed means for both architectures. The RunPod data was not saved locally.
- The local output.log numbers do NOT match the summary JSON — see ISSUES.md for details.
- Run 081 (gradient starvation) ran on Ali's laptop with multiple backup attempts.
