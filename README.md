# Feature Collapse in Deep Networks: Layer-Wise Rank Analysis for OOD Detection

**Author:** Ali Saffarini, Harvard University

## Key Results

- Deep layers show significantly more rank collapse on OOD data than shallow layers (t=20.37, p=0.002)
- ResNet-18: layer4 rank collapse 0.62 on uniform OOD, layer1 only 0.007
- Energy AUROC: 0.97 (Gaussian), 0.92 (SVHN), 0.86 (CIFAR-100), 0.73 (Uniform)
- VGG-16 shows same pattern with different magnitudes
- 3 seeds, 50 epochs, 4 OOD datasets

## Structure

```
paper/       (Paper needs to be written from results)
results/     Summary JSON from experiment
experiment/  Python experiment code
```

## Status

Experiment complete. Paper not yet written.

## Citation

Target venue: ICLR 2027
