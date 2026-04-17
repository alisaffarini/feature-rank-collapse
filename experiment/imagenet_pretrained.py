#!/usr/bin/env python3
"""ImageNet-scale Feature Rank Collapse using pretrained models.

Uses pretrained models to measure effective rank of activations on:
- In-distribution: ImageNet validation
- OOD: Textures (DTD), Gaussian noise, Uniform noise

No training needed — just forward pass and SVD.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_SAMPLES = 500
BATCH_SIZE = 64
N_SEEDS = 3

# ─── Feature Extraction ───

class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.features = {}
        self.hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._hook_fn(name, out))
                self.hooks.append(hook)

    def _hook_fn(self, name, output):
        if isinstance(output, tuple):
            output = output[0]
        # Global avg pool spatial dims to avoid OOM on SVD
        if output.dim() == 4:
            output = output.mean([2, 3])
        self.features[name] = output.detach()

    def remove(self):
        for h in self.hooks:
            h.remove()

# ─── Effective Rank ───

def effective_rank(activations):
    """Compute effective rank via entropy of normalized singular values."""
    X = activations
    if X.dim() == 4:
        X = X.mean([2, 3])  # global avg pool
    elif X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])

    X = X.float().cpu()
    X = X - X.mean(0, keepdim=True)

    try:
        s = torch.linalg.svdvals(X)
        s = s[s > 1e-10]
        p = s / s.sum()
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy).item()
    except:
        return float('nan')

# ─── Main ───

def main():
    import timm

    print("Loading pretrained models...")
    resnet = timm.create_model('resnet50', pretrained=True).to(device).eval()
    vgg = timm.create_model('vgg19_bn', pretrained=True).to(device).eval()

    resnet_layers = ['layer2', 'layer3', 'layer4']
    vgg_layers = ['features.26', 'features.39', 'features.52']

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading ImageNet-V2 validation set...")
    val_dataset = torchvision.datasets.ImageFolder('/workspace/imagenet_v2/imagenetv2-matched-frequency-format-val', transform=transform)

    print(f"Validation set size: {len(val_dataset)}")

    # OOD datasets
    def make_gaussian_dataset(n, shape=(3, 224, 224)):
        data = torch.randn(n, *shape)
        data = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data)
        return TensorDataset(data, torch.zeros(n, dtype=torch.long))

    def make_uniform_dataset(n, shape=(3, 224, 224)):
        data = torch.rand(n, *shape)
        data = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data)
        return TensorDataset(data, torch.zeros(n, dtype=torch.long))

    # Try loading Textures (DTD)
    try:
        dtd = torchvision.datasets.DTD(root='/workspace/data', split='test', download=True, transform=transform)
        has_dtd = True
        print(f"DTD (Textures) loaded: {len(dtd)} images")
    except:
        has_dtd = False
        print("DTD not available, skipping")

    all_results = []

    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ID subset
        indices = np.random.permutation(len(val_dataset))[:N_SAMPLES]
        id_loader = DataLoader(Subset(val_dataset, indices), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

        # OOD loaders
        gaussian_loader = DataLoader(make_gaussian_dataset(N_SAMPLES), batch_size=BATCH_SIZE)
        uniform_loader = DataLoader(make_uniform_dataset(N_SAMPLES), batch_size=BATCH_SIZE)

        ood_loaders = {'gaussian': gaussian_loader, 'uniform': uniform_loader}
        if has_dtd:
            dtd_indices = np.random.permutation(len(dtd))[:min(N_SAMPLES, len(dtd))]
            ood_loaders['textures'] = DataLoader(Subset(dtd, dtd_indices), batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=4)

        seed_result = {'seed': seed, 'architectures': {}}

        for arch_name, model, layers in [('resnet50', resnet, resnet_layers),
                                          ('vgg19_bn', vgg, vgg_layers)]:
            print(f"\n  --- {arch_name} ---")
            extractor = FeatureExtractor(model, layers)

            # ID ranks
            print("  Computing ID ranks...")
            id_feats = {name: [] for name in layers}
            with torch.no_grad():
                for x, _ in id_loader:
                    x = x.to(device)
                    _ = model(x)
                    for name in layers:
                        if name in extractor.features:
                            id_feats[name].append(extractor.features[name].cpu())

            id_ranks = {}
            for name in layers:
                if id_feats[name]:
                    feat = torch.cat(id_feats[name], dim=0)
                    id_ranks[name] = effective_rank(feat)
                    print(f"    {name}: rank={id_ranks[name]:.1f}")

            # OOD ranks
            arch_ood = {}
            for ood_name, ood_loader in ood_loaders.items():
                print(f"  OOD={ood_name}...")
                ood_feats = {name: [] for name in layers}
                with torch.no_grad():
                    for x, _ in ood_loader:
                        x = x.to(device)
                        _ = model(x)
                        for name in layers:
                            if name in extractor.features:
                                ood_feats[name].append(extractor.features[name].cpu())

                ood_ranks = {}
                collapse = {}
                for name in layers:
                    if ood_feats[name] and name in id_ranks:
                        feat = torch.cat(ood_feats[name], dim=0)
                        ood_r = effective_rank(feat)
                        ood_ranks[name] = ood_r
                        collapse[name] = 1.0 - (ood_r / id_ranks[name]) if id_ranks[name] > 0 else 0.0
                        print(f"    {name}: rank={ood_r:.1f}, collapse={collapse[name]:.4f}")

                arch_ood[ood_name] = {'ranks': ood_ranks, 'collapse': collapse}

            extractor.remove()
            seed_result['architectures'][arch_name] = {
                'id_ranks': id_ranks,
                'ood': arch_ood,
            }

        all_results.append(seed_result)

    # Aggregate
    print(f"\n{'='*50}\nAGGREGATED ({N_SEEDS} seeds)\n{'='*50}")

    summary = {'n_seeds': N_SEEDS, 'n_samples': N_SAMPLES, 'architectures': {}}

    for arch_name in all_results[0]['architectures']:
        arch_summary = {'ood': {}}
        for ood_name in all_results[0]['architectures'][arch_name]['ood']:
            layers = list(all_results[0]['architectures'][arch_name]['ood'][ood_name]['collapse'].keys())
            layer_collapse = {}
            for layer in layers:
                vals = [r['architectures'][arch_name]['ood'][ood_name]['collapse'][layer]
                        for r in all_results]
                layer_collapse[layer] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1))}
                print(f"  {arch_name} / {ood_name} / {layer}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
            arch_summary['ood'][ood_name] = layer_collapse
        summary['architectures'][arch_name] = arch_summary

    out_path = 'imagenet_pretrained_results.json'
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'per_seed': all_results}, f, indent=2, default=str)
    print(f"\nRESULTS: {json.dumps(summary, default=str)}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
