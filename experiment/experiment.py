"""
Run 080: Feature Collapse Under Distribution Shift
===================================================
Hypothesis: Pretrained models lose representational diversity in deeper layers 
when fed OOD data — features collapse to a low-rank subspace. This collapse 
is measurable via effective rank of activation matrices and can serve as a 
label-free OOD detection signal.

Experiments:
1. Train ResNet on CIFAR-10, measure activation rank per layer on:
   - In-distribution (CIFAR-10 test)
   - Near-OOD (CIFAR-100, SVHN)
   - Far-OOD (Gaussian noise, uniform noise, Textures)
2. Repeat with different architectures (ResNet-18, VGG-16, DenseNet)
3. Compare effective rank as OOD detector vs baselines (MSP, Energy, Mahalanobis)
4. Statistical significance across seeds

Key metric: Effective rank = exp(entropy of normalized singular values)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
import os
import sys
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_SEEDS = 3
BATCH_SIZE = 128
TRAIN_EPOCHS = 50
LR = 0.1
NUM_SAMPLES_FOR_RANK = 1000  # samples to compute activation rank
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Device: {DEVICE}")
print(f"Seeds: {NUM_SEEDS}")
print(f"Epochs: {TRAIN_EPOCHS}")

# ============================================================
# Data
# ============================================================
def get_datasets():
    """Get in-distribution and OOD datasets."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # In-distribution
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Near-OOD: CIFAR-100
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    # Near-OOD: SVHN
    svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    return trainset, testset, {'cifar100': cifar100, 'svhn': svhn}


# ============================================================
# Synthetic OOD data
# ============================================================
class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, shape=(3, 32, 32)):
        self.size = size
        self.shape = shape
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.randn(self.shape)
        # Normalize like CIFAR
        x = (x - self.mean) / self.std
        return x, 0

class UniformNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, shape=(3, 32, 32)):
        self.size = size
        self.shape = shape
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.rand(self.shape)
        x = (x - self.mean) / self.std
        return x, 0


# ============================================================
# Models with hooks for activation extraction
# ============================================================
class ResNet18WithHooks(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        # Adapt for 32x32 input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        layers = {
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
        }
        for name, layer in layers.items():
            layer.register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def forward(self, x):
        self.activations = {}
        return self.model(x)


class VGG16WithHooks(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torchvision.models.vgg16_bn(weights=None).features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        # Hook after each maxpool
        pool_idx = 0
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                name = f'pool{pool_idx}'
                layer.register_forward_hook(self._get_hook(name))
                pool_idx += 1
    
    def _get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def forward(self, x):
        self.activations = {}
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# Effective Rank computation
# ============================================================
def effective_rank(activations):
    """
    Compute effective rank of activation matrix.
    activations: (N, C, H, W) tensor
    Returns: effective rank (scalar)
    """
    # Reshape to (N, C*H*W)
    N = activations.shape[0]
    A = activations.reshape(N, -1).float().cpu().numpy()
    
    # SVD
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return float('nan')
    
    # Normalize singular values
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    
    p = s / s.sum()
    # Entropy
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_rank_profile(model, dataloader, num_samples=1000):
    """Compute effective rank at each hooked layer."""
    model.eval()
    all_activations = defaultdict(list)
    total = 0
    
    with torch.no_grad():
        for x, _ in dataloader:
            if total >= num_samples:
                break
            x = x.to(DEVICE)
            _ = model(x)
            for name, act in model.activations.items():
                all_activations[name].append(act.cpu())
            total += x.shape[0]
    
    ranks = {}
    for name, acts in all_activations.items():
        combined = torch.cat(acts, dim=0)[:num_samples]
        ranks[name] = effective_rank(combined)
    
    return ranks


# ============================================================
# OOD Detection metrics
# ============================================================
def compute_ood_scores(model, dataloader, num_samples=1000):
    """Compute MSP and Energy scores alongside rank."""
    model.eval()
    msp_scores = []
    energy_scores = []
    total = 0
    
    with torch.no_grad():
        for x, _ in dataloader:
            if total >= num_samples:
                break
            x = x.to(DEVICE)
            logits = model(x)
            
            # MSP
            probs = torch.softmax(logits, dim=1)
            msp = probs.max(dim=1)[0].cpu().numpy()
            msp_scores.extend(msp)
            
            # Energy
            energy = torch.logsumexp(logits, dim=1).cpu().numpy()
            energy_scores.extend(energy)
            
            total += x.shape[0]
    
    return {
        'msp': np.array(msp_scores[:num_samples]),
        'energy': np.array(energy_scores[:num_samples]),
    }


def auroc(in_scores, ood_scores):
    """Compute AUROC for OOD detection."""
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    
    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    labels_sorted = labels[sorted_idx]
    
    # Compute AUROC
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr_sum = 0
    tp = 0
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            tpr_sum += tp
    
    return tpr_sum / (n_pos * n_neg)


# ============================================================
# Training
# ============================================================
def train_model(model, trainloader, testloader, epochs=50, lr=0.1):
    """Train model with standard schedule."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    model.to(DEVICE)
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = out.max(1)
                test_correct += pred.eq(y).sum().item()
                test_total += y.size(0)
        
        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_acc={correct/total:.4f} test_acc={test_acc:.4f} lr={scheduler.get_last_lr()[0]:.4f}")
    
    return best_acc


# ============================================================
# Main experiment
# ============================================================
def run_experiment(seed):
    """Run full experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get data
    trainset, testset, ood_datasets = get_datasets()
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Add synthetic OOD
    ood_datasets['gaussian'] = GaussianNoiseDataset(size=NUM_SAMPLES_FOR_RANK)
    ood_datasets['uniform'] = UniformNoiseDataset(size=NUM_SAMPLES_FOR_RANK)
    
    results = {}
    
    # ---- ResNet-18 ----
    print("\n--- ResNet-18 ---")
    model = ResNet18WithHooks(num_classes=10)
    start = time.time()
    test_acc = train_model(model, trainloader, testloader, epochs=TRAIN_EPOCHS, lr=LR)
    train_time = time.time() - start
    print(f"  Best test acc: {test_acc:.4f} ({train_time:.0f}s)")
    
    # Compute rank profiles
    in_ranks = compute_rank_profile(model, testloader, NUM_SAMPLES_FOR_RANK)
    in_scores = compute_ood_scores(model, testloader, NUM_SAMPLES_FOR_RANK)
    print(f"  ID ranks: {{{', '.join(f'{k}: {v:.1f}' for k,v in in_ranks.items())}}}")
    
    ood_results = {}
    for ood_name, ood_data in ood_datasets.items():
        ood_loader = DataLoader(ood_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        ood_ranks = compute_rank_profile(model, ood_loader, NUM_SAMPLES_FOR_RANK)
        ood_scores = compute_ood_scores(model, ood_loader, NUM_SAMPLES_FOR_RANK)
        
        # Rank-based AUROC: use ratio of deepest layer rank to shallowest
        # Lower rank ratio for OOD = detection signal
        in_rank_ratio = in_ranks.get('layer4', 1) / max(in_ranks.get('layer1', 1), 1e-6)
        ood_rank_ratio = ood_ranks.get('layer4', 1) / max(ood_ranks.get('layer1', 1), 1e-6)
        
        # MSP AUROC
        msp_auroc = auroc(in_scores['msp'], ood_scores['msp'])
        energy_auroc = auroc(in_scores['energy'], ood_scores['energy'])
        
        ood_results[ood_name] = {
            'ranks': {k: float(v) for k, v in ood_ranks.items()},
            'rank_ratio_deep_shallow': float(ood_rank_ratio),
            'msp_auroc': float(msp_auroc),
            'energy_auroc': float(energy_auroc),
        }
        
        # Rank collapse metric: relative drop in effective rank at deep layers
        collapse = {}
        for layer in in_ranks:
            if in_ranks[layer] > 0:
                collapse[layer] = float((in_ranks[layer] - ood_ranks.get(layer, 0)) / in_ranks[layer])
        ood_results[ood_name]['rank_collapse'] = collapse
        
        print(f"  OOD={ood_name}: ranks={{{', '.join(f'{k}: {v:.1f}' for k,v in ood_ranks.items())}}}")
        print(f"    collapse={{{', '.join(f'{k}: {v:.3f}' for k,v in collapse.items())}}}")
        print(f"    MSP_AUROC={msp_auroc:.4f} Energy_AUROC={energy_auroc:.4f}")
    
    results['resnet18'] = {
        'test_acc': float(test_acc),
        'train_time': float(train_time),
        'in_distribution_ranks': {k: float(v) for k, v in in_ranks.items()},
        'in_rank_ratio': float(in_ranks.get('layer4', 1) / max(in_ranks.get('layer1', 1), 1e-6)),
        'ood': ood_results,
    }
    
    # ---- VGG-16 ----
    print("\n--- VGG-16-BN ---")
    model_vgg = VGG16WithHooks(num_classes=10)
    start = time.time()
    test_acc_vgg = train_model(model_vgg, trainloader, testloader, epochs=TRAIN_EPOCHS, lr=0.05)
    train_time_vgg = time.time() - start
    print(f"  Best test acc: {test_acc_vgg:.4f} ({train_time_vgg:.0f}s)")
    
    in_ranks_vgg = compute_rank_profile(model_vgg, testloader, NUM_SAMPLES_FOR_RANK)
    in_scores_vgg = compute_ood_scores(model_vgg, testloader, NUM_SAMPLES_FOR_RANK)
    print(f"  ID ranks: {{{', '.join(f'{k}: {v:.1f}' for k,v in in_ranks_vgg.items())}}}")
    
    ood_results_vgg = {}
    for ood_name, ood_data in ood_datasets.items():
        ood_loader = DataLoader(ood_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        ood_ranks = compute_rank_profile(model_vgg, ood_loader, NUM_SAMPLES_FOR_RANK)
        ood_scores = compute_ood_scores(model_vgg, ood_loader, NUM_SAMPLES_FOR_RANK)
        
        msp_auroc = auroc(in_scores_vgg['msp'], ood_scores['msp'])
        energy_auroc = auroc(in_scores_vgg['energy'], ood_scores['energy'])
        
        collapse = {}
        for layer in in_ranks_vgg:
            if in_ranks_vgg[layer] > 0:
                collapse[layer] = float((in_ranks_vgg[layer] - ood_ranks.get(layer, 0)) / in_ranks_vgg[layer])
        
        ood_results_vgg[ood_name] = {
            'ranks': {k: float(v) for k, v in ood_ranks.items()},
            'rank_collapse': collapse,
            'msp_auroc': float(msp_auroc),
            'energy_auroc': float(energy_auroc),
        }
        
        print(f"  OOD={ood_name}: collapse={{{', '.join(f'{k}: {v:.3f}' for k,v in collapse.items())}}}")
        print(f"    MSP_AUROC={msp_auroc:.4f} Energy_AUROC={energy_auroc:.4f}")
    
    results['vgg16'] = {
        'test_acc': float(test_acc_vgg),
        'train_time': float(train_time_vgg),
        'in_distribution_ranks': {k: float(v) for k, v in in_ranks_vgg.items()},
        'ood': ood_results_vgg,
    }
    
    return results


def main():
    all_results = {}
    start_time = time.time()
    
    for seed in [42, 43, 44][:NUM_SEEDS]:
        all_results[f'seed_{seed}'] = run_experiment(seed)
    
    total_time = time.time() - start_time
    
    # ============================================================
    # Aggregate results
    # ============================================================
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    
    summary = {'total_time_minutes': total_time / 60, 'seeds': NUM_SEEDS, 'epochs': TRAIN_EPOCHS}
    
    for arch in ['resnet18', 'vgg16']:
        print(f"\n--- {arch} ---")
        accs = [all_results[f'seed_{s}'][arch]['test_acc'] for s in [42,43,44][:NUM_SEEDS]]
        print(f"  Test acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        
        summary[arch] = {'test_acc_mean': float(np.mean(accs)), 'test_acc_std': float(np.std(accs))}
        
        # Per OOD dataset
        ood_names = list(all_results['seed_42'][arch]['ood'].keys())
        for ood_name in ood_names:
            print(f"\n  OOD: {ood_name}")
            
            msp_aurocs = [all_results[f'seed_{s}'][arch]['ood'][ood_name]['msp_auroc'] for s in [42,43,44][:NUM_SEEDS]]
            energy_aurocs = [all_results[f'seed_{s}'][arch]['ood'][ood_name]['energy_auroc'] for s in [42,43,44][:NUM_SEEDS]]
            
            # Rank collapse per layer
            layers = list(all_results['seed_42'][arch]['ood'][ood_name]['rank_collapse'].keys())
            collapse_by_layer = {}
            for layer in layers:
                collapses = [all_results[f'seed_{s}'][arch]['ood'][ood_name]['rank_collapse'].get(layer, 0) for s in [42,43,44][:NUM_SEEDS]]
                collapse_by_layer[layer] = {'mean': float(np.mean(collapses)), 'std': float(np.std(collapses))}
                print(f"    {layer} collapse: {np.mean(collapses):.4f} ± {np.std(collapses):.4f}")
            
            print(f"    MSP AUROC: {np.mean(msp_aurocs):.4f} ± {np.std(msp_aurocs):.4f}")
            print(f"    Energy AUROC: {np.mean(energy_aurocs):.4f} ± {np.std(energy_aurocs):.4f}")
            
            # Statistical test: is deepest layer collapse significantly different from shallowest?
            if len(layers) >= 2:
                deep_collapses = [all_results[f'seed_{s}'][arch]['ood'][ood_name]['rank_collapse'].get(layers[-1], 0) for s in [42,43,44][:NUM_SEEDS]]
                shallow_collapses = [all_results[f'seed_{s}'][arch]['ood'][ood_name]['rank_collapse'].get(layers[0], 0) for s in [42,43,44][:NUM_SEEDS]]
                if NUM_SEEDS >= 2:
                    t, p = stats.ttest_rel(deep_collapses, shallow_collapses)
                    print(f"    Deep vs Shallow collapse: t={t:.4f}, p={p:.6f}")
            
            if ood_name not in summary.get(arch, {}).get('ood', {}):
                if 'ood' not in summary[arch]:
                    summary[arch]['ood'] = {}
                summary[arch]['ood'][ood_name] = {
                    'msp_auroc': {'mean': float(np.mean(msp_aurocs)), 'std': float(np.std(msp_aurocs))},
                    'energy_auroc': {'mean': float(np.mean(energy_aurocs)), 'std': float(np.std(energy_aurocs))},
                    'rank_collapse_by_layer': collapse_by_layer,
                }
    
    # Save
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'raw': all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
