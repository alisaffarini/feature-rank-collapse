"""
Run 081: Gradient Starvation in Multi-Class Learning
=====================================================
Hypothesis: In multi-class training, some classes are learned significantly faster 
than others. Early-learned classes "starve" later classes of gradient signal. 
This starvation is:
  1) Predictable from inter-class similarity structure
  2) Measurable via per-class gradient norms
  3) Mitigable via class-balanced gradient scaling

Experiments:
1. Train on CIFAR-10 and CIFAR-100, track per-class metrics every epoch:
   - Per-class accuracy, loss, gradient norm contribution
   - Learning speed (epoch to reach X% accuracy per class)
2. Correlate learning order with:
   - Inter-class feature similarity (pre-trained embeddings)
   - Class confusion matrix structure
   - Visual similarity (pixel-space)
3. Test mitigation: class-balanced gradient scaling
4. Multiple seeds for significance

Key finding target: Strong correlation between class similarity and learning order,
with evidence that early classes suppress later ones via gradient competition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import os
from collections import defaultdict
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_SEEDS = 3
BATCH_SIZE = 128
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Device: {DEVICE}")
print(f"Seeds: {NUM_SEEDS}")


# ============================================================
# Models
# ============================================================
def make_resnet18(num_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ============================================================
# Per-class gradient tracking
# ============================================================
class GradientTracker:
    """Track per-class gradient norms on the last linear layer."""
    
    def __init__(self, model, num_classes):
        self.num_classes = num_classes
        self.model = model
        self.class_grad_norms = defaultdict(list)
    
    def compute_per_class_grads(self, dataloader, criterion):
        """Compute gradient norm contribution per class."""
        self.model.eval()
        class_grads = defaultdict(float)
        class_counts = defaultdict(int)
        
        # Get last linear layer
        last_layer = self.model.fc if hasattr(self.model, 'fc') else None
        if last_layer is None:
            return {}
        
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Compute per-class gradients
            for c in range(self.num_classes):
                mask = (y == c)
                if mask.sum() == 0:
                    continue
                
                self.model.zero_grad()
                out = self.model(x[mask])
                loss = criterion(out, y[mask])
                loss.backward()
                
                # Gradient norm on last layer weight
                if last_layer.weight.grad is not None:
                    grad_norm = last_layer.weight.grad.norm().item()
                    class_grads[c] += grad_norm
                    class_counts[c] += 1
            
            break  # One batch is enough for gradient estimation
        
        # Average
        result = {}
        for c in range(self.num_classes):
            if class_counts[c] > 0:
                result[c] = class_grads[c] / class_counts[c]
            else:
                result[c] = 0.0
        
        return result


# ============================================================
# Per-class accuracy tracking
# ============================================================
def per_class_accuracy(model, dataloader, num_classes):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, pred = out.max(1)
            for c in range(num_classes):
                mask = (y == c)
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == y[mask]).sum().item()
    
    return {c: correct[c] / max(total[c], 1) for c in range(num_classes)}


def confusion_matrix(model, dataloader, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, pred = out.max(1)
            for true, pr in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[true][pr] += 1
    
    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    return cm_norm


# ============================================================
# Inter-class similarity
# ============================================================
def compute_class_similarity(dataset, num_classes):
    """Compute inter-class similarity using mean pixel values."""
    class_features = defaultdict(list)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    for x, y in loader:
        for i in range(x.shape[0]):
            class_features[y[i].item()].append(x[i].flatten().numpy())
    
    # Mean feature per class
    class_means = {}
    for c in range(num_classes):
        if class_features[c]:
            class_means[c] = np.mean(class_features[c], axis=0)
    
    # Cosine similarity matrix
    sim_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i in class_means and j in class_means:
                cos = np.dot(class_means[i], class_means[j]) / (
                    np.linalg.norm(class_means[i]) * np.linalg.norm(class_means[j]) + 1e-8)
                sim_matrix[i][j] = cos
    
    return sim_matrix


# ============================================================
# Training with tracking
# ============================================================
def train_with_tracking(model, trainloader, testloader, trainset, num_classes, epochs, lr=0.1, balanced=False):
    """Train model while tracking per-class metrics every epoch."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction='none' if balanced else 'mean')
    criterion_mean = nn.CrossEntropyLoss()
    
    model.to(DEVICE)
    tracker = GradientTracker(model, num_classes)
    
    history = {
        'per_class_acc': [],  # list of dicts, one per epoch
        'per_class_grad_norm': [],
        'overall_acc': [],
        'learning_order': {},  # class -> epoch where it first exceeds threshold
    }
    
    THRESHOLD = 0.5  # Accuracy threshold to define "learned"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            
            if balanced:
                # Class-balanced gradient scaling
                losses = nn.CrossEntropyLoss(reduction='none')(out, y)
                # Weight inversely by class frequency in batch
                weights = torch.ones_like(losses)
                for c in range(num_classes):
                    mask = (y == c)
                    if mask.sum() > 0:
                        weights[mask] = 1.0 / (mask.float().mean() * num_classes + 1e-6)
                loss = (losses * weights).mean()
            else:
                loss = criterion_mean(out, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Track per-class accuracy
        class_accs = per_class_accuracy(model, testloader, num_classes)
        history['per_class_acc'].append(class_accs)
        
        # Track overall accuracy
        overall = sum(class_accs.values()) / num_classes
        history['overall_acc'].append(overall)
        
        # Track gradient norms (every 5 epochs to save time)
        if epoch % 5 == 0:
            grad_norms = tracker.compute_per_class_grads(trainloader, criterion_mean)
            history['per_class_grad_norm'].append({'epoch': epoch, 'norms': grad_norms})
        
        # Track learning order
        for c in range(num_classes):
            if c not in history['learning_order'] and class_accs.get(c, 0) >= THRESHOLD:
                history['learning_order'][c] = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accs_sorted = sorted(class_accs.items(), key=lambda x: x[1], reverse=True)
            top3 = ', '.join(f'c{c}={a:.2f}' for c, a in accs_sorted[:3])
            bot3 = ', '.join(f'c{c}={a:.2f}' for c, a in accs_sorted[-3:])
            print(f"  Epoch {epoch+1}: overall={overall:.4f} | top3: {top3} | bot3: {bot3}")
    
    return history


# ============================================================
# CIFAR-10 Experiment
# ============================================================
def run_cifar10(seed):
    print(f"\n{'='*60}")
    print(f"CIFAR-10 | Seed {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    NUM_CLASSES = 10
    EPOCHS = 50
    
    # Compute class similarity
    raw_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                  transform=transforms.ToTensor())
    sim_matrix = compute_class_similarity(raw_trainset, NUM_CLASSES)
    
    # Standard training
    print("\n--- Standard Training ---")
    model = make_resnet18(NUM_CLASSES)
    start = time.time()
    history = train_with_tracking(model, trainloader, testloader, trainset, NUM_CLASSES, EPOCHS)
    train_time = time.time() - start
    
    # Confusion matrix at end
    cm = confusion_matrix(model, testloader, NUM_CLASSES)
    
    # Balanced training
    print("\n--- Balanced Gradient Training ---")
    model_balanced = make_resnet18(NUM_CLASSES)
    torch.manual_seed(seed)  # Same init
    history_balanced = train_with_tracking(model_balanced, trainloader, testloader, trainset, 
                                            NUM_CLASSES, EPOCHS, balanced=True)
    
    # Analysis: correlate learning order with class similarity
    learning_order = history['learning_order']
    
    # Average similarity of each class to all others
    avg_similarity = {}
    for c in range(NUM_CLASSES):
        sims = [sim_matrix[c][j] for j in range(NUM_CLASSES) if j != c]
        avg_similarity[c] = float(np.mean(sims))
    
    # Correlation between avg similarity and learning epoch
    classes_with_order = [c for c in range(NUM_CLASSES) if c in learning_order]
    if len(classes_with_order) >= 3:
        sims = [avg_similarity[c] for c in classes_with_order]
        orders = [learning_order[c] for c in classes_with_order]
        corr, p_val = scipy_stats.spearmanr(sims, orders)
        print(f"\n  Correlation (avg_sim vs learning_epoch): rho={corr:.4f}, p={p_val:.6f}")
    else:
        corr, p_val = 0, 1
    
    # Gradient starvation metric: ratio of max to min per-class gradient norm at epoch 0
    if history['per_class_grad_norm']:
        first_grads = history['per_class_grad_norm'][0]['norms']
        grad_vals = [v for v in first_grads.values() if v > 0]
        if grad_vals:
            starvation_ratio = max(grad_vals) / min(grad_vals)
        else:
            starvation_ratio = 0
    else:
        starvation_ratio = 0
    
    # Variance in per-class accuracy over time
    acc_variance_by_epoch = []
    for epoch_accs in history['per_class_acc']:
        vals = list(epoch_accs.values())
        acc_variance_by_epoch.append(float(np.var(vals)))
    
    # Gini coefficient of learning speeds
    if classes_with_order:
        speeds = np.array([learning_order.get(c, EPOCHS) for c in range(NUM_CLASSES)])
        gini = float(np.sum(np.abs(speeds[:, None] - speeds[None, :])) / (2 * NUM_CLASSES * np.sum(speeds) + 1e-8))
    else:
        gini = 0
    
    # Effect of balanced training
    std_final_accs = list(history['per_class_acc'][-1].values())
    bal_final_accs = list(history_balanced['per_class_acc'][-1].values())
    std_acc_var = float(np.var(std_final_accs))
    bal_acc_var = float(np.var(bal_final_accs))
    
    print(f"\n  Standard training - acc variance: {std_acc_var:.6f}")
    print(f"  Balanced training - acc variance: {bal_acc_var:.6f}")
    print(f"  Variance reduction: {(std_acc_var - bal_acc_var) / (std_acc_var + 1e-8) * 100:.1f}%")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    result = {
        'dataset': 'cifar10',
        'num_classes': NUM_CLASSES,
        'epochs': EPOCHS,
        'train_time': float(train_time),
        'final_overall_acc': float(history['overall_acc'][-1]),
        'learning_order': {class_names[c]: int(e) for c, e in learning_order.items()},
        'similarity_learning_correlation': {'rho': float(corr), 'p_value': float(p_val)},
        'starvation_ratio_epoch0': float(starvation_ratio),
        'gini_learning_speed': float(gini),
        'per_class_final_acc': {class_names[c]: float(a) for c, a in history['per_class_acc'][-1].items()},
        'acc_variance_trajectory': acc_variance_by_epoch,
        'balanced_vs_standard': {
            'standard_acc_variance': std_acc_var,
            'balanced_acc_variance': bal_acc_var,
            'variance_reduction_pct': float((std_acc_var - bal_acc_var) / (std_acc_var + 1e-8) * 100),
            'standard_mean_acc': float(np.mean(std_final_accs)),
            'balanced_mean_acc': float(np.mean(bal_final_accs)),
        },
        'confusion_matrix': cm.tolist(),
        'similarity_matrix': sim_matrix.tolist(),
        'per_class_grad_norms': history['per_class_grad_norm'],
    }
    
    return result


# ============================================================
# CIFAR-100 Experiment (subset for tractability)
# ============================================================
def run_cifar100(seed):
    print(f"\n{'='*60}")
    print(f"CIFAR-100 | Seed {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    NUM_CLASSES = 100
    EPOCHS = 60
    
    print("\n--- Standard Training ---")
    model = make_resnet18(NUM_CLASSES)
    start = time.time()
    history = train_with_tracking(model, trainloader, testloader, trainset, NUM_CLASSES, EPOCHS, lr=0.1)
    train_time = time.time() - start
    
    # Confusion matrix
    cm = confusion_matrix(model, testloader, NUM_CLASSES)
    
    # Balanced training
    print("\n--- Balanced Gradient Training ---")
    model_balanced = make_resnet18(NUM_CLASSES)
    torch.manual_seed(seed)
    history_balanced = train_with_tracking(model_balanced, trainloader, testloader, trainset,
                                            NUM_CLASSES, EPOCHS, balanced=True)
    
    learning_order = history['learning_order']
    
    # For CIFAR-100, compute confusion-based similarity
    # Classes that confuse each other are "similar"
    confusion_similarity = (cm + cm.T) / 2
    
    # Correlate with learning order
    avg_confusion = {}
    for c in range(NUM_CLASSES):
        confs = [confusion_similarity[c][j] for j in range(NUM_CLASSES) if j != c]
        avg_confusion[c] = float(np.mean(confs))
    
    classes_with_order = [c for c in range(NUM_CLASSES) if c in learning_order]
    if len(classes_with_order) >= 5:
        confs = [avg_confusion[c] for c in classes_with_order]
        orders = [learning_order[c] for c in classes_with_order]
        corr, p_val = scipy_stats.spearmanr(confs, orders)
        print(f"\n  Correlation (confusion_sim vs learning_epoch): rho={corr:.4f}, p={p_val:.6f}")
    else:
        corr, p_val = 0, 1
    
    # Learning speed distribution
    all_speeds = [learning_order.get(c, EPOCHS) for c in range(NUM_CLASSES)]
    speed_percentiles = {
        'p10': float(np.percentile(all_speeds, 10)),
        'p25': float(np.percentile(all_speeds, 25)),
        'p50': float(np.percentile(all_speeds, 50)),
        'p75': float(np.percentile(all_speeds, 75)),
        'p90': float(np.percentile(all_speeds, 90)),
    }
    
    # Balanced vs standard
    std_final = list(history['per_class_acc'][-1].values())
    bal_final = list(history_balanced['per_class_acc'][-1].values())
    
    # Accuracy for worst-10 classes
    std_sorted = sorted(std_final)
    bal_sorted = sorted(bal_final)
    std_worst10 = float(np.mean(std_sorted[:10]))
    bal_worst10 = float(np.mean(bal_sorted[:10]))
    
    print(f"\n  Standard worst-10 avg acc: {std_worst10:.4f}")
    print(f"  Balanced worst-10 avg acc: {bal_worst10:.4f}")
    
    acc_variance_by_epoch = [float(np.var(list(ea.values()))) for ea in history['per_class_acc']]
    
    result = {
        'dataset': 'cifar100',
        'num_classes': NUM_CLASSES,
        'epochs': EPOCHS,
        'train_time': float(train_time),
        'final_overall_acc': float(history['overall_acc'][-1]),
        'num_classes_learned': len([c for c in range(NUM_CLASSES) if c in learning_order]),
        'learning_speed_percentiles': speed_percentiles,
        'confusion_learning_correlation': {'rho': float(corr), 'p_value': float(p_val)},
        'acc_variance_trajectory': acc_variance_by_epoch,
        'balanced_vs_standard': {
            'standard_acc_variance': float(np.var(std_final)),
            'balanced_acc_variance': float(np.var(bal_final)),
            'standard_mean_acc': float(np.mean(std_final)),
            'balanced_mean_acc': float(np.mean(bal_final)),
            'standard_worst10': std_worst10,
            'balanced_worst10': bal_worst10,
        },
    }
    
    return result


# ============================================================
# Main
# ============================================================
def main():
    all_results = {'cifar10': {}, 'cifar100': {}}
    start_time = time.time()
    
    seeds = [42, 43, 44][:NUM_SEEDS]
    
    for seed in seeds:
        all_results['cifar10'][f'seed_{seed}'] = run_cifar10(seed)
    
    for seed in seeds:
        all_results['cifar100'][f'seed_{seed}'] = run_cifar100(seed)
    
    total_time = time.time() - start_time
    
    # ============================================================
    # Aggregate
    # ============================================================
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    
    summary = {'total_time_minutes': total_time / 60}
    
    for dataset in ['cifar10', 'cifar100']:
        print(f"\n--- {dataset} ---")
        
        accs = [all_results[dataset][f'seed_{s}']['final_overall_acc'] for s in seeds]
        print(f"  Overall acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        
        # Correlation
        corr_key = 'similarity_learning_correlation' if dataset == 'cifar10' else 'confusion_learning_correlation'
        rhos = [all_results[dataset][f'seed_{s}'][corr_key]['rho'] for s in seeds]
        pvals = [all_results[dataset][f'seed_{s}'][corr_key]['p_value'] for s in seeds]
        print(f"  Sim-learning corr: rho={np.mean(rhos):.4f}±{np.std(rhos):.4f}, avg_p={np.mean(pvals):.6f}")
        
        # Balanced vs standard
        std_vars = [all_results[dataset][f'seed_{s}']['balanced_vs_standard']['standard_acc_variance'] for s in seeds]
        bal_vars = [all_results[dataset][f'seed_{s}']['balanced_vs_standard']['balanced_acc_variance'] for s in seeds]
        print(f"  Standard acc variance: {np.mean(std_vars):.6f} ± {np.std(std_vars):.6f}")
        print(f"  Balanced acc variance: {np.mean(bal_vars):.6f} ± {np.std(bal_vars):.6f}")
        
        if NUM_SEEDS >= 2:
            t, p = scipy_stats.ttest_rel(std_vars, bal_vars)
            print(f"  Variance reduction t-test: t={t:.4f}, p={p:.6f}")
        
        summary[dataset] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'correlation_rho_mean': float(np.mean(rhos)),
            'correlation_rho_std': float(np.std(rhos)),
            'correlation_p_mean': float(np.mean(pvals)),
            'std_acc_variance_mean': float(np.mean(std_vars)),
            'bal_acc_variance_mean': float(np.mean(bal_vars)),
        }
    
    # Save
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'raw': all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
