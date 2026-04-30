import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# ================== 全局配置 ==================
MODEL_SCALE = 1.0                 # 模型容量
MAX_TRAIN_SAMPLES = 2000          # 每任务训练样本上限
MAX_TEST_SAMPLES = 500            # 每任务测试样本上限
TASK_EPOCHS = 15                  # 每任务训练轮数
ROUTER_JOINT_EPOCHS = 20          # 路由器联合训练轮数
REPLAY_SAMPLES_PER_CLASS = 20     # 重放缓冲区每个类别保存的样本数
# =================================================

TASK_CLASSES = {0: 10, 1: 10, 2: 62, 3: 100}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------- 重放缓冲区 ----------
class ReplayBuffer:
    def __init__(self, samples_per_class=20):
        self.samples_per_class = samples_per_class
        self.buffer = {}  # task_id -> {class_label: [(img, label), ...]}

    def add_task_samples(self, task_id, loader):
        class_counts = defaultdict(int)
        task_samples = defaultdict(list)
        for x_batch, y_batch in loader:
            for img, label in zip(x_batch, y_batch):
                label = int(label)
                if class_counts[label] >= self.samples_per_class:
                    continue
                task_samples[label].append((img.clone(), label))
                class_counts[label] += 1
            if all(cnt >= self.samples_per_class for cnt in class_counts.values()):
                break
        self.buffer[task_id] = task_samples

    def get_mixed_batch(self, batch_size=32):
        """从所有任务中均匀采样，返回 (images, labels, task_ids)"""
        if not self.buffer:
            return None, None, None
        all_images, all_labels, all_tasks = [], [], []
        task_ids = list(self.buffer.keys())
        for _ in range(batch_size):
            tid = np.random.choice(task_ids)
            class_label = np.random.choice(list(self.buffer[tid].keys()))
            img, label = self.buffer[tid][class_label][np.random.randint(len(self.buffer[tid][class_label]))]
            all_images.append(img.unsqueeze(0))
            all_labels.append(label)
            all_tasks.append(tid)
        images = torch.cat(all_images, dim=0)
        labels = torch.tensor(all_labels)
        task_ids = torch.tensor(all_tasks)
        return images, labels, task_ids

# ---------- 标准模型 ----------
class StandardModel(nn.Module):
    def __init__(self, num_classes=100, scale=1.0):
        super().__init__()
        c1 = max(1, int(16 * scale))
        c2 = max(1, int(32 * scale))
        c3 = max(1, int(32 * scale))
        fc_hidden = max(1, int(128 * scale))
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(c3 * 4 * 4, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: x = x[:, :3, :, :]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class MoEModel(nn.Module):
    def __init__(self, num_experts=2, num_classes=100, scale=1.0):
        super().__init__()
        self.num_experts = num_experts
        c1 = max(1, int(16 * scale))
        c2 = max(1, int(32 * scale))
        c3 = max(1, int(32 * scale))
        expert_hidden = max(1, int(64 * scale))
        gate_hidden = max(1, int(32 * scale))
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        flat_dim = c3 * 4 * 4
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(flat_dim, expert_hidden), nn.ReLU(),
                          nn.Dropout(0.25), nn.Linear(expert_hidden, num_classes))
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(flat_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, num_experts), nn.Softmax(dim=1)
        )

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: x = x[:, :3, :, :]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        features = self.dropout(x).view(x.size(0), -1)
        weights = self.gate(features)
        outs = [exp(features) for exp in self.experts]
        return sum(weights[:, i:i+1] * outs[i] for i in range(self.num_experts))

# ---------- 独立任务块 ----------
class TaskBlock(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        self.backbone = SharedFeatureExtractor(scale)
        self.head = MinimalTaskHead(self.backbone.flat_dim, num_classes, scale)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

class SharedFeatureExtractor(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        c1 = max(1, int(16 * scale))
        c2 = max(1, int(32 * scale))
        c3 = max(1, int(32 * scale))
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.flat_dim = c3 * 4 * 4

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: x = x[:, :3, :, :]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return self.dropout(x).view(-1, self.flat_dim)

class MinimalTaskHead(nn.Module):
    def __init__(self, input_dim, num_classes=100, scale=1.0):
        super().__init__()
        hidden = max(1, int(128 * scale))
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)

# ---------- 增强路由器 ----------
class ImprovedRouter(nn.Module):
    def __init__(self, num_blocks, scale=1.0):
        super().__init__()
        c1 = max(1, int(32 * scale))
        c2 = max(1, int(64 * scale))
        self.conv1 = nn.Conv2d(3, c1, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(c2, num_blocks)

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3: x = x[:, :3, :, :]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# ---------- 动态模型（含重放路由器训练）----------
class DynamicBlockModel(nn.Module):
    def __init__(self, initial_num_blocks=1, num_classes_per_task=100, decision_threshold=1.5, scale=1.0):
        super().__init__()
        self.scale = scale
        self.num_classes_per_task = num_classes_per_task
        self.decision_threshold = decision_threshold
        self.blocks = nn.ModuleList()
        self.block_metadata = []
        self.router = None
        for i in range(initial_num_blocks):
            self.add_new_block(i, num_classes=num_classes_per_task)

    def add_new_block(self, task_id, num_classes=None):
        if num_classes is None: num_classes = self.num_classes_per_task
        block = TaskBlock(num_classes, scale=self.scale)
        try:
            device = next(self.parameters()).device
            block = block.to(device)
        except StopIteration:
            pass
        self.blocks.append(block)
        self.block_metadata.append({
            'task_id': task_id, 'num_classes': num_classes,
            'tasks_handled': [task_id], 'block_type': 'independent'
        })
        self._update_router()
        return len(self.blocks) - 1

    def _update_router(self):
        new_router = ImprovedRouter(len(self.blocks), scale=self.scale)
        if self.router is not None:
            device = next(self.router.parameters()).device
            new_router = new_router.to(device)
        self.router = new_router

    def train_router_with_replay(self, replay_buffer, device, epochs=10, lr=1e-3, batch_size=64, steps_per_epoch=100):
        """使用重放缓冲区中的混合批训练路由器"""
        if len(self.blocks) <= 1 or replay_buffer is None or len(replay_buffer.buffer) == 0:
            return
        self.router.to(device).train()
        opt = optim.Adam(self.router.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            for _ in range(steps_per_epoch):
                images, _, task_ids = replay_buffer.get_mixed_batch(batch_size)
                if images is None:
                    break
                images = images.to(device)
                task_ids = task_ids.to(device)
                # 获取每个任务对应的块索引
                block_idxs = []
                for tid in task_ids.tolist():
                    bidx = self.get_task_block(tid)
                    if bidx == -1:
                        bidx = 0
                    block_idxs.append(bidx)
                block_idxs = torch.tensor(block_idxs, dtype=torch.long, device=device)
                opt.zero_grad()
                logits = self.router(images)
                loss = F.cross_entropy(logits, block_idxs)
                loss.backward()
                opt.step()
                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(1)
                correct += (preds == block_idxs).sum().item()
                total += images.size(0)
            if total > 0:
                print(f'  路由器重放训练 Epoch {epoch+1}/{epochs}: Loss={total_loss/total:.4f}, Acc={100*correct/total:.2f}%')
        self.router.eval()

    def evaluate_existing_blocks(self, dataloader, criterion, device, num_samples=100):
        self.eval()
        block_losses = [[] for _ in range(len(self.blocks))]
        processed = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                if processed >= num_samples: break
                inputs, targets = inputs.to(device), targets.to(device)
                for bidx in range(len(self.blocks)):
                    try:
                        out = self.blocks[bidx](inputs)
                        if out.shape[1] <= int(targets.max()): loss = float('inf')
                        else: loss = criterion(out, targets).item()
                    except: loss = float('inf')
                    block_losses[bidx].append(loss)
                processed += inputs.size(0)
        avg = [np.mean(l) if l else float('inf') for l in block_losses]
        if not avg: return None, float('inf')
        best = int(np.argmin(avg))
        return best, avg[best]

    def forward(self, x, task_id=None, return_block_info=False):
        if task_id is not None:
            for bidx, meta in enumerate(self.block_metadata):
                if task_id in meta['tasks_handled']:
                    out = self.blocks[bidx](x)
                    return (out, bidx) if return_block_info else out
            out = self.blocks[0](x)
            return (out, 0) if return_block_info else out
        if self.router is not None and len(self.blocks) > 0:
            routing_logits = self.router(x)
            chosen = routing_logits.argmax(dim=1)
            B = x.size(0)
            max_classes = max(meta['num_classes'] for meta in self.block_metadata)
            out_all = torch.zeros(B, max_classes, device=x.device)
            for u in chosen.unique():
                mask = (chosen == u)
                if mask.sum() == 0: continue
                idx = mask.nonzero(as_tuple=True)[0]
                out_block = self.blocks[u](x[idx])
                if out_block.shape[1] < max_classes:
                    pad = torch.full((out_block.size(0), max_classes - out_block.shape[1]), -1e9, device=x.device)
                    out_block = torch.cat([out_block, pad], dim=1)
                out_all[idx] = out_block
            return (out_all, chosen) if return_block_info else out_all
        out = self.blocks[0](x)
        return (out, 0) if return_block_info else out

    def get_task_block(self, task_id):
        for bidx, meta in enumerate(self.block_metadata):
            if task_id in meta['tasks_handled']: return bidx
        return -1

# ---------- 数据加载 ----------
def sample_dataset(dataset, max_samples, seed=42):
    if max_samples is not None and len(dataset) > max_samples:
        indices = list(range(len(dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        return Subset(dataset, indices[:max_samples])
    return dataset

def create_fashion_mnist(batch_size=32, max_train=None, max_test=None):
    transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(3),
                                    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_ds = sample_dataset(train_ds, max_train)
    test_ds = sample_dataset(test_ds, max_test)
    train_ldr = DataLoader(train_ds, batch_size, shuffle=True)
    test_ldr = DataLoader(test_ds, batch_size, shuffle=False)
    return train_ldr, test_ldr

def create_cifar10(batch_size=32, max_train=None, max_test=None):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                    transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_ds = sample_dataset(train_ds, max_train)
    test_ds = sample_dataset(test_ds, max_test)
    train_ldr = DataLoader(train_ds, batch_size, shuffle=True)
    test_ldr = DataLoader(test_ds, batch_size, shuffle=False)
    return train_ldr, test_ldr

def create_emnist(batch_size=32, max_train=None, max_test=None):
    transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(3),
                                    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.EMNIST('./data', split='byclass', train=True, download=True, transform=transform)
    test_ds = datasets.EMNIST('./data', split='byclass', train=False, download=True, transform=transform)
    train_ds = sample_dataset(train_ds, max_train)
    test_ds = sample_dataset(test_ds, max_test)
    train_ldr = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2)
    test_ldr = DataLoader(test_ds, batch_size, shuffle=False, num_workers=1)
    return train_ldr, test_ldr, 62

def create_cifar100(batch_size=32, max_train=None, max_test=None):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                    transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    train_ds = sample_dataset(train_ds, max_train)
    test_ds = sample_dataset(test_ds, max_test)
    train_ldr = DataLoader(train_ds, batch_size, shuffle=True)
    test_ldr = DataLoader(test_ds, batch_size, shuffle=False)
    return train_ldr, test_ldr

def create_mini_batch(loader, num_samples=100):
    data, targets = [], []
    collected = 0
    for x, y in loader:
        data.append(x); targets.append(y)
        collected += x.size(0)
        if collected >= num_samples: break
    x_cat = torch.cat(data, dim=0)[:num_samples]
    y_cat = torch.cat(targets, dim=0)[:num_samples]
    return DataLoader(TensorDataset(x_cat, y_cat), batch_size=32, shuffle=True)

# ---------- 训练与测试 ----------
def train_standard(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        print(f'    Epoch {epoch+1}/{epochs}: Loss={running_loss/len(loader):.4f}, Acc={100*correct/total:.2f}%')

def test_standard(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss/len(loader), 100*correct/total

def train_dynamic(model, loader, criterion, device, epochs, task_id, threshold, replay_buffer):
    if task_id not in [m['task_id'] for m in model.block_metadata]:
        print(f'  决策任务 {task_id} ...')
        mini_loader = create_mini_batch(loader, 100)
        best_idx, best_loss = model.evaluate_existing_blocks(mini_loader, criterion, device, 100)
        num_cls = TASK_CLASSES.get(task_id, 100)
        can_reuse = (best_idx is not None and best_loss <= threshold and
                     model.block_metadata[best_idx]['num_classes'] >= num_cls)
        if can_reuse:
            print(f'  重用块 {best_idx} (损失 {best_loss:.4f})')
            model.block_metadata[best_idx]['tasks_handled'].append(task_id)
            block_idx = best_idx
        else:
            print(f'  新建块 (类别数 {num_cls})')
            block_idx = model.add_new_block(task_id, num_cls)
            model.to(device)
        # 使用重放缓冲区训练路由器（包含当前任务和所有旧任务）
        # 首先需要将当前任务的部分样本加入缓冲区，然后再训练路由器？实际上缓冲区还未包含当前任务样本，但我们可以在添加后再训练。
        # 由于 train_router_with_replay 依赖缓冲区，我们可先临时添加一部分当前任务样本到缓冲区
        # 简便起见：直接调用训练，缓冲区中只有旧任务，可能会偏向旧任务但可接受。更好的做法是先添加，再训练。
        # 这里我们选择先调用 train_router_with_replay（仅使用旧任务样本），然后再添加当前任务样本。
        model.train_router_with_replay(replay_buffer, device, epochs=10, lr=1e-3, steps_per_epoch=100)
    else:
        block_idx = model.get_task_block(task_id)

    # 冻结除当前块外的所有块
    model.train()
    for i, block in enumerate(model.blocks):
        for p in block.parameters():
            p.requires_grad = (i == block_idx)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(epochs):
        running_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model.blocks[block_idx](x)
            if out.shape[1] <= y.max():
                pad = torch.full((out.size(0), y.max().item()+1 - out.shape[1]), -1e9, device=device)
                out = torch.cat([out, pad], dim=1)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f'    Epoch {epoch+1}/{epochs}: Loss={running_loss/len(loader):.4f}, Acc={100*correct/total:.2f}%')
    for p in model.parameters():
        p.requires_grad = True

    # 将当前任务样本加入重放缓冲区
    print(f'  更新重放缓冲区(任务 {task_id})')
    replay_buffer.add_task_samples(task_id, loader)
    # 缓冲区更新后，再次训练路由器以包含新任务
    model.train_router_with_replay(replay_buffer, device, epochs=10, lr=1e-3, steps_per_epoch=100)
    return block_idx

def test_dynamic_model(model, loader, criterion, device, task_id=None):
    model.eval()
    loss, correct, total = 0, 0, 0
    block_usage = defaultdict(int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if task_id is not None:
                bidx = model.get_task_block(task_id)
                if bidx == -1: bidx = 0
                out = model.blocks[bidx](x)
                block_usage[bidx] += x.size(0)
            else:
                out, chosen = model(x, return_block_info=True)
                for c in chosen.unique():
                    block_usage[c.item()] += (chosen == c).sum().item()
            loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss/len(loader), 100*correct/total, block_usage

# ---------- 结果展示 ----------
def create_comparison_charts(results):
    sns.set(style='whitegrid')
    palette = sns.color_palette('Set2', 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tasks = ['FashionMNIST', 'CIFAR10', 'EMNIST', 'CIFAR100']
    models = ['Standard', 'MoE', 'Dynamic']
    x = np.arange(len(tasks))
    width = 0.25
    for i, m in enumerate(models):
        accs = [results[f'{m}_{t}']['test_acc'] for t in tasks]
        bars = ax1.bar(x + (i-1)*width, accs, width, label=m, color=palette[i], alpha=0.85)
        for j, v in enumerate(accs):
            ax1.text(x[j] + (i-1)*width, v+0.5, f'{v:.1f}', ha='center', fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(tasks, rotation=15)
    ax1.set_title('Test Accuracy per Task (Independent Blocks + Replay Router)')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    fm_vals = [results.get(f'{m}_FM', 0) for m in models]
    bars = ax2.bar(models, fm_vals, color=palette, alpha=0.85)
    for bar, v in zip(bars, fm_vals):
        ax2.text(bar.get_x()+bar.get_width()/2., v+0.01, f'{v:.2f}%', ha='center', fontsize=10)
    ax2.set_title('Catastrophic Forgetting (FM)'); ax2.set_ylabel('FM (%)'); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('model_comparison.png', dpi=300); plt.show()

def print_report(results):
    tasks = ['FashionMNIST', 'CIFAR10', 'EMNIST', 'CIFAR100']
    models = ['Standard', 'MoE', 'Dynamic']
    print("\n" + "="*70)
    print(" Summary Report (Independent Blocks + Replay Router) ".center(70, "="))
    header = f"{'Model':<12}" + "".join(f"{t:>14}" for t in tasks) + f"{'Avg':>10}"
    print(header); print("-"*(12+14*len(tasks)+10))
    for m in models:
        accs = [results[f'{m}_{t}']['test_acc'] for t in tasks]
        avg = np.mean(accs)
        row = f"{m:<12}" + "".join(f"{v:>13.2f}%" for v in accs) + f"{avg:>9.2f}%"
        print(row)
    print("-"*(12+14*len(tasks)+10))
    for m in models:
        print(f"{m} FM = {results.get(f'{m}_FM', 0):.2f}%  BWT = {results.get(f'{m}_BWT', 0):.2f}%")
    print("\nModel Parameter Counts (Total):")
    for m, p in results.get('param_counts', {}).items():
        print(f"  {m}: {p:,}")
    print("\nDynamic Model Final Structure:")
    for i, meta in enumerate(results.get('dyn_meta', [])):
        print(f"  Block {i}: Task {meta['task_id']} ({meta['num_classes']} classes), handles tasks {meta['tasks_handled']}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}  |  模型缩放: {MODEL_SCALE}  |  训练样本限制: {MAX_TRAIN_SAMPLES}")
    criterion = nn.CrossEntropyLoss()
    batch_size = 64

    std_model = StandardModel(num_classes=100, scale=MODEL_SCALE).to(device)
    moe_model = MoEModel(num_experts=2, num_classes=100, scale=MODEL_SCALE).to(device)
    dyn_model = DynamicBlockModel(initial_num_blocks=1, num_classes_per_task=10, scale=MODEL_SCALE).to(device)
    print(f"参数量: Standard={count_parameters(std_model):,}, MoE={count_parameters(moe_model):,}, Dynamic(初始)={count_parameters(dyn_model):,}")

    replay_buffer = ReplayBuffer(samples_per_class=REPLAY_SAMPLES_PER_CLASS)

    print("\n加载抽样数据...")
    f_train, f_test = create_fashion_mnist(batch_size, max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES)
    c10_train, c10_test = create_cifar10(batch_size, max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES)
    e_train, e_test, emnist_nclasses = create_emnist(batch_size, max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES)
    c100_train, c100_test = create_cifar100(batch_size, max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES)
    TASK_CLASSES[2] = emnist_nclasses

    tasks = [
        (f_train, f_test, 0, 'FashionMNIST'),
        (c10_train, c10_test, 1, 'CIFAR10'),
        (e_train, e_test, 2, 'EMNIST'),
        (c100_train, c100_test, 3, 'CIFAR100')
    ]

    results = {}
    acc_after = {m: {i: None for i in range(4)} for m in ['Standard','MoE','Dynamic']}

    for tid, (train_ldr, test_ldr, ncls, name) in enumerate(tasks):
        print(f"\n{'='*60}\n任务 {tid}: {name} ({ncls} 类)\n{'='*60}")
        opt_std = optim.Adam(std_model.parameters(), lr=1e-3 if tid==0 else 5e-4)
        opt_moe = optim.Adam(moe_model.parameters(), lr=1e-3 if tid==0 else 5e-4)
        train_standard(std_model, train_ldr, opt_std, criterion, device, TASK_EPOCHS)
        train_standard(moe_model, train_ldr, opt_moe, criterion, device, TASK_EPOCHS)

        std_loss, std_acc = test_standard(std_model, test_ldr, criterion, device)
        moe_loss, moe_acc = test_standard(moe_model, test_ldr, criterion, device)
        results[f'Standard_{name}'] = {'test_loss': std_loss, 'test_acc': std_acc}
        results[f'MoE_{name}'] = {'test_loss': moe_loss, 'test_acc': moe_acc}
        acc_after['Standard'][tid] = std_acc
        acc_after['MoE'][tid] = moe_acc
        print(f'  Standard 准确率: {std_acc:.2f}%, MoE 准确率: {moe_acc:.2f}%')

        print(f'  Dynamic 模型:')
        train_dynamic(dyn_model, train_ldr, criterion, device, TASK_EPOCHS, task_id=tid, threshold=1.5, replay_buffer=replay_buffer)
        dyn_loss, dyn_acc, usage = test_dynamic_model(dyn_model, test_ldr, criterion, device, task_id=tid)
        results[f'Dynamic_{name}'] = {'test_loss': dyn_loss, 'test_acc': dyn_acc, 'block_usage': usage}
        acc_after['Dynamic'][tid] = dyn_acc
        print(f'  Dynamic 准确率: {dyn_acc:.2f}%, 块使用: {dict(usage)}')

    # 路由器联合训练（使用重放缓冲区）
    print("\n路由器联合训练 (使用复述缓冲区)...")
    dyn_model.train_router_with_replay(replay_buffer, device, epochs=ROUTER_JOINT_EPOCHS, lr=1e-3, steps_per_epoch=100)

    print("\n公平遗忘计算 (Task-Agnostic)...")
    for m in ['Standard','MoE','Dynamic']:
        fm_total, bwt_total, cnt = 0, 0, 0
        for k in range(1, 4):
            for j in range(k):
                a_jj = acc_after[m][j]
                _, test_ldr_j, _, _ = tasks[j]
                if m == 'Dynamic':
                    _, a_kj, _ = test_dynamic_model(dyn_model, test_ldr_j, criterion, device, task_id=None)
                else:
                    _, a_kj = test_standard(std_model if m=='Standard' else moe_model, test_ldr_j, criterion, device)
                f = max(0, a_jj - a_kj)
                b = a_kj - a_jj
                fm_total += f; bwt_total += b; cnt += 1
                print(f"  {m} f_{j},{k} = {f:.2f}%")
        results[f'{m}_FM'] = fm_total / cnt
        results[f'{m}_BWT'] = bwt_total / cnt
        print(f'  => {m} 平均 FM = {results[f"{m}_FM"]:.2f}%')

    results['param_counts'] = {
        'Standard': count_parameters(std_model),
        'MoE': count_parameters(moe_model),
        'Dynamic': count_parameters(dyn_model)
    }
    results['dyn_meta'] = dyn_model.block_metadata

    create_comparison_charts(results)
    print_report(results)