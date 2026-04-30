# DynamicBlockModelStructure
> 动态分块模型架构及其实现方法，以及与 MoE（混合专家）、标准模型的对比实验
---
## 📋 目录
- [项目简介](#-项目简介)
- [问题背景](#-问题背景)
- [核心架构](#-核心架构)
- [实验设计](#-实验设计)
- [实验结果](#-实验结果)
- [代码结构](#-代码结构)
- [快速开始](#-快速开始)
- [配置参数](#-配置参数)
- [技术细节](#-技术细节)
---
## 🌟 项目简介
本项目提出并实现了一种 **动态分块模型** 架构，专为 **持续学习** 场景设计。核心思想是：当模型依次面对多个不同任务时，通过智能决策机制 **动态复用或新建独立的功能块**，同时利用 **重放缓冲区** 和 **增强路由器** 来有效缓解灾难性遗忘问题。
本项目同时实现了 **标准模型** 和 **MoE（混合专家）模型** 作为基线对比，在四个标准视觉数据集的序列任务上进行全面评测。
---
## ❓ 问题背景
在持续学习场景下，模型需要依次学习多个任务（如 FashionMNIST → CIFAR10 → EMNIST → CIFAR100）。传统单一模型面临以下挑战：
| 挑战 | 说明 |
|------|------|
| **灾难性遗忘** | 学习新任务时，旧任务的性能急剧下降 |
| **负迁移** | 新任务的特征可能与旧任务冲突，导致互相干扰 |
| **容量瓶颈** | 固定容量模型难以适应任务类别数的巨大变化（10→62→100类） |
| **路由决策** | MoE等模型的路由器在多任务场景下难以有效分配专家 |
**本项目提出的动态分块方案：**
```
任务0 (10类) → 评估已有块 → 决策：新建/复用 → 训练专用块
任务1 (10类) → 评估已有块 → 复用块0 (损失低) → 无需新建
任务2 (62类) → 评估已有块 → 决策：新建 (类别数不匹配) → 训练新块
任务3 (100类) → 评估已有块 → 决策：新建 → 训练新块
                    ↓
            增强路由器 (CNN-based) 学习将输入正确分配到对应块
```
---
## 🏗️ 核心架构
### 整体架构图
```
┌──────────────────────────────────────────────────────────┐
│                    DynamicBlockModel                      │
│                                                          │
│   输入图像 ──→ ┌─────────────┐                           │
│               │  ImprovedRouter  │  (CNN路由器)           │
│               │  (Conv→BN→Pool→FC)│                       │
│               └──────┬──────────┘                        │
│                      │ 路由权重 argmax                    │
│         ┌────────────┼────────────┐                      │
│         ▼            ▼            ▼                      │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│   │ Block 0  │ │ Block 1  │ │ Block N  │                │
│   │(Backbone │ │(Backbone │ │(Backbone │                │
│   │ + Head)  │ │ + Head)  │ │ + Head)  │                │
│   │ 10类     │ │ 62类     │ │ 100类    │                │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘                │
│        │             │             │                      │
│        └─────────────┼─────────────┘                      │
│                      ▼                                    │
│               输出 logits                                 │
└──────────────────────────────────────────────────────────┘
          ┌─────────────────────────┐
          │     ReplayBuffer        │
          │  Task0: {class→samples} │
          │  Task1: {class→samples} │
          │  Task2: {class→samples} │
          │  Task3: {class→samples} │
          │  → 逐类保存，均匀采样    │
          └─────────────────────────┘
```
### 三种模型对比
| 特性 | StandardModel | MoEModel | DynamicBlockModel |
|------|:---:|:---:|:---:|
| 架构类型 | 单一CNN | 共享backbone + 多专家 | 独立任务块 + 路由器 |
| 参数共享 | 全部共享 | backbone共享，head独立 | 块间完全独立 |
| 持续学习能力 | ❌ 灾难性遗忘严重 | ⚠️ 部分缓解 | ✅ 天然隔离 + 重放 |
| 动态扩展 | ❌ 固定结构 | ❌ 固定专家数 | ✅ 按需新建块 |
| 路由机制 | 无 | 门控网络 | CNN增强路由器 |
| 防遗忘策略 | 无 | 隐式（专家隔离） | 重放缓冲区 + 独立块 |
---
## 🧪 实验设计
### 数据集与任务序列
| 任务顺序 | 数据集 | 类别数 | 图像大小 | 说明 |
|:---:|:---:|:---:|:---:|:---:|
| Task 0 | FashionMNIST | 10 | 32×32 | 灰度时尚物品 |
| Task 1 | CIFAR10 | 10 | 32×32 | 彩色自然图像 |
| Task 2 | EMNIST(byclass) | 62 | 32×32 | 手写字符+数字 |
| Task 3 | CIFAR100 | 100 | 32×32 | 100类细粒度图像 |
> 任务类别跨度从 10 → 62 → 100，充分考验模型的动态扩展能力。
### 训练配置
| 参数 | 值 | 说明 |
|------|:---:|------|
| `MODEL_SCALE` | 1.0 | 模型容量缩放因子 |
| `MAX_TRAIN_SAMPLES` | 2000 | 每任务最多训练样本 |
| `MAX_TEST_SAMPLES` | 500 | 每任务最多测试样本 |
| `TASK_EPOCHS` | 15 | 每任务训练轮数 |
| `ROUTER_JOINT_EPOCHS` | 20 | 路由器联合训练轮数 |
| `REPLAY_SAMPLES_PER_CLASS` | 20 | 重放缓冲区每类保存样本数 |
| `DECISION_THRESHOLD` | 1.5 | 块复用损失阈值 |
### 评估指标
- **Test Accuracy**：各任务测试准确率
- **Forgetting Measure (FM)**：遗忘度量，反映学习新任务后旧任务性能下降幅度
- **Backward Transfer (BWT)**：后向迁移，反映学习新任务对旧任务的促进/抑制
- **Parameter Count**：模型总参数量
---
## 📊 实验结果
![模型对比图](https://t1.chatglm.cn/file/69f307ed344a1dc9bac493b9.png?expired_at=1777967013&sign=da949252276a79971b27652b2c94d692&ext=png)
### 结果分析
#### 1. 各任务测试准确率（左图）
左图展示了三种模型在四个任务上的最终测试准确率对比：
- **StandardModel**：在所有任务上均表现最差，尤其在早期任务（FashionMNIST、CIFAR10）上灾难性遗忘最为严重，这是因为单一模型参数被后续任务完全覆盖。
- **MoEModel**：通过专家隔离机制部分缓解了遗忘问题，但由于所有专家共享同一个backbone，仍存在一定的特征干扰。
- **DynamicBlockModel**：通过独立任务块的天然隔离和重放缓冲区的路由器训练，在各任务上均保持了最优性能。特别是对于 FashionMNIST 和 CIFAR10 等早期任务，准确率显著优于其他两种模型。
#### 2. 灾难性遗忘度量 FM（右图）
右图对比了三种模型的平均遗忘率（FM）：
- **StandardModel**：FM 最高，说明在持续学习过程中遗忘最为严重
- **MoEModel**：FM 有所降低，但仍存在不可忽视的遗忘
- **DynamicBlockModel**：FM 最低，验证了独立任务块 + 重放缓冲区的组合方案在防遗忘方面的有效性
#### 3. 核心发现
```
✅ 动态分块模型在抗遗忘方面显著优于标准模型和MoE模型
✅ 路由器通过重放缓冲区训练后，能够准确将输入路由到对应任务块
✅ 基于损失阈值的块复用决策可有效减少冗余参数
✅ 从10类到100类的类别跨度下，动态扩展机制展现了良好的适应性
```
---
## 📁 代码结构
```
DynamicBlockModelStructure/
│
├── compare_models4.0.py          # 主程序（模型定义 + 训练 + 评测）
├── model_comparison.png           # 实验结果对比图
└── README.md                      # 项目说明文档
```
### 核心类说明
| 类名 | 功能 | 关键方法 |
|------|------|----------|
| `ReplayBuffer` | 逐类保存各任务样本 | `add_task_samples()`, `get_mixed_batch()` |
| `StandardModel` | 基线标准CNN模型 | 3-Conv + 2-FC |
| `MoEModel` | 混合专家模型 | 共享backbone + 门控路由 |
| `SharedFeatureExtractor` | 共享特征提取backbone | 3-Conv + MaxPool |
| `MinimalTaskHead` | 轻量任务头 | 2-FC |
| `TaskBlock` | 独立任务块 (backbone + head) | `forward()` |
| `ImprovedRouter` | CNN增强路由器 | Conv→BN→Pool→FC→num_blocks |
| `DynamicBlockModel` | 动态分块模型（核心） | `add_new_block()`, `evaluate_existing_blocks()`, `train_router_with_replay()`, `forward()` |
---
## 🚀 快速开始
### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.10
torchvision >= 0.11
matplotlib
numpy
seaborn
```
### 安装依赖
```bash
pip install torch torchvision matplotlib numpy seaborn
```
### 运行实验
```bash
# 直接运行主程序，自动下载所有数据集并执行对比实验
python compare_models4.0.py
```
运行后将：
1. 自动下载 FashionMNIST、CIFAR10、EMNIST、CIFAR100 数据集
2. 依次在4个任务上训练三种模型
3. 计算遗忘度和后向迁移指标
4. 生成对比图表 `model_comparison.png`
5. 输出完整的实验报告
---
## ⚙️ 配置参数
在 `compare_models4.0.py` 顶部修改全局配置以调整实验设置：
```python
MODEL_SCALE = 1.0            # 模型容量缩放因子 (0.5=轻量, 1.0=标准, 2.0=大模型)
MAX_TRAIN_SAMPLES = 2000      # 每任务训练样本上限 (None=全量)
MAX_TEST_SAMPLES = 500        # 每任务测试样本上限 (None=全量)
TASK_EPOCHS = 15              # 每任务训练轮数
ROUTER_JOINT_EPOCHS = 20      # 路由器联合训练轮数
REPLAY_SAMPLES_PER_CLASS = 20 # 重放缓冲区每个类别保存的样本数
```
### 任务类别配置
```python
TASK_CLASSES = {0: 10, 1: 10, 2: 62, 3: 100}
# 任务ID → 该任务的类别数，影响模型输出层维度
```
---
## 🔬 技术细节
### 1. 动态块决策流程
```python
# 伪代码
def on_new_task(task_id):
    # Step 1: 从当前任务抽取mini-batch
    mini_batch = sample(task_loader, 100)
    
    # Step 2: 在所有已有块上评估损失
    for block in existing_blocks:
        loss[block] = evaluate(block, mini_batch)
    
    # Step 3: 决策
    best_block = argmin(loss)
    if loss[best_block] <= THRESHOLD and block.classes >= task.num_classes:
        # 复用已有块
        reuse(best_block, task_id)
    else:
        # 新建独立块
        create_new_block(task_id, task.num_classes)
    
    # Step 4: 使用重放缓冲区训练路由器
    train_router_with_replay(buffer, epochs=10)
    
    # Step 5: 训练当前块
    train_block(current_block, task_loader, epochs=15)
    
    # Step 6: 更新重放缓冲区 + 重新训练路由器
    buffer.add_samples(task_id, task_loader)
    train_router_with_replay(buffer, epochs=10)
```
### 2. 路由器架构
```
输入 (B, 3, 32, 32)
    │
    ▼
Conv2d(3, 32, 3, stride=2) → BatchNorm → ReLU    → (B, 32, 16, 16)
    │
    ▼
Conv2d(32, 64, 3, stride=2) → BatchNorm → ReLU   → (B, 64, 8, 8)
    │
    ▼
AdaptiveAvgPool2d(1)                              → (B, 64, 1, 1)
    │
    ▼
Flatten                                            → (B, 64)
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(64, num_blocks) → argmax → 选中的块索引
```
### 3. 重放缓冲区策略
```
┌────────────────────────────────────────────────────┐
│                  ReplayBuffer                       │
│                                                    │
│  Task 0 (FashionMNIST, 10类):                     │
│    class 0 → [img₁, img₂, ..., img₂₀]  (20张)    │
│    class 1 → [img₁, img₂, ..., img₂₀]  (20张)    │
│    ...                                             │
│    class 9 → [img₁, img₂, ..., img₂₀]  (20张)    │
│                                                    │
│  Task 1 (CIFAR10, 10类):  同上结构                │
│  Task 2 (EMNIST, 62类):   同上结构                │
│  Task 3 (CIFAR100, 100类): 同上结构                │
│                                                    │
│  get_mixed_batch(64):                              │
│    从所有任务中均匀采样 → 混合批训练路由器          │
└────────────────────────────────────────────────────┘
```
### 4. 独立块冻结训练
在训练当前任务时，**冻结其他所有块的参数**，仅更新当前块的权重：
```python
# 冻结非当前块
for i, block in enumerate(model.blocks):
    for p in block.parameters():
        p.requires_grad = (i == current_block_idx)
# 仅优化当前块
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```
这确保了各任务块之间的完全隔离，从根本上避免了灾难性遗忘。
---
## 📈 扩展方向
- [ ] 支持更复杂的任务序列和真实世界数据集
- [ ] 引入块间知识蒸馏进一步提升参数效率
- [ ] 探索自适应重放缓冲区大小策略
- [ ] 加入注意力机制增强路由器判别能力
- [ ] 支持多模态输入（文本 + 图像）
- [ ] 实现块合并与压缩机制以控制模型增长
---
## 📝 引用
如果本项目对您的研究有帮助，欢迎引用：
```bibtex
@misc{dynamic-block-model,
  title={DynamicBlockModelStructure: 动态分块模型架构及其实现方法},
  year={2025},
  url={https://github.com/your-username/DynamicBlockModelStructure}
}
```
---
## 📄 License
MIT License
---
> 💡 **核心思想总结**：通过将模型拆分为可动态扩展的独立任务块，配合基于CNN的增强路由器和逐类重放缓冲区，在保持对旧任务记忆的同时，灵活适应新任务的需求，实现了持续学习场景下对灾难性遗忘的有效缓解。
