[README.md](https://github.com/user-attachments/files/27161615/README.md)
# 34
easy
# 基于 MobileNetV2 与通道注意力的瓶子图像分类系统

## 摘要

本项目实现了一种基于深度卷积神经网络与迁移学习的瓶子图像分类方法。采用 **MobileNetV2** 作为特征提取骨干网络，在其输出特征图上引入 **Squeeze-and-Excitation（SE）通道注意力模块**，并设计 **自定义两层全连接分类头**（含 BatchNorm 与 Dropout），在 Kaggle 瓶子合成图像数据集上进行训练与评估。实验包含完整的数据集划分、数据增强、训练过程监控、测试集评价指标（准确率、精确率、召回率、F1、混淆矩阵）以及单张/批量预测演示。本项目使用 **PyTorch** 实现，代码结构清晰，便于复现与写入论文。

---

## 1. 研究背景与目的

- **背景**：瓶类物品的自动分类在零售、回收、仓储等场景中有实际需求；基于图像的分类方法可依托卷积神经网络与迁移学习在有限数据上取得较好效果。
- **目的**：在公开瓶子图像数据集上，设计并实现一个轻量、可复现的分类模型，在保证精度的同时便于在普通 PC/笔记本（含 GPU）上训练与部署，并输出完整的评价指标与可视化结果，供本科毕业设计或课程项目使用。

---

## 2. 数据集说明

### 2.1 数据来源

- **数据集名称**：Bottle Synthetic Images Dataset（瓶子合成图像数据集）
- **来源**：Kaggle，`vencerlanz09/bottle-synthetic-images-dataset`
- **用途**：多类别瓶子图像分类（Water Bottle、Plastic Bottles、Beer Bottles、Soda Bottle、Wine Bottle 等）。

### 2.2 数据规模与结构

- 图像总数量约 **50,000** 张（具体以本地解压后的 `Bottle Images` 目录为准）。
- **目录结构**：按类别分文件夹存放，每个子文件夹名为类别标签，例如：
  ```
  Bottle Images/
  ├── Water Bottle/
  ├── Plastic Bottles/
  ├── Beer Bottles/
  ├── Soda Bottle/
  └── Wine Bottle/
  ```
- 支持的图像格式：`.jpg`、`.jpeg`、`.png`（大小写不敏感）。

### 2.3 数据划分

- **训练集 / 验证集 / 测试集** 按比例划分（默认 60% / 20% / 20%），采用 **分层抽样**（stratified sampling）保证各类别在各子集中比例一致。
- 划分比例与随机种子在 `config/bottle_config.py` 中可配置（`TRAIN_RATIO`、`VAL_RATIO`、`TEST_RATIO`，`random_state=42`）。

---

## 3. 方法概述（模型结构）

### 3.1 整体架构

本模型由三部分组成：

1. **特征提取骨干**：**MobileNetV2**（ImageNet 预训练权重），去掉原始分类头，仅保留 `features` 部分，输出空间特征图，通道数为 1280。
2. **通道注意力模块**：**Squeeze-and-Excitation（SE）** 模块，对骨干输出做通道维度的加权，突出对分类更有用的通道。
3. **分类头**：全局平均池化 + 自定义两层全连接 MLP（256 → 128 → num_classes），中间加入 BatchNorm1d 与 Dropout，输出各类别 logits。

### 3.2 Squeeze-and-Excitation（SE）模块

- **作用**：对骨干网络输出的特征图在通道维度上进行重新标定（recalibration）。
- **步骤**：
  - **Squeeze**：对特征图做全局平均池化，得到每个通道的全局描述（长度为 C 的向量）。
  - **Excitation**：通过两层全连接（先压缩到 C/ratio 维，再恢复到 C 维），经 ReLU 与 Sigmoid 得到各通道权重。
  - **Scale**：将权重与原始特征图逐通道相乘，得到加权后的特征图。
- **超参数**：通道压缩比 `ratio=16`（在 `models/attention.py` 的 `SEBlock` 中可调）。

### 3.3 分类头结构

- 全局平均池化：`AdaptiveAvgPool2d((1, 1))`。
- 全连接层 1：1280 → 256，ReLU，BatchNorm1d，Dropout(0.4)。
- 全连接层 2：256 → 128，ReLU，BatchNorm1d，Dropout(0.3)。
- 输出层：128 → num_classes（无激活，训练时配合交叉熵损失）。

### 3.4 损失函数与优化

- **损失函数**：交叉熵损失（CrossEntropyLoss），并采用 **Label Smoothing**（默认 0.1），以缓解过拟合、提升泛化。
- **优化器**：Adam，学习率由 `config/bottle_config.py` 中的 `LEARNING_RATE` 指定（默认 1e-4）。

---

## 4. 实验设置

### 4.1 超参数（可配置）

| 名称 | 默认值 | 说明 |
|------|--------|------|
| IMAGE_SIZE | (224, 224) | 输入图像尺寸（高×宽） |
| BATCH_SIZE | 64 | 批大小 |
| EPOCHS | 5 | 训练轮数（可按需增大，如 15） |
| TRAIN_RATIO | 0.60 | 训练集比例 |
| VAL_RATIO | 0.20 | 验证集比例 |
| TEST_RATIO | 0.20 | 测试集比例 |
| LEARNING_RATE | 1e-4 | Adam 学习率 |
| LABEL_SMOOTHING | 0.1 | 标签平滑系数 |

以上均在 `config/bottle_config.py` 中修改。

### 4.2 数据增强

- **训练集**：Resize 至 224×224、随机水平翻转（p=0.5）、随机旋转（±20°）、随机缩放裁剪（scale 0.8–1.0），再转为 Tensor（数值范围 [0,1]，未在配置中做 ImageNet 标准化，可按需在 `data/bottle_data.py` 的 `build_transforms` 中增加 Normalize）。
- **验证集 / 测试集**：Resize、中心裁剪至 224×224，仅做 ToTensor，不做随机增强。

### 4.3 评价指标

在测试集上计算并输出：

- **准确率（Accuracy）**
- **精确率（Precision）**、**召回率（Recall）**、**F1-score**（按类别及宏平均、加权平均）
- **混淆矩阵**（数值打印 + 可视化保存为 `plots/confusion_matrix.png`）

训练过程中每个 epoch 输出训练集与验证集的 **Loss** 与 **Accuracy**；训练结束后绘制 **训练/验证 Loss 曲线** 与 **Accuracy 曲线**，保存至 `plots/` 目录。

---

## 5. 项目结构

```
瓶子分类/
├── README.md                 # 本说明文档（可作论文附录或实验说明参考）
├── requirements.txt          # Python 依赖列表
├── config/                   # 配置
│   ├── __init__.py
│   └── bottle_config.py      # 超参数、路径、数据划分比例等
├── data/                     # 数据读取与划分
│   ├── __init__.py
│   └── bottle_data.py        # 数据集扫描、划分、Dataset/DataLoader、数据增强
├── models/                   # 模型结构
│   ├── __init__.py
│   ├── attention.py          # SE 通道注意力模块（SEBlock）
│   └── bottle_model.py       # BottleNet：MobileNetV2 + SE + 分类头
├── utils/                    # 工具
│   ├── __init__.py
│   ├── losses.py             # 损失函数（含 label smoothing 的交叉熵）
│   └── eval_utils.py         # 评估、画曲线、混淆矩阵、预测预处理等
├── scripts/                  # 可执行脚本
│   ├── __init__.py
│   ├── train.py              # 训练入口：数据加载、训练循环、保存模型、画图、测试集评估、预测 Demo
│   ├── predict.py            # 预测入口：加载已训练模型，对测试集做单张示例 + 整测试集批量预测
│   └── bottle_app.py         # GUI 入口：图形界面，支持单张/多图选择、开始识别、检测结果与识别日志
├── Bottle Images/            # 数据集目录（需自行放置于项目根目录下，见 7.1）
├── saved_models/             # 保存的模型权重（.pt）
└── plots/                    # 训练曲线、混淆矩阵等图片
```

---

## 6. 环境与依赖

- **Python**：建议 3.8 及以上。
- **主要依赖**（见 `requirements.txt`）：
  - `torch`、`torchvision`：模型与数据预处理
  - `numpy`、`pandas`、`scikit-learn`：数据处理与评估指标
  - `matplotlib`、`Pillow`：可视化与图像读取
  - `tqdm`：训练过程进度条

安装：

```bash
cd 瓶子分类
pip install -r requirements.txt
```

若有 GPU，请根据 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版本的 PyTorch，以加速训练。

---

## 7. 使用说明

以下所有命令均需在**项目根目录**（即 `瓶子分类` 文件夹内）执行，以便正确导入 `config`、`data`、`models`、`utils` 等包。例如：`cd 瓶子分类` 后再执行 `python scripts/train.py`。

### 7.1 准备数据

将瓶子数据集解压至项目根目录下的 **`Bottle Images`** 文件夹中，保持上述目录结构（每类一个子文件夹）。若数据放在其他路径，请在 `config/bottle_config.py` 中修改 `DATASET_DIR` 为实际路径。

### 7.2 训练

在项目根目录（`瓶子分类`）下执行：

```bash
python scripts/train.py
```

脚本将依次完成：读取数据、划分训练/验证/测试集、构建 DataLoader、初始化模型与优化器、按 epoch 训练（终端显示每个 batch 的进度条及每个 epoch 的 Loss/Acc）、保存最佳权重到 `saved_models/bottle_mobilenetv2_custom.pt`、绘制训练曲线到 `plots/`、在测试集上计算并打印准确率/精确率/召回率/F1/混淆矩阵、并做单张与批量预测 Demo。

### 7.3 仅做预测

在已训练并保存模型后，可单独运行预测脚本：

```bash
python scripts/predict.py
```

脚本会加载 `saved_models/bottle_mobilenetv2_custom.pt`，**对与训练时相同划分的测试集**进行：单张图片预测示例、整测试集批量预测（输出每张的预测类别与置信度及总体准确率）。数据划分与训练一致（`random_state=42`）。若要对自定义路径的图片预测，可修改 `scripts/predict.py` 中的路径或增加命令行参数。

### 7.4 图形界面（可选）

运行图形界面，支持**单张或批量选择图片**后点击「开始识别」进行瓶子分类：

```bash
python scripts/bottle_app.py
```

- **控制面板**：**选择图片**（单张）、**批量选择图片**（多张）、**开始识别**。
- **图像预览**：黑底显示当前选中的图片（多张时显示第一张）。
- **检测结果**：单张时显示「类别 + 置信度」；多张时显示「共 N 张」及每张的「文件名: 类别 (置信度)」。
- **处理建议**：根据识别出的瓶子类型给出回收/投放建议（可于 `scripts/bottle_app.py` 中修改 `SUGGESTIONS`）。
- **识别日志**：实时输出模型加载、已选图片数量、每张识别结果及「识别完成」等，便于核对。

界面为白底 + 绿色按钮 + 浅绿标题条 + 黑底图像区，风格参考常见检测/分类 Demo。依赖见 `requirements.txt`（PIL 即 Pillow；tkinter 为 Python 自带）。

---

## 8. 输出与结果

- **模型权重**：`saved_models/bottle_mobilenetv2_custom.pt`（PyTorch `state_dict`）。
- **训练曲线**：`plots/training_loss.png`、`plots/training_accuracy.png`（训练/验证的 Loss 与 Accuracy 随 epoch 变化）。
- **混淆矩阵图**：`plots/confusion_matrix.png`。
- **终端输出**：每个 epoch 的 Train/Val Loss 与 Acc；测试集上的分类报告（含每类及宏平均、加权平均的精确率、召回率、F1）、总体准确率、宏平均 F1、加权平均 F1，以及混淆矩阵数值。

以上内容可直接作为论文中的“实验设置”“结果与分析”部分的依据；曲线与混淆矩阵图可插入论文或答辩 PPT。

---

## 9. 参考文献与说明（供论文引用）

- **MobileNetV2**：Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[C]. CVPR, 2018.
- **Squeeze-and-Excitation Networks**：Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]. CVPR, 2018.
- **数据集**：Kaggle - Bottle Synthetic Images Dataset (vencerlanz09/bottle-synthetic-images-dataset).

---

