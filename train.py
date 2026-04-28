"""
训练与评估脚本（PyTorch 版）。

运行方式：
    python -m scripts.train
或在项目根目录下：
    python scripts/train.py
"""

import os
import sys

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

# 确保可以从项目根目录导入 config/data/models/utils 等包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.bottle_config import (
    EPOCHS,
    MODEL_PATH,
    PLOTS_DIR,
)
from data.bottle_data import (
    load_image_dataframe,
    split_dataset,
    create_dataloaders,
)
from models.bottle_model import BottleNet
from utils.losses import classification_loss
from utils.eval_utils import (
    plot_training_curves,
    evaluate_on_test_set,
    demo_single_and_batch_prediction,
)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(data_loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, data_loader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 1. 从本地数据集目录读取数据
    df_all = load_image_dataframe()

    # 2. 划分训练 / 验证 / 测试集
    df_train, df_val, df_test = split_dataset(df_all)

    # 3. 创建 Dataset / DataLoader
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        df_train, df_val, df_test
    )
    num_classes = len(class_names)
    print(f"检测到的类别数：{num_classes}")
    print("类别名称顺序：", class_names)

    # 4. 构建模型
    model = BottleNet(num_classes=num_classes).to(device)

    # 5. 定义损失函数和优化器
    criterion = classification_loss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters())

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # 6. 训练
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    # 7. 保存模型
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型权重已保存到：{MODEL_PATH}")

    # 8. 绘制训练曲线
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_training_curves(history, save_path_prefix=os.path.join(PLOTS_DIR, "training"))

    # 9. 在测试集上评估
    evaluate_on_test_set(model, test_loader, class_names=class_names, device=device)

    # 10. 单张图片与批量预测 Demo（从测试集中随机抽取若干张）
    sample_df = df_test.sample(n=min(5, len(df_test)), random_state=42)
    sample_paths = sample_df["Filepath"].tolist()
    demo_single_and_batch_prediction(model, sample_paths, class_names, device=device)


if __name__ == "__main__":
    main()

