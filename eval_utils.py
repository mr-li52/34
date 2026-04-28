"""
评价指标与画图等工具函数。
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from PIL import Image
import torch
from torchvision import transforms as T

from config.bottle_config import IMAGE_SIZE, PLOTS_DIR


def plot_training_curves(history: dict, save_path_prefix: str) -> None:
    """
    绘制并保存训练与验证集上的 accuracy / loss 曲线。
    """
    acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    acc_path = f"{save_path_prefix}_accuracy.png"
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"accuracy 曲线已保存到：{acc_path}")

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_path = f"{save_path_prefix}_loss.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"loss 曲线已保存到：{loss_path}")


def evaluate_on_test_set(model, data_loader, class_names, device: torch.device) -> None:
    """
    在测试集上评估模型，输出准确率、精确率、召回率、F1 和混淆矩阵。
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    target_names: List[str] = list(class_names)

    # 分类报告
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        digits=4,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("分类报告（包含精确率、召回率、F1 等）：")
    print(report_df)

    # 从报告中提取整体 F1（宏平均和加权平均）
    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", None)
    weighted_f1 = report_dict.get("weighted avg", {}).get("f1-score", None)
    if macro_f1 is not None:
        print(f"测试集宏平均 F1：{macro_f1:.4f}")
    if weighted_f1 is not None:
        print(f"测试集加权平均 F1：{weighted_f1:.4f}")

    # 准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"测试集总体准确率：{acc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵：")
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵图已保存到：{cm_path}")


def build_inference_transform() -> T.Compose:
    """
    预测阶段的图像预处理（与验证/测试保持一致）。
    """
    return T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
        ]
    )


def preprocess_image(image_path: str, transform: T.Compose) -> torch.Tensor:
    """
    读取单张图片并预处理成模型可接受的张量。
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def demo_single_and_batch_prediction(
    model,
    sample_paths: List[str],
    class_names: List[str],
    device: torch.device,
) -> None:
    """
    单张图片预测与批量预测 Demo。
    这里默认从测试集中随机选取若干图片进行演示。
    """
    transform = build_inference_transform()
    print("\n===== 单张图片预测 Demo =====")
    for path in sample_paths[:1]:
        img_tensor = preprocess_image(path, transform).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
        pred_label = class_names[pred_idx]
        print(f"图片：{path}")
        print(f"预测类别：{pred_label}")

    print("\n===== 批量图片预测 Demo =====")
    batch_tensors = []
    for path in sample_paths:
        img_tensor = preprocess_image(path, transform)
        batch_tensors.append(img_tensor)
    batch_input = torch.cat(batch_tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch_input)
        batch_pred_indices = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

    batch_pred_labels = [class_names[i] for i in batch_pred_indices]

    for path, label in zip(sample_paths, batch_pred_labels):
        print(f"图片：{path} | 预测类别：{label}")

