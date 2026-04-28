"""
独立预测脚本（PyTorch 版）。

运行方式示例：
    python -m scripts.predict
或在项目根目录下：
    python scripts/predict.py
"""

import os
import sys
from typing import List

import torch

# 确保可以从项目根目录导入 config/data/models/utils 等包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.bottle_config import MODEL_PATH
from data.bottle_data import load_image_dataframe, split_dataset, create_dataloaders
from models.bottle_model import BottleNet
from utils.eval_utils import build_inference_transform, preprocess_image


def load_trained_model(num_classes: int, model_path: str = MODEL_PATH, device=None):
    """
    从指定路径加载已经训练好的 PyTorch 模型。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}\n"
            f"请先运行 scripts/train.py 进行训练，或将现有模型拷贝到该路径。"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型：{model_path}")
    model = BottleNet(num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_single_image(model, image_path: str, class_names: List[str], device) -> None:
    """
    对单张图片进行预测并打印结果。
    """
    transform = build_inference_transform()
    img_tensor = preprocess_image(image_path, transform).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
    pred_label = class_names[pred_idx]
    print(f"单张图片预测：{image_path}")
    print(f"预测类别：{pred_label}")


def predict_test_set(model, test_loader, class_names: List[str], device, max_print: int = 20) -> None:
    """
    对测试集进行批量预测，并打印前 max_print 条结果及总体统计。
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []  # 需要从 DataLoader 的 dataset 拿路径，这里只统计对错

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    n_total = len(all_preds)
    n_correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    acc = n_correct / n_total if n_total else 0
    print(f"测试集批量预测：共 {n_total} 张，预测正确 {n_correct} 张，准确率 {acc:.4f}")

    # 打印前 max_print 条（需要路径则从 dataset 取）
    dataset = test_loader.dataset
    print(f"前 {min(max_print, n_total)} 条预测结果：")
    for i in range(min(max_print, n_total)):
        path = dataset.df.iloc[i]["Filepath"]
        pred_label = class_names[all_preds[i]]
        true_label = class_names[all_labels[i]]
        mark = "✓" if all_preds[i] == all_labels[i] else "✗"
        print(f"  {mark} {path} | 预测: {pred_label} | 真实: {true_label}")
    if n_total > max_print:
        print(f"  ... 其余 {n_total - max_print} 条未逐条打印。")


def main():
    # 1. 读取数据集并按与训练时相同方式划分（random_state=42），得到测试集及 test_loader
    df_all = load_image_dataframe()
    df_train, df_val, df_test = split_dataset(df_all)
    _, _, test_loader, class_names = create_dataloaders(df_train, df_val, df_test)

    num_classes = len(class_names)
    print("类别名称顺序（与预测索引一一对应）：", class_names)

    # 2. 加载已经训练好的模型
    model, device = load_trained_model(num_classes=num_classes)

    # 3. 单张图片预测示例（从测试集中随机选一张）
    sample_path = df_test.sample(n=1, random_state=123)["Filepath"].iloc[0]
    print("\n===== 单张图片预测（来自测试集）=====")
    predict_single_image(model, sample_path, class_names, device)

    # 4. 批量预测：仅对测试集进行预测
    print("\n===== 批量预测（测试集）=====")
    predict_test_set(model, test_loader, class_names, device, max_print=20)


if __name__ == "__main__":
    main()

