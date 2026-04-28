"""
数据集读取、划分和数据增强相关函数。

功能包含：
- 从本地目录结构中读取图片路径和类别标签；
- 划分训练集 / 验证集 / 测试集；
- 基于 PyTorch 和 torchvision.transforms 构建带有数据增强的 Dataset / DataLoader。
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from config.bottle_config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DATASET_DIR,
)


def load_image_dataframe(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """
    扫描数据集目录，生成包含 Filepath 和 Label 两列的 DataFrame。
    目录结构应类似：
        dataset_dir/
            Water Bottle/
            Plastic Bottles/
            Beer Bottles/
            Soda Bottle/
            Wine Bottle/
    """
    image_dir = Path(dataset_dir)

    # 支持常见图片后缀
    filepaths = list(image_dir.glob(r"**/*.jpg")) + \
                list(image_dir.glob(r"**/*.JPG")) + \
                list(image_dir.glob(r"**/*.jpeg")) + \
                list(image_dir.glob(r"**/*.png"))

    if len(filepaths) == 0:
        raise RuntimeError(f"在目录 {dataset_dir} 下未找到任何图片文件，请检查数据集路径是否正确。")

    labels = [fp.parent.name for fp in filepaths]

    filepaths_series = pd.Series(filepaths, name="Filepath").astype(str)
    labels_series = pd.Series(labels, name="Label")

    df = pd.concat([filepaths_series, labels_series], axis=1)
    print("数据集共计样本数：", len(df))
    print("类别分布：")
    print(df["Label"].value_counts())
    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将完整数据集划分为训练集 / 验证集 / 测试集。
    使用分层抽样，保证各子集类别分布尽量一致。
    """
    train_val_ratio = TRAIN_RATIO + VAL_RATIO
    df_train_val, df_test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["Label"],
        random_state=42,
        shuffle=True,
    )

    val_relative_ratio = VAL_RATIO / train_val_ratio
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_ratio,
        stratify=df_train_val["Label"],
        random_state=42,
        shuffle=True,
    )

    print(f"训练集样本数：{len(df_train)}")
    print(f"验证集样本数：{len(df_val)}")
    print(f"测试集样本数：{len(df_test)}")
    return df_train, df_val, df_test


class BottleDataset(Dataset):
    """
    基于 DataFrame 的自定义 Dataset。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: Dict[str, int],
        transform: T.Compose,
    ):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["Filepath"]
        label_str = row["Label"]
        label_idx = self.label_to_idx[label_str]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


def build_transforms():
    """
    定义训练 / 验证 / 测试阶段的数据增强和预处理。
    """
    train_transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.ToTensor(),
        ]
    )

    eval_transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    基于 DataFrame 创建带有数据增强的训练 / 验证 / 测试 DataLoader。
    """
    class_names: List[str] = sorted(df_train["Label"].unique().tolist())
    label_to_idx: Dict[str, int] = {name: i for i, name in enumerate(class_names)}

    train_transform, eval_transform = build_transforms()

    train_dataset = BottleDataset(df_train, label_to_idx, transform=train_transform)
    val_dataset = BottleDataset(df_val, label_to_idx, transform=eval_transform)
    test_dataset = BottleDataset(df_test, label_to_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, class_names


def walk_through_dir(root_dir: str = DATASET_DIR) -> None:
    """
    辅助函数：遍历目录结构并打印每个子目录下的文件数量。
    主要用于快速检查数据集是否读取正确。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(
            f"当前目录：{dirpath} | 子目录数量：{len(dirnames)} | 文件数量：{len(filenames)}"
        )


if __name__ == "__main__":
    # 简单自测：从本地数据集目录读取并查看基本信息
    walk_through_dir()
    df_all = load_image_dataframe()
    split_dataset(df_all)

