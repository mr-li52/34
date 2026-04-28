"""
项目配置与超参数。
在这里可以方便地修改训练轮次、学习率、图像大小等参数，形成你自己的模型配置。
"""

import os

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# 批大小与训练轮次（单阶段训练）
BATCH_SIZE = 64
EPOCHS = 5  # 可根据机器性能酌情增减

# 训练集 / 验证集 / 测试集划分比例
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# 优化器与学习率
LEARNING_RATE = 1e-4

# 分类损失的 label smoothing 系数（0 表示关闭，典型值如 0.1）
LABEL_SMOOTHING = 0.1

# 项目根目录（config 目录的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集路径（本地已解压好的 Bottle Images）
# 如果你的数据集就在当前项目根目录下的 Bottle Images 文件夹中，保持如下配置即可；
# 若位置不同，可以将下面改成你的实际绝对路径。
DATASET_DIR = os.path.join(PROJECT_ROOT, "Bottle Images")

# 模型与输出路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 训练完成后保存的模型文件名（PyTorch 权重）
MODEL_PATH = os.path.join(MODEL_DIR, "bottle_mobilenetv2_custom.pt")

