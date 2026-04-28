"""
瓶子分类识别 - 简单 GUI 界面。

运行方式（在项目根目录「瓶子分类」下）：
    python scripts/bottle_app.py
"""

import os
import sys
from typing import List, Optional, Tuple

# 确保可以从项目根目录导入包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config.bottle_config import MODEL_PATH
from data.bottle_data import load_image_dataframe, split_dataset, create_dataloaders
from models.bottle_model import BottleNet
from utils.eval_utils import build_inference_transform, preprocess_image


# 根据瓶子类型给出简单处理建议（可自行修改）
SUGGESTIONS = {
    "Water Bottle": "建议回收利用或清洗后重复使用。",
    "Plastic Bottles": "请投入可回收物桶，便于资源再利用。",
    "Beer Bottles": "玻璃瓶请投入可回收物桶。",
    "Soda Bottle": "塑料/金属罐请分类投放可回收物。",
    "Wine Bottle": "玻璃酒瓶建议清洗后投入可回收物桶。",
}


def load_model_and_classes():
    """加载模型与类别列表。"""
    df_all = load_image_dataframe()
    df_train, df_val, df_test = split_dataset(df_all)
    _, _, _, class_names = create_dataloaders(df_train, df_val, df_test)
    num_classes = len(class_names)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型：{MODEL_PATH}\n请先运行 scripts/train.py 训练模型。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BottleNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, class_names, device


def predict_image(
    model, image_path: str, class_names: List[str], device
) -> Tuple[str, float]:
    """对单张图片预测，返回 (类别名, 置信度)。"""
    transform = build_inference_transform()
    img_tensor = preprocess_image(image_path, transform).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).cpu().numpy()[0])
        conf = float(probs.cpu().numpy()[0][pred_idx])
    return class_names[pred_idx], conf


# 界面配色（对齐参考图：白底 + 绿色按钮 + 浅绿标题条 + 白底内容区）
COLORS = {
    "bg": "#ffffff",
    "card_title_bg": "#dcfce7",
    "card_content_bg": "#ffffff",
    "card_border": "#bbf7d0",
    "accent": "#22c55e",
    "accent_dark": "#16a34a",
    "text": "#171717",
    "text_light": "#525252",
    "canvas_bg": "#0a0a0a",
    "canvas_border": "#262626",
    "header_bg": "#ffffff",
    "panel_bg": "#f5f5f5",
    "panel_border": "#e5e5e5",
}
FONT_TITLE = ("Microsoft YaHei UI", 15, "bold")
FONT_HEAD = ("Microsoft YaHei UI", 10, "bold")
FONT_BODY = ("Microsoft YaHei UI", 10)
FONT_SMALL = ("Microsoft YaHei UI", 9)
FONT_RESULT = ("Microsoft YaHei UI", 11, "bold")


class BottleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("瓶子分类识别系统")
        self.root.geometry("920x640")
        self.root.minsize(820, 520)
        self.root.configure(bg=COLORS["bg"])

        self.model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = []
        self.device = None
        self.current_image_path: Optional[str] = None
        self.current_image_paths: List[str] = []
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self._setup_styles()
        self._build_ui()
        self._load_model()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Accent.TButton",
            font=FONT_BODY,
            padding=(14, 8),
            background=COLORS["accent"],
            foreground="white",
        )
        style.map("Accent.TButton", background=[("active", COLORS["accent_dark"])])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("TLabel", background=COLORS["bg"], font=FONT_BODY, foreground=COLORS["text"])
        style.configure("Header.TLabel", font=FONT_HEAD, foreground=COLORS["text"])

    def _section_card(self, parent, icon: str, title: str):
        """右侧区块：浅绿标题条 + 白底内容区（与参考图一致）。"""
        outer = tk.Frame(parent, bg=COLORS["card_border"], padx=1, pady=1)
        title_bar = tk.Frame(outer, bg=COLORS["card_title_bg"], height=36)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(
            title_bar, text=f"  {icon}  {title}", font=FONT_HEAD,
            fg=COLORS["text"], bg=COLORS["card_title_bg"]
        ).pack(side=tk.LEFT, padx=10, pady=8)
        content = tk.Frame(outer, bg=COLORS["card_content_bg"], padx=12, pady=10)
        content.pack(fill=tk.BOTH, expand=True)
        return outer, content

    def _build_ui(self):
        # 顶部：标题 + 绿色图标（与参考图一致）
        header = tk.Frame(self.root, bg=COLORS["header_bg"], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="  🌿  ", font=("Segoe UI Symbol", 18), fg=COLORS["accent"], bg=COLORS["header_bg"]
        ).pack(side=tk.LEFT, padx=(20, 0), pady=16)
        tk.Label(
            header,
            text="瓶子分类识别系统",
            font=FONT_TITLE,
            fg=COLORS["text"],
            bg=COLORS["header_bg"],
        ).pack(side=tk.LEFT, padx=(0, 24), pady=16)

        # 控制面板（独立灰色条 + 绿色按钮，开始识别略大）
        panel_wrap = tk.Frame(self.root, bg=COLORS["panel_border"], padx=20, pady=16)
        panel_wrap.pack(fill=tk.X)
        panel = tk.Frame(panel_wrap, bg=COLORS["panel_bg"], padx=16, pady=12)
        panel.pack(fill=tk.X)
        tk.Label(
            panel, text="控制面板", font=FONT_HEAD, fg=COLORS["text"], bg=COLORS["panel_bg"]
        ).pack(side=tk.LEFT, padx=(0, 20))
        btn_style = {"font": FONT_BODY, "bg": COLORS["accent"], "fg": "white", "relief": tk.FLAT, "padx": 16, "pady": 8, "cursor": "hand2", "activebackground": COLORS["accent_dark"], "activeforeground": "white"}
        self.btn_select = tk.Button(panel, text="选择图片", command=self._on_select_image, **btn_style)
        self.btn_select.pack(side=tk.LEFT, padx=8)
        self.btn_select_multi = tk.Button(panel, text="批量选择图片", command=self._on_select_images, **btn_style)
        self.btn_select_multi.pack(side=tk.LEFT, padx=8)
        self.btn_recognize = tk.Button(
            panel, text="开始识别", command=self._on_recognize, state=tk.DISABLED,
            font=FONT_HEAD, padx=20, pady=8, bg=COLORS["accent"], fg="white", relief=tk.FLAT, cursor="hand2",
            activebackground=COLORS["accent_dark"], activeforeground="white",
        )
        self.btn_recognize.pack(side=tk.LEFT, padx=8)

        # 主内容区
        main = tk.Frame(self.root, bg=COLORS["bg"], padx=16, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧：图像预览（黑底 + 细边框，与参考图一致）
        left = tk.Frame(main, bg=COLORS["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            left, text="图像预览", font=FONT_HEAD, fg=COLORS["text"], bg=COLORS["bg"]
        ).pack(anchor=tk.W, pady=(0, 8))
        canvas_frame = tk.Frame(left, bg=COLORS["canvas_border"], padx=2, pady=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS["canvas_bg"],
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.label_placeholder = tk.Label(
            self.canvas,
            text="请点击「选择图片」或「批量选择图片」加载图片",
            font=FONT_SMALL,
            fg="#737373",
            bg=COLORS["canvas_bg"],
        )
        self.placeholder_id = self.canvas.create_window(
            0, 0, window=self.label_placeholder, anchor=tk.CENTER
        )

        # 右侧：检测结果、处理建议、识别日志（浅绿标题条 + 白底内容区）
        right = tk.Frame(main, bg=COLORS["bg"], width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(24, 0))
        right.pack_propagate(False)

        card_result, inner_result = self._section_card(right, "🔍", "检测结果")
        card_result.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            inner_result,
            text="检测到的瓶子类型：",
            font=FONT_SMALL,
            fg=COLORS["text_light"],
            bg=COLORS["card_content_bg"],
        ).pack(anchor=tk.W)
        self.text_result = tk.Text(
            inner_result,
            height=6,
            width=30,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=FONT_RESULT,
            fg=COLORS["accent"],
            bg=COLORS["card_content_bg"],
            relief=tk.FLAT,
            padx=4,
            pady=6,
        )
        self.text_result.pack(fill=tk.X, pady=(6, 0))

        card_sugg, inner_sugg = self._section_card(right, "💡", "处理建议")
        card_sugg.pack(fill=tk.X, pady=(0, 12))
        self.text_suggestion = tk.Text(
            inner_sugg,
            height=3,
            width=30,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=FONT_SMALL,
            fg=COLORS["text"],
            bg=COLORS["card_content_bg"],
            relief=tk.FLAT,
            padx=4,
            pady=6,
        )
        self.text_suggestion.pack(fill=tk.X, pady=(6, 0))

        card_log, inner_log = self._section_card(right, "📋", "识别日志")
        card_log.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        self.text_log = tk.Text(
            inner_log,
            height=14,
            width=30,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9),
            fg=COLORS["text"],
            bg=COLORS["card_content_bg"],
            relief=tk.FLAT,
            padx=4,
            pady=6,
        )
        self.text_log.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self._center_placeholder()
        # 初始提示一行，方便用户知道这里会显示日志
        self._log("识别日志将在这里显示（模型加载、图片选择、识别结果等）。")

    def _center_placeholder(self):
        self.canvas.update_idletasks()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 1 and h > 1:
            self.canvas.coords(self.placeholder_id, w // 2, h // 2)

    def _log(self, msg: str):
        self.text_log.configure(state=tk.NORMAL)
        self.text_log.insert(tk.END, msg + "\n")
        self.text_log.see(tk.END)
        self.text_log.configure(state=tk.DISABLED)

    def _load_model(self):
        try:
            self._log("正在加载模型...")
            self.model, self.class_names, self.device = load_model_and_classes()
            self._log(f"模型加载成功，类别: {self.class_names}")
        except Exception as e:
            self._log(f"加载失败: {e}")
            messagebox.showerror("错误", str(e))

    def _on_select_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有", "*.*")],
        )
        if not path:
            return
        self.current_image_path = path
        self.current_image_paths = [path]
        self._show_image(path)
        self.btn_recognize.configure(state=tk.NORMAL)
        self._log(f"已选择 1 张图片: {os.path.basename(path)}")

    def _on_select_images(self):
        paths = filedialog.askopenfilenames(
            title="批量选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有", "*.*")],
        )
        if not paths:
            return
        paths = list(paths)
        self.current_image_paths = paths
        self.current_image_path = paths[0]
        self._show_image(paths[0])
        self.btn_recognize.configure(state=tk.NORMAL)
        self._log(f"已批量选择 {len(paths)} 张图片")

    def _show_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            # 按 Canvas 大小等比缩放显示
            self.canvas.update_idletasks()
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw < 10:
                cw, ch = 640, 480
            ratio = min(cw / img.width, ch / img.height)
            if ratio > 1:
                ratio = 1.0
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self.placeholder_id, state="hidden")
            self.canvas.delete("img")
            self.canvas.create_image(cw // 2, ch // 2, image=self.current_photo, tags="img")
        except Exception as e:
            self._log(f"显示图片失败: {e}")
            messagebox.showerror("错误", f"无法打开图片：{e}")

    def _on_recognize(self):
        if not self.current_image_paths or not self.model:
            messagebox.showwarning("提示", "请先选择图片（单张或批量），并确保模型已加载。")
            return
        paths = self.current_image_paths
        self.btn_recognize.configure(state=tk.DISABLED)
        self._log("正在识别...")
        self.root.update()
        try:
            results: List[Tuple[str, float]] = []
            for i, path in enumerate(paths):
                label, conf = predict_image(
                    self.model, path, self.class_names, self.device
                )
                results.append((path, label, conf))
                self._log(f"[{i + 1}/{len(paths)}] {os.path.basename(path)} → {label} ({conf:.2%})")

            # 检测结果区：单张显示类别+置信度，多张显示列表
            lines = []
            if len(results) == 1:
                path, label, conf = results[0]
                lines.append(f"{label}")
                lines.append(f"置信度: {conf:.2%}")
                suggestion = SUGGESTIONS.get(label, "请按当地规定进行垃圾分类。")
            else:
                lines.append(f"共 {len(results)} 张")
                for i, (path, label, conf) in enumerate(results, 1):
                    name = os.path.basename(path)
                    if len(name) > 18:
                        name = name[:15] + "…"
                    lines.append(f"{i}. {name}: {label} ({conf:.2%})")
                suggestion = "以上为各张图片的识别结果，处理建议请参考对应瓶子类型。"

            self.text_result.configure(state=tk.NORMAL)
            self.text_result.delete(1.0, tk.END)
            self.text_result.insert(1.0, "\n".join(lines))
            self.text_result.configure(state=tk.DISABLED)

            self.text_suggestion.configure(state=tk.NORMAL)
            self.text_suggestion.delete(1.0, tk.END)
            self.text_suggestion.insert(1.0, suggestion)
            self.text_suggestion.configure(state=tk.DISABLED)

            self._log("识别完成。")
        except Exception as e:
            self._log(f"识别失败: {e}")
            messagebox.showerror("错误", str(e))
        finally:
            self.btn_recognize.configure(state=tk.NORMAL)

    def run(self):
        self.root.after(100, self._center_placeholder)
        self.root.mainloop()


if __name__ == "__main__":
    app = BottleApp()
    app.run()
