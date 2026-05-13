"""
反馈数据保存模块
负责将用户反馈保存为 CSV 记录，并保存原始图片
"""
import os
import csv
from datetime import datetime


# 反馈存储目录
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback.csv")
IMAGES_DIR = os.path.join(FEEDBACK_DIR, "images")

# CSV 列定义
CSV_HEADERS = [
    "created_time",
    "image_name",
    "predicted_class_id",
    "predicted_class_name",
    "correct_class_id",
    "correct_class_name",
    "confidence",
    "remark",
    "saved_image_path",
]


def init_feedback_dir() -> None:
    """
    初始化反馈目录和 CSV 文件（如果不存在则创建）。
    在应用启动时调用。
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"[STARTUP] Feedback CSV created: {FEEDBACK_CSV}")


def save_feedback(
    image_name: str,
    predicted_class_id: int,
    predicted_class_name: str,
    correct_class_id: int,
    correct_class_name: str,
    confidence: float,
    remark: str = "",
    image_bytes: bytes = None,
) -> str:
    """
    保存一条反馈记录到 CSV，同时保存原始图片。

    参数:
        image_name: 原始图片文件名
        predicted_class_id: 模型预测的类别 ID
        predicted_class_name: 模型预测的类别名称
        correct_class_id: 用户修正的类别 ID
        correct_class_name: 用户修正的类别名称
        confidence: 模型预测置信度
        remark: 用户备注（可选）
        image_bytes: 原始图片字节数据（可选）

    返回:
        保存的图片相对路径（空字符串表示未保存图片）
    """
    now = datetime.now()
    created_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---- 保存图片 ----
    saved_image_path = ""
    if image_bytes:
        # 生成文件名: 日期_时间_原始文件名
        safe_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{image_name}"
        image_path = os.path.join(IMAGES_DIR, safe_name)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        saved_image_path = f"feedback/images/{safe_name}"

    # ---- 追加 CSV 记录 ----
    row = {
        "created_time": created_time,
        "image_name": image_name,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": predicted_class_name,
        "correct_class_id": correct_class_id,
        "correct_class_name": correct_class_name,
        "confidence": confidence,
        "remark": remark,
        "saved_image_path": saved_image_path,
    }

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)

    return saved_image_path
