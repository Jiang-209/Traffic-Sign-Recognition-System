"""
图片预处理模块
将上传的图片转换为模型可接受的输入格式：
RGB -> resize(32x32) -> 灰度化 -> 归一化到 [-0.5, 0.5] -> reshape(1,32,32,1)
"""
import cv2
import numpy as np

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    对上传图片的二进制数据进行预处理。

    处理流程（与训练时完全一致）：
    1. 从 bytes 解码为 BGR 图像
    2. BGR 转 RGB
    3. Resize 到 32×32
    4. RGB 转灰度图
    5. 归一化：gray / 255.0 - 0.5

    参数:
        image_bytes: 图片文件的原始字节数据

    返回:
        np.ndarray, shape=(1, 32, 32, 1), dtype=float32
    """
    # 1. 从内存中解码图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("无法解码图片，请确认上传的是有效的 jpg/jpeg/png 格式文件")

    # 2. BGR -> RGB（OpenCV 默认 BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Resize 到 32×32
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # 4. RGB -> 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 5. 归一化到 [-0.5, 0.5]（与训练代码一致：gray / 255.0 - 0.5）
    gray = gray.astype(np.float32) / 255.0 - 0.5

    # 6. Reshape 为 (1, 32, 32, 1) 以匹配模型输入
    processed = gray.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return processed
