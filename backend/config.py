"""
项目配置文件
"""
import os

# ============================================================
# 模型配置
# ============================================================
# 模型文件路径（不含扩展名），环境变量可覆盖
# 支持 TF1 checkpoint 格式（.meta / .index / .data-*）
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "traffic_modelora")
)

# 输入图像尺寸（必须与训练时一致）
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 1   # 灰度图单通道

# ============================================================
# 推理配置
# ============================================================
CONFIDENCE_THRESHOLD = 0.7   # 置信度阈值，低于此值标记 unreliable
TOP_K = 5                    # 返回 Top-K 概率（预留）

# ============================================================
# API 配置
# ============================================================
API_TITLE = "Traffic Sign Recognition API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "基于 CNN 的 GTSRB 交通标志识别系统后端接口"

# 允许的上传图片格式
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ============================================================
# 类别名称文件路径
# ============================================================
CLASS_NAMES_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "signnames.csv"
)
