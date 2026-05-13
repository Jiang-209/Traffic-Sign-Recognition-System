"""
FastAPI 主程序
提供交通标志识别的后端接口：
- GET  /health    健康检查
- POST /predict   图片识别
- GET  /classes   获取类别列表
- POST /feedback  用户反馈提交
"""
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    MODEL_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    CONFIDENCE_THRESHOLD,
    ALLOWED_EXTENSIONS,
    CLASS_NAMES_CSV,
)
from model_loader import ModelLoader
from preprocess import preprocess_image
from class_names import load_class_names, get_class_name
from feedback_handler import init_feedback_dir, save_feedback


# ============================================================
# Pydantic 数据模型
# ============================================================
class FeedbackData(BaseModel):
    """用户反馈数据结构"""
    image_name: str = Field(..., description="原始图片文件名")
    predicted_class_id: int = Field(..., ge=0, le=57, description="模型预测的类别 ID")
    predicted_class_name: str = Field(..., description="模型预测的类别名称")
    correct_class_id: int = Field(..., ge=0, le=57, description="用户修正的类别 ID")
    correct_class_name: str = Field(..., description="用户修正的类别名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="模型预测置信度")
    remark: str = Field(default="", description="用户备注")

# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# 配置 CORS，允许后续 Vue 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # 生产环境应限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 全局变量（应用启动时初始化）
# ============================================================
model: ModelLoader = None
class_names: dict = {}

# ============================================================
# 应用启动与关闭事件
# ============================================================
@app.on_event("startup")
def startup():
    """应用启动时：加载模型与类别名称"""
    global model, class_names

    print("=" * 50)
    print("[STARTUP] Starting traffic sign recognition backend service...")

    # 加载类别名称
    class_names = load_class_names(CLASS_NAMES_CSV)
    print(f"[STARTUP] Class names loaded: {len(class_names)} classes")

    # 加载模型
    model = ModelLoader(
        model_path=MODEL_PATH,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_channels=IMG_CHANNELS,
    )
    try:
        model.load_model()
    except FileNotFoundError as e:
        print(f"[STARTUP] Model load failed: {e}")
        print("[STARTUP] Service can still start, but /predict will be unavailable")
    except Exception as e:
        print(f"[STARTUP] Unknown error during model loading: {e}")

    # 初始化反馈目录与 CSV
    init_feedback_dir()

    print("=" * 50)


@app.on_event("shutdown")
def shutdown():
    """应用关闭时：释放 TF Session"""
    global model
    if model is not None:
        model.close()


# ============================================================
# 辅助函数
# ============================================================
def validate_file_extension(filename: str) -> str:
    """
    检查文件扩展名是否允许。
    返回小写的扩展名，或抛出 HTTPException。
    """
    if "." not in filename:
        raise HTTPException(
            status_code=400,
            detail="文件名缺少扩展名，仅支持 jpg / jpeg / png 格式",
        )

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '.{ext}'，仅允许 {ALLOWED_EXTENSIONS}",
        )
    return ext


# ============================================================
# 接口：GET /health
# ============================================================
@app.get("/health")
async def health_check():
    """
    健康检查接口。

    返回服务状态和模型加载状态。
    可通过 curl http://127.0.0.1:8000/health 测试。
    """
    return {
        "status": "ok",
        "model_loaded": model is not None and model.loaded,
    }


# ============================================================
# 接口：POST /predict
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    交通标志识别接口。

    上传一张交通标志图片，返回识别结果。

    请求格式: multipart/form-data，字段名为 "file"
    支持格式: jpg, jpeg, png

    返回格式:
    {
        "class_id": int,        # 预测类别 ID (0-57)
        "class_name": str,      # 类别中文名
        "confidence": float,    # 置信度 (0-1)
        "processing_time": float, # 推理耗时（秒）
        "reliable": bool        # 是否可靠 (confidence >= 阈值)
    }
    """
    # ---- 1. 检查模型是否就绪 ----
    if model is None or not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，服务暂不可用。请检查模型文件路径是否正确。",
        )

    # ---- 2. 验证文件类型 ----
    validate_file_extension(file.filename)

    # ---- 3. 读取上传文件内容 ----
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取上传文件")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # ---- 4. 图片预处理 ----
    try:
        processed = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片预处理失败: {str(e)}")

    # ---- 5. 模型推理 ----
    start_time = time.time()
    try:
        proba = model.predict(processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")
    elapsed = time.time() - start_time

    # ---- 6. 解析结果 ----
    proba = proba[0]                        # shape (58,)
    class_id = int(np.argmax(proba))        # 最高概率的类别索引
    confidence = float(proba[class_id])     # 对应的置信度
    class_name = get_class_name(class_id, class_names)
    reliable = confidence >= CONFIDENCE_THRESHOLD

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "processing_time": round(elapsed, 4),
        "reliable": reliable,
    }


# ============================================================
# 接口：GET /classes
# ============================================================
@app.get("/classes")
async def get_classes():
    """
    获取所有 TSRD 58 类交通标志名称列表。

    返回格式:
    [
        { "class_id": 0, "class_name": "Speed limit (20km/h)" },
        ...
    ]
    """
    return [
        {"class_id": cid, "class_name": cname}
        for cid, cname in sorted(class_names.items())
    ]


# ============================================================
# 接口：POST /feedback
# ============================================================
@app.post("/feedback")
async def submit_feedback(
    feedback_data: str = Form(
        ...,
        description="JSON 字符串，包含 FeedbackData 所有字段",
    ),
    image: UploadFile = File(
        default=None,
        description="原始图片文件（可选）",
    ),
):
    """
    用户反馈接口。

    当模型识别错误时，用户可选择正确类别并提交反馈。
    反馈数据保存到 feedback/feedback.csv，图片保存到 feedback/images/。

    请求格式: multipart/form-data
      - feedback_data: JSON 字符串
      - image: 图片文件（可选）
    """
    # ---- 1. 解析 JSON 数据 ----
    try:
        data_dict = json.loads(feedback_data)
        data = FeedbackData(**data_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="feedback_data 不是有效的 JSON 字符串")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"反馈数据格式错误: {str(e)}")

    # ---- 2. 核对正确类别的名称 ----
    expected_name = get_class_name(data.correct_class_id, class_names)
    if data.correct_class_name != expected_name:
        # 以前端传的为准，但打印一条警告
        print(f"[WARN] Class name mismatch: corrected_id={data.correct_class_id}")

    # ---- 3. 读取图片内容 ----
    image_bytes = None
    if image is not None:
        try:
            image_bytes = await image.read()
        except Exception:
            print("[WARN] Failed to read feedback image, saving CSV record only")

    # ---- 4. 保存反馈 ----
    try:
        saved_path = save_feedback(
            image_name=data.image_name,
            predicted_class_id=data.predicted_class_id,
            predicted_class_name=data.predicted_class_name,
            correct_class_id=data.correct_class_id,
            correct_class_name=data.correct_class_name,
            confidence=data.confidence,
            remark=data.remark,
            image_bytes=image_bytes,
        )
        print(f"[FEEDBACK] Saved: "
              f"pred_id={data.predicted_class_id} -> correct_id={data.correct_class_id}, "
              f"image={'saved' if saved_path else 'not saved'}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈保存失败: {str(e)}")

    return {
        "success": True,
        "message": "Feedback submitted. Thank you!",
    }
