"""
FastAPI 主程序
提供交通标志识别的后端接口：
- GET  /health    健康检查
- POST /predict   图片识别（支持多模型自动切换）
- GET  /classes   获取类别列表
- POST /feedback  用户反馈提交

多模型模式：
  - mode=batch          → TSRD Model（全图 resize）
  - mode=upload_roi     → TSRD-ROI Model（用户框选 ROI）
  - mode=camera_roi     → TSRD-ROI Model（摄像头 ROI）
  - mode 缺失/其他       → 默认走 TSRD Model（向后兼容）
"""
import json
import time
from typing import Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    MODEL_PATH_TSRD,
    MODEL_PATH_ROI,
    MODEL_NAME_TSRD,
    MODEL_NAME_ROI,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    CONFIDENCE_THRESHOLD,
    ALLOWED_EXTENSIONS,
    CLASS_NAMES_CSV,
)
from model_loader import ModelLoader
from preprocess import preprocess_image, preprocess_image_roi
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
    correct_class_id: int = Field(..., description="用户修正的类别 ID（-1 表示无正确类别）")
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
model_tsrd: ModelLoader = None      # TSRD Model（batch 模式）
model_tsrd_roi: ModelLoader = None  # TSRD-ROI Model（upload_roi / camera_roi 模式）
class_names: dict = {}

# mode -> (model_loader_var, model_name, use_roi_preprocess)
MODE_ROUTING = {
    "batch":       ("model_tsrd",    MODEL_NAME_TSRD, False),
    "upload_roi":  ("model_tsrd_roi", MODEL_NAME_ROI,  True),
    "camera_roi":  ("model_tsrd_roi", MODEL_NAME_ROI,  True),
}

# ============================================================
# 应用启动与关闭事件
# ============================================================
@app.on_event("startup")
def startup():
    """应用启动时：加载两个模型与类别名称"""
    global model_tsrd, model_tsrd_roi, class_names

    print("=" * 50)
    print("[STARTUP] Starting traffic sign recognition backend service...")

    # 加载类别名称
    class_names = load_class_names(CLASS_NAMES_CSV)
    print(f"[STARTUP] Class names loaded: {len(class_names)} classes")

    def _load_single(name, model_path):
        """加载单个模型并打印状态"""
        loader = ModelLoader(
            model_path=model_path,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            img_channels=IMG_CHANNELS,
        )
        try:
            loader.load_model()
            print(f"[STARTUP] {name} loaded successfully")
            return loader
        except FileNotFoundError as e:
            print(f"[STARTUP] {name} load failed: {e}")
            print(f"[STARTUP] {name} will be unavailable")
        except Exception as e:
            print(f"[STARTUP] Unknown error loading {name}: {e}")
        return None

    # 加载 TSRD Model（batch）
    model_tsrd = _load_single("TSRD Model", MODEL_PATH_TSRD)
    # 加载 TSRD-ROI Model（upload_roi / camera_roi）
    model_tsrd_roi = _load_single("TSRD-ROI Model", MODEL_PATH_ROI)

    # 初始化反馈目录与 CSV
    init_feedback_dir()

    total = sum(1 for m in [model_tsrd, model_tsrd_roi] if m is not None and m.loaded)
    print(f"[STARTUP] {total}/2 models loaded")
    print("=" * 50)


@app.on_event("shutdown")
def shutdown():
    """应用关闭时：释放 TF Session"""
    for name, loader in [("TSRD", model_tsrd), ("TSRD-ROI", model_tsrd_roi)]:
        if loader is not None:
            loader.close()


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

    返回服务状态和两个模型的加载状态。
    可通过 curl http://127.0.0.1:8000/health 测试。
    """
    tsrd_loaded = model_tsrd is not None and model_tsrd.loaded
    roi_loaded = model_tsrd_roi is not None and model_tsrd_roi.loaded

    # 当前使用的模型名称（前端兼容）
    active_name = MODEL_NAME_TSRD
    if tsrd_loaded:
        active_name = MODEL_NAME_TSRD
    elif roi_loaded:
        active_name = MODEL_NAME_ROI

    return {
        "status": "ok",
        "model_loaded": tsrd_loaded or roi_loaded,
        "model_name": active_name,
        "models": {
            "tsrd": {"loaded": tsrd_loaded, "name": MODEL_NAME_TSRD},
            "tsrd_roi": {"loaded": roi_loaded, "name": MODEL_NAME_ROI},
        },
    }


# ============================================================
# 接口：POST /predict
# ============================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("batch"),
    roi_x1: Optional[int] = Form(None),
    roi_y1: Optional[int] = Form(None),
    roi_x2: Optional[int] = Form(None),
    roi_y2: Optional[int] = Form(None),
):
    """
    交通标志识别接口（支持多模型自动切换）。

    根据 mode 参数自动选择模型和预处理方式：
      - mode=batch       → TSRD Model（全图 resize，默认）
      - mode=upload_roi  → TSRD-ROI Model（需提供 ROI 坐标）
      - mode=camera_roi  → TSRD-ROI Model（需提供 ROI 坐标）

    请求格式: multipart/form-data
      - file: 图片文件
      - mode: 识别模式（可选，默认 "batch"）
      - roi_x1, roi_y1, roi_x2, roi_y2: ROI 边界框（upload_roi / camera_roi 模式必需）

    支持格式: jpg, jpeg, png

    返回格式:
    {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "processing_time": float,
        "reliable": bool,
        "mode": str,           # 当前使用的模式
        "model_name": str,     # 当前使用的模型名称
    }
    """
    # ---- 1. 确定模式与选择模型 ----
    if mode not in MODE_ROUTING:
        mode = "batch"

    model_var_name, model_display_name, use_roi = MODE_ROUTING[mode]
    selected_model = model_tsrd if model_var_name == "model_tsrd" else model_tsrd_roi

    if selected_model is None or not selected_model.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"模型 {model_display_name} 未加载，请检查模型文件路径是否正确",
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

    # ---- 4. ROI 模式参数校验 ----
    if use_roi:
        if any(v is None for v in [roi_x1, roi_y1, roi_x2, roi_y2]):
            raise HTTPException(
                status_code=400,
                detail=f"模式 '{mode}' 需要提供 ROI 坐标 (roi_x1, roi_y1, roi_x2, roi_y2)",
            )

    # ---- 5. 图片预处理 ----
    try:
        if use_roi:
            processed = preprocess_image_roi(image_bytes, roi_x1, roi_y1, roi_x2, roi_y2)
        else:
            processed = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片预处理失败: {str(e)}")

    # ---- 6. 模型推理 ----
    start_time = time.time()
    try:
        proba = selected_model.predict(processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")
    elapsed = time.time() - start_time

    # ---- 7. 解析结果 ----
    proba = proba[0]

    # Top-1
    class_id = int(np.argmax(proba))
    confidence = float(proba[class_id])
    class_name = get_class_name(class_id, class_names)
    reliable = confidence >= CONFIDENCE_THRESHOLD

    # Top-5
    top5_indices = np.argsort(proba)[::-1][:5]
    top5 = [
        {
            "class_id": int(idx),
            "class_name": get_class_name(int(idx), class_names),
            "confidence": round(float(proba[idx]), 4),
        }
        for idx in top5_indices
    ]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "processing_time": round(elapsed, 4),
        "reliable": reliable,
        "mode": mode,
        "model_name": model_display_name,
        "top5": top5,
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
    # 当 correct_class_id 为 -1 时，表示"无正确类别"，跳过名称核对
    if data.correct_class_id >= 0:
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


# ============================================================
# 接口：GET /signs-data  交通标志大全数据
# ============================================================
SIGNS_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "交通标志大全"
SIGNS_JSON_PATH = SIGNS_DATA_DIR / "traffic_signs.json"
SIGNS_IMAGES_DIR = SIGNS_DATA_DIR / "images"

# TSRD 58 类名称列表（用于标注是否支持识别）
_TSRD_NAMES = {
    "限制速度5km/h", "限制速度15km/h", "限制速度30km/h", "限制速度40km/h",
    "限制速度50km/h", "限制速度60km/h", "限制速度70km/h", "限制速度80km/h",
    "禁止直行和向左转弯", "禁止直行和向右转弯", "禁止直行", "禁止向左转弯",
    "禁止向左向右转弯", "禁止向右转弯", "禁止超车", "禁止掉头",
    "禁止机动车驶入", "禁止鸣喇叭", "解除限制速度40km/h", "解除限制速度50km/h",
    "直行和向右转弯", "直行", "向左转弯", "向左和向右转弯", "向右转弯",
    "靠左侧车道行驶", "靠右侧车道行驶", "环岛行驶", "机动车行驶", "鸣喇叭",
    "非机动车行驶", "允许掉头", "注意合流", "注意信号灯", "注意危险",
    "注意行人", "注意非机动车", "注意儿童", "向右急弯路", "向左急弯路",
    "下陡坡", "上陡坡", "减速慢行", "右侧T形交叉", "左侧T形交叉",
    "村庄", "反向弯路", "无人看守铁路道口", "施工", "连续弯路",
    "有人看守铁路道口", "事故易发路段", "停车让行", "禁止通行",
    "禁止停车", "禁止驶入", "减速让行", "停车检查",
}


def _match_tsrd(sign_name: str) -> bool:
    """判断一个交通标志名称是否在 TSRD 58 类中（模糊匹配）"""
    if sign_name in _TSRD_NAMES:
        return True
    # 尝试去掉 "标志" 后缀再匹配
    cleaned = sign_name.replace("标志", "").replace("（", "(").replace("）", ")")
    if cleaned in _TSRD_NAMES:
        return True
    return False


print("[SETUP] Loading traffic signs data...")
if SIGNS_JSON_PATH.exists():
    with open(SIGNS_JSON_PATH, "r", encoding="utf-8") as f:
        _raw_signs = json.load(f)
    print(f"[SETUP] Loaded {len(_raw_signs)} signs from JSON")

    for s in _raw_signs:
        s["image_file"] = f"{s['category']}_{s['name']}.png"
        s["in_tsrd"] = _match_tsrd(s["name"])
    print(f"[SETUP] in_tsrd annotations added to {sum(1 for s in _raw_signs if s['in_tsrd'])}/{len(_raw_signs)} signs")
else:
    print(f"[SETUP] SIGNS JSON NOT FOUND at: {SIGNS_JSON_PATH}")

# 挂载交通标志图片目录
print(f"[SETUP] Checking images dir: {SIGNS_IMAGES_DIR}")
if SIGNS_IMAGES_DIR.exists():
    try:
        app.mount("/signs-images", StaticFiles(directory=str(SIGNS_IMAGES_DIR)), name="signs_images")
        print(f"[SETUP] Signs images mounted: {SIGNS_IMAGES_DIR}")
    except Exception as e:
        print(f"[SETUP] Failed to mount signs images: {e}")
else:
    print(f"[SETUP] Images dir NOT FOUND")

print("[SETUP] Registering /signs-data endpoint...")
@app.get("/signs-data")
async def get_signs_data():
    """返回交通标志大全数据"""
    if not SIGNS_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail="交通标志数据文件未找到")
    try:
        return _raw_signs
    except NameError:
        raise HTTPException(status_code=500, detail="交通标志数据未加载")
