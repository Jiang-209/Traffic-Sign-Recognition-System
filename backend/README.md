# 交通标志识别后端 (FastAPI + TensorFlow)

基于 CNN 的 GTSRB 交通标志识别系统后端接口。

## 项目结构

```
backend/
├── main.py           FastAPI 主程序（/health, /predict）
├── model_loader.py   模型加载与推理（TF1 checkpoint）
├── preprocess.py     图片预处理（RGB→灰度→归一化）
├── class_names.py    GTSRB 43 类名称映射
├── config.py         配置文件（模型路径、阈值等）
├── requirements.txt  依赖清单
└── README.md         本文件
```

## 环境要求

- Python 3.8+
- TensorFlow 2.x（兼容模式运行 TF1 checkpoint）

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动服务

默认会加载 `../traffic_modelora` 下的模型文件：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

如果模型在其他位置，通过环境变量指定：

```bash
# Windows CMD
set MODEL_PATH=E:\your\path\to\traffic_modelora
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Linux / Git Bash
MODEL_PATH=/path/to/traffic_modelora uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. 访问 Swagger 文档

浏览器打开: http://127.0.0.1:8000/docs

## 接口说明

### GET /health

健康检查，确认服务状态和模型是否加载。

```bash
curl http://127.0.0.1:8000/health
```

返回：
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

上传图片进行交通标志识别。

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@your_traffic_sign.jpg"
```

返回示例：
```json
{
  "class_id": 14,
  "class_name": "Stop",
  "confidence": 0.9821,
  "processing_time": 0.032,
  "reliable": true
}
```

- `reliable`: 当 confidence >= 0.7 时为 true
- `processing_time`: 推理耗时（秒）

## 配置说明

在 `config.py` 中可修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | ../traffic_modelora | 模型文件路径 |
| CONFIDENCE_THRESHOLD | 0.7 | 置信度阈值 |
| IMG_HEIGHT | 32 | 输入图像高度 |
| IMG_WIDTH | 32 | 输入图像宽度 |

## 注意事项

1. 模型必须与训练代码使用相同的 TF 版本保存
2. 图片预处理流程（灰度化、归一化方式）必须与训练时一致
3. 生产环境应将 CORS `allow_origins` 限制为前端实际域名
