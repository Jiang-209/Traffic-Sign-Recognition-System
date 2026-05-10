# 交通标志识别系统 - Vue 3 前端

基于 CNN 的 GTSRB 交通标志识别系统前端界面。

## 项目结构

```
frontend/
├── index.html              HTML 入口
├── package.json            依赖配置
├── vite.config.js          Vite 配置（含 API 代理）
├── README.md               本文件
└── src/
    ├── main.js             Vue 应用入口
    ├── App.vue             根组件（布局 + 状态管理）
    ├── style.css           全局样式（CSS 变量、主题）
    ├── api/
    │   └── predict.js      Axios 封装的 API 请求
    └── components/
        ├── HeaderBar.vue   顶部导航栏（标题 + 后端状态）
        ├── UploadPanel.vue 图片上传面板（预览 + 拖拽）
        ├── ResultCard.vue  识别结果卡片（置信度进度条）
        └── HistoryTable.vue 识别历史记录表格
```

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

前端运行在 `http://127.0.0.1:5173`

## 与后端联调

Vite 已配置代理：前端 `/api/*` 请求自动转发到 `http://127.0.0.1:8000`

所以请确保后端先启动：

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

然后启动前端即可。

## 构建生产版本

```bash
npm run build        # 输出到 dist/
npm run preview      # 预览构建产物
```

## 技术栈

- Vue 3 (Composition API + `<script setup>`)
- Vite 5
- Axios
- 原生 CSS（Flex / Grid / CSS 变量）

## 浏览器支持

Chrome / Edge / Firefox 最新版本
