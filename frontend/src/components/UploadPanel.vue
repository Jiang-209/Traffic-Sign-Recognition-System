<!--
  UploadPanel.vue - 图片上传面板
  支持点击上传、拖拽上传、图片预览、ROI 框选识别
-->
<template>
  <div class="upload-panel">
    <h3 class="panel-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      上传图片
    </h3>

    <!-- 拖拽上传区域 -->
    <div
      class="upload-zone"
      :class="{ 'has-image': previewUrl, 'is-dragover': isDragover }"
      @click="triggerFileInput"
      @dragover.prevent="isDragover = true"
      @dragleave.prevent="isDragover = false"
      @drop.prevent="handleDrop"
    >
      <!-- 未选择图片 -->
      <div v-if="!previewUrl" class="upload-placeholder">
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
        <p class="upload-hint">点击或拖拽图片到此处</p>
        <p class="upload-limit">支持 JPG、JPEG、PNG 格式</p>
      </div>

      <!-- 已选择图片 - 预览 + ROI 框选 -->
      <div v-else class="preview-container-inner" @click.stop>
        <img
          ref="previewImgRef"
          :src="previewUrl"
          alt="预览图片"
          class="preview-image"
          @load="onImageLoaded"
          draggable="false"
        />

        <!-- ROI 交互层 -->
        <div
          class="roi-layer"
          @mousedown.prevent="onRoiMouseDown"
          @mousemove.prevent="onRoiMouseMove"
          @mouseup.prevent="onRoiMouseUp"
          @mouseleave="onRoiMouseUp"
        >
          <!-- ROI 选框 -->
          <div v-if="roiRect.display" class="roi-box" :style="roiBoxStyle">
            <svg class="roi-clear-icon" viewBox="0 0 20 20" @mousedown.stop="clearRoi">
              <circle cx="10" cy="10" r="8" fill="#dc2626" opacity="0.9"/>
              <line x1="7" y1="7" x2="13" y2="13" stroke="#fff" stroke-width="2"/>
              <line x1="13" y1="7" x2="7" y2="13" stroke="#fff" stroke-width="2"/>
            </svg>
          </div>

          <div v-if="roiRect.display" class="roi-handle roi-handle-nw" />
          <div v-if="roiRect.display" class="roi-handle roi-handle-ne" />
          <div v-if="roiRect.display" class="roi-handle roi-handle-sw" />
          <div v-if="roiRect.display" class="roi-handle roi-handle-se" />

        </div>

        <!-- 换图按钮（替代原来的 hover 遮罩） -->
        <div class="reload-btn" title="重新选择图片" @click.stop="triggerFileInput">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M23 4v6h-6"/><path d="M1 20v-6h6"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
          </svg>
        </div>
      </div>
    </div>

    <!-- 框选提示 -->
    <div v-if="isDrawing" class="roi-draw-hint">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/>
      </svg>
      <span>松开鼠标完成框选</span>
    </div>

    <!-- ROI 状态提示 -->
    <div v-if="roiRect.display && !isDrawing" class="roi-status">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/>
      </svg>
      <span>已框选 ROI 区域（{{ roiRect.w_orig }}×{{ roiRect.h_orig }}px）</span>
      <button class="roi-clear-btn" @click="clearRoi">清除</button>
    </div>

    <!-- 隐藏的文件输入 -->
    <input
      ref="fileInputRef"
      type="file"
      accept=".jpg,.jpeg,.png"
      style="display: none"
      @change="handleFileSelect"
    />

    <!-- 操作按钮 -->
    <div class="upload-actions">
      <button class="btn btn-primary" :disabled="!previewUrl || isUploading" @click="handlePredict">
        <span v-if="isUploading" class="btn-spinner"></span>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        {{ isUploading ? '识别中...' : '开始识别' }}
      </button>

      <button class="btn btn-outline" :disabled="!previewUrl || isUploading" @click="handleReset">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="1 4 1 10 7 10"/>
          <path d="M3.51 15a9 9 0 102.13-9.36L1 10"/>
        </svg>
        重新选择
      </button>
    </div>

    <!-- 模式提示 -->
    <div v-if="previewUrl && roiRect.display" class="mode-hint mode-roi">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
      </svg>
      将使用 <strong>TSRD-ROI Model</strong> 识别框选区域
    </div>
    <div v-else-if="previewUrl" class="mode-hint mode-batch">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
      </svg>
      未框选 ROI，将使用 <strong>TSRD Model</strong> 识别整张图片
      <span class="mode-hint-action">（在图片上拖拽框选可启用 ROI 识别）</span>
    </div>

    <!-- 错误提示 -->
    <Transition name="fade">
      <div v-if="errorMsg" class="error-toast" @click="errorMsg = ''">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        {{ errorMsg }}
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['predict-start', 'predict-success', 'predict-error', 'reset', 'file-selected'])

// ===== 状态 =====
const fileInputRef = ref(null)
const previewUrl = ref(null)
const selectedFile = ref(null)
const isUploading = ref(false)
const isDragover = ref(false)
const errorMsg = ref('')

// ROI 相关状态
const previewImgRef = ref(null)
const imageLoaded = ref(false)
const naturalSize = ref({ w: 0, h: 0 })
const isDrawing = ref(false)
const isDragging = ref(false)
const roiStart = ref({ x: 0, y: 0 })
const roiCurrent = ref({ x: 0, y: 0 })
const roiFinal = ref(null)   // { x1, y1, x2, y2 } in original image coords

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

// ===== ROI 计算 =====

/** 获取图片在 display 中的实际渲染区域（处理 object-fit: contain） */
function getImageRenderInfo() {
  const img = previewImgRef.value
  if (!img) return null
  const rect = img.getBoundingClientRect()
  const natW = img.naturalWidth
  const natH = img.naturalHeight
  const elW = rect.width
  const elH = rect.height
  const imgAspect = natW / natH
  const elAspect = elW / elH

  let renderW, renderH, offsetX, offsetY
  if (elAspect > imgAspect) {
    renderH = elH
    renderW = renderH * imgAspect
    offsetX = (elW - renderW) / 2
    offsetY = 0
  } else {
    renderW = elW
    renderH = renderW / imgAspect
    offsetX = 0
    offsetY = (elH - renderH) / 2
  }
  return { rect, renderW, renderH, offsetX, offsetY, natW, natH }
}

/** 将鼠标 clientX/Y 转换到原始图像坐标 */
function clientToOriginal(cx, cy) {
  const info = getImageRenderInfo()
  if (!info) return null
  const { rect, renderW, renderH, offsetX, offsetY, natW, natH } = info
  const relX = cx - rect.left - offsetX
  const relY = cy - rect.top - offsetY
  const x = Math.round((relX / renderW) * natW)
  const y = Math.round((relY / renderH) * natH)
  return { x: Math.max(0, Math.min(natW - 1, x)),
           y: Math.max(0, Math.min(natH - 1, y)) }
}

/** 将原始图像坐标转换到父容器内的 display 坐标 */
function originalToDisplay(ox, oy) {
  const info = getImageRenderInfo()
  if (!info) return { x: 0, y: 0 }
  const { renderW, renderH, offsetX, offsetY, natW, natH } = info
  return {
    x: offsetX + (ox / natW) * renderW,
    y: offsetY + (oy / natH) * renderH,
  }
}

// ===== ROI 选框状态 =====

/** 当前 ROI 的 display 坐标（用于渲染选框） */
const roiRect = computed(() => {
  if (isDrawing.value) {
    return buildRoiRect(roiStart.value, roiCurrent.value)
  }
  if (roiFinal.value) {
    return buildRoiRect(
      { x: roiFinal.value.x1, y: roiFinal.value.y1, isOriginal: true },
      { x: roiFinal.value.x2, y: roiFinal.value.y2, isOriginal: true },
    )
  }
  return { display: false }
})

function buildRoiRect(p1, p2) {
  // 转换到原始坐标
  const a = p1.isOriginal ? p1 : clientToOriginal(p1.x, p1.y)
  const b = p2.isOriginal ? p2 : clientToOriginal(p2.x, p2.y)
  if (!a || !b) return { display: false }

  const x1 = Math.min(a.x, b.x)
  const y1 = Math.min(a.y, b.y)
  const x2 = Math.max(a.x, b.x)
  const y2 = Math.max(a.y, b.y)

  // 转到 display 坐标
  const d1 = originalToDisplay(x1, y1)
  const d2 = originalToDisplay(x2, y2)

  return {
    display: true,
    left: d1.x,
    top: d1.y,
    w: d2.x - d1.x,
    h: d2.y - d1.y,
    w_orig: x2 - x1,
    h_orig: y2 - y1,
  }
}

const roiBoxStyle = computed(() => {
  const r = roiRect.value
  if (!r || !r.display) return {}
  return {
    left: r.left + 'px',
    top: r.top + 'px',
    width: r.w + 'px',
    height: r.h + 'px',
  }
})

// ===== 图片加载 =====

function onImageLoaded() {
  const img = previewImgRef.value
  if (img) {
    naturalSize.value = { w: img.naturalWidth, h: img.naturalHeight }
    imageLoaded.value = true
  }
}

function processFile(file) {
  errorMsg.value = ''
  if (!ALLOWED_TYPES.includes(file.type)) {
    errorMsg.value = '仅支持 JPG、JPEG、PNG 格式的图片'
    return
  }
  if (file.size > 20 * 1024 * 1024) {
    errorMsg.value = '图片大小不能超过 20MB'
    return
  }

  selectedFile.value = file
  roiFinal.value = null
  imageLoaded.value = false
  emit('file-selected', file)

  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
}

// ===== ROI 鼠标事件 =====

function onRoiMouseDown(e) {
  if (e.button !== 0) return
  isDrawing.value = true
  isDragging.value = false
  roiStart.value = { x: e.clientX, y: e.clientY }
  roiCurrent.value = { x: e.clientX, y: e.clientY }
}

function onRoiMouseMove(e) {
  if (!isDrawing.value) return
  isDragging.value = true
  roiCurrent.value = { x: e.clientX, y: e.clientY }
}

function onRoiMouseUp() {
  if (!isDrawing.value) return
  isDrawing.value = false

  if (isDragging.value) {
    const a = clientToOriginal(roiStart.value.x, roiStart.value.y)
    const b = clientToOriginal(roiCurrent.value.x, roiCurrent.value.y)
    if (a && b) {
      const x1 = Math.min(a.x, b.x), y1 = Math.min(a.y, b.y)
      const x2 = Math.max(a.x, b.x), y2 = Math.max(a.y, b.y)
      if (x2 - x1 >= 5 && y2 - y1 >= 5) {
        roiFinal.value = { x1, y1, x2, y2 }
      }
    }
  }
  isDragging.value = false
}

function clearRoi() {
  roiFinal.value = null
}

// ===== 文件选择 =====

function triggerFileInput() {
  if (isUploading.value) return
  fileInputRef.value?.click()
}

function handleFileSelect(e) {
  const file = e.target.files?.[0]
  if (file) processFile(file)
  e.target.value = ''
}

function handleDrop(e) {
  isDragover.value = false
  const file = e.dataTransfer.files?.[0]
  if (file) processFile(file)
}

// ===== 识别 =====

async function handlePredict() {
  if (!selectedFile.value || isUploading.value) return

  isUploading.value = true
  errorMsg.value = ''
  emit('predict-start')

  try {
    const { predictImage } = await import('../api/predict.js')

    let mode = 'batch'
    let roiBox = null

    if (roiFinal.value) {
      mode = 'upload_roi'
      roiBox = {
        x1: roiFinal.value.x1,
        y1: roiFinal.value.y1,
        x2: roiFinal.value.x2,
        y2: roiFinal.value.y2,
      }
    }

    const result = await predictImage(selectedFile.value, mode, roiBox)
    emit('predict-success', {
      ...result,
      fileName: selectedFile.value.name,
      timestamp: new Date().toLocaleString('zh-CN'),
    })
  } catch (err) {
    const msg = err.response?.data?.detail || err.message || '网络请求失败，请检查后端服务是否启动'
    errorMsg.value = msg
    emit('predict-error', msg)
  } finally {
    isUploading.value = false
  }
}

function handleReset() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  selectedFile.value = null
  imageLoaded.value = false
  roiFinal.value = null
  errorMsg.value = ''
  emit('file-selected', null)
  emit('reset')
}
</script>

<style scoped>
.upload-panel {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.panel-title {
  font-size: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
}

/* ===== 上传区域 ===== */
.upload-zone {
  border: 2px dashed var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all var(--transition);
  position: relative;
  min-height: 260px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.upload-zone:hover {
  border-color: var(--color-primary);
  background: var(--color-primary-light);
}

.upload-zone.is-dragover {
  border-color: var(--color-primary);
  background: var(--color-primary-light);
  transform: scale(1.01);
}

.upload-zone.has-image {
  border-style: solid;
  border-color: var(--color-border);
  padding: 0;
}

.upload-zone.has-image:hover {
  border-color: var(--color-primary);
}

/* 占位提示 */
.upload-placeholder {
  text-align: center;
  padding: 40px 20px;
}

.upload-icon {
  width: 56px;
  height: 56px;
  margin: 0 auto 12px;
  color: #94a3b8;
  display: block;
}

.upload-hint {
  font-size: 0.95rem;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.upload-limit {
  font-size: 0.78rem;
  color: #94a3b8;
}

/* ===== 图片预览容器 ===== */
.preview-container-inner {
  width: 100%;
  position: relative;
  line-height: 0;
}

.preview-image {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  display: block;
  background: #f8fafc;
  user-select: none;
  -webkit-user-drag: none;
}

/* ===== ROI 交互层 ===== */
.roi-layer {
  position: absolute;
  inset: 0;
  cursor: crosshair;
  z-index: 5;
}

/* ROI 选框 */
.roi-box {
  position: absolute;
  border: 2px dashed #2563eb;
  background: rgba(37, 99, 235, 0.08);
  pointer-events: none;
  z-index: 6;
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.15);
}

.roi-clear-icon {
  position: absolute;
  top: -10px;
  right: -10px;
  width: 20px;
  height: 20px;
  cursor: pointer;
  pointer-events: auto;
  z-index: 10;
  opacity: 0.8;
  transition: opacity var(--transition);
}

.roi-clear-icon:hover {
  opacity: 1;
}

/* ROI 角标 */
.roi-handle {
  position: absolute;
  width: 10px;
  height: 10px;
  border: 2px solid #2563eb;
  background: #fff;
  border-radius: 2px;
  z-index: 7;
  pointer-events: none;
}

.roi-handle-nw { top: -5px; left: -5px; }
.roi-handle-ne { top: -5px; right: -5px; }
.roi-handle-sw { bottom: -5px; left: -5px; }
.roi-handle-se { bottom: -5px; right: -5px; }

.roi-draw-hint {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  background: #f8fafc;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  color: var(--color-text-secondary);
  font-size: 0.82rem;
}

/* ===== 换图按钮 ===== */
.reload-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 8px;
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--transition);
  z-index: 10;
}

.preview-container-inner:hover .reload-btn {
  opacity: 1;
}

.reload-btn:hover {
  background: rgba(0, 0, 0, 0.7);
}

/* ===== ROI 状态栏 ===== */
.roi-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  color: var(--color-primary);
  background: var(--color-primary-light);
  padding: 6px 12px;
  border-radius: 6px;
  margin-top: -8px;
}

.roi-clear-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--color-danger);
  font-size: 0.78rem;
  cursor: pointer;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  transition: background var(--transition);
}

.roi-clear-btn:hover {
  background: rgba(220, 38, 38, 0.1);
}

/* ===== 模式提示 ===== */
.mode-hint {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  padding: 6px 12px;
  border-radius: 6px;
  margin-top: -8px;
}

.mode-roi {
  background: #ecfdf5;
  color: #065f46;
}

.mode-roi strong {
  color: #059669;
}

.mode-batch {
  background: #fffbeb;
  color: #92400e;
}

.mode-batch strong {
  color: #d97706;
}

.mode-hint-action {
  color: #a16207;
  opacity: 0.75;
}

/* ===== 按钮区 ===== */
.upload-actions {
  display: flex;
  gap: 12px;
}

.btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 11px 20px;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all var(--transition);
  white-space: nowrap;
}

.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--color-primary);
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  background: var(--color-primary-dark);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

.btn-outline {
  background: var(--color-card);
  color: var(--color-text-secondary);
  border: 1.5px solid var(--color-border);
}

.btn-outline:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
  background: var(--color-primary-light);
}

.btn-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ===== 错误提示 ===== */
.error-toast {
  background: #fef2f2;
  color: var(--color-danger);
  border: 1px solid #fecaca;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
</style>
