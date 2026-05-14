<template>
  <div class="camera-page">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2 class="page-title">摄像头识别</h2>
      <span v-if="stream" class="status-badge" :class="statusBadgeClass">{{ statusBadgeText }}</span>
    </div>

    <div class="camera-layout">
      <!-- ===== 摄像头区域 ===== -->
      <div class="camera-section">
        <div class="video-container">
          <!-- 摄像头不可用 -->
          <div v-if="cameraError" class="camera-error">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <p>{{ cameraError }}</p>
            <button class="btn btn-primary" @click="startCamera">重试</button>
          </div>

          <!-- 未启动 -->
          <div v-else-if="!stream" class="camera-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M23 7l-7 5 7 5V7z"/>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            <p>点击下方按钮打开摄像头</p>
            <p class="hint">将交通标志对准画面中央的框内</p>
          </div>

          <!-- 摄像头画面 -->
          <div v-else class="video-wrapper" ref="videoWrapperRef">
            <video
              ref="videoRef"
              autoplay
              playsinline
              muted
              @loadedmetadata="onVideoReady"
            />

            <!-- ROI 聚焦框 -->
            <template v-if="videoReady">
              <div class="roi-overlay" :style="roiOverlayStyle" />
              <div class="roi-box" :style="roiBoxStyle">
                <div class="roi-corner roi-corner-nw" />
                <div class="roi-corner roi-corner-ne" />
                <div class="roi-corner roi-corner-sw" />
                <div class="roi-corner roi-corner-se" />
              </div>
            </template>

            <!-- 识别中指示 -->
            <div v-if="isRecognizing" class="rec-indicator">
              <span class="rec-dot" />
              识别中
            </div>
          </div>
        </div>

        <!-- 控制按钮 -->
        <div class="controls">
          <button v-if="!stream" class="btn btn-primary" @click="startCamera">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            打开摄像头
          </button>

          <template v-else>
            <button
              v-if="!isRecognizing"
              class="btn btn-primary"
              :disabled="!videoReady"
              @click="startRecognition"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
              </svg>
              开始识别
            </button>

            <button v-else class="btn btn-warning" @click="stopRecognition">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
              </svg>
              暂停识别
            </button>

            <button class="btn btn-outline btn-danger-text" @click="stopCamera">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
              关闭摄像头
            </button>
          </template>
        </div>
      </div>

      <!-- ===== 识别结果 ===== -->
      <div class="result-section">
        <div class="result-card">
          <div class="result-header">
            <h3 class="result-title">识别结果</h3>
            <span class="rec-count">{{ recognitionCount }} 次</span>
          </div>

          <!-- 等待状态 -->
          <div v-if="!lastResult && !isRecognizing && stream" class="result-placeholder">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            <p>等待开始识别...</p>
          </div>

          <div v-if="!stream && !cameraError" class="result-placeholder">
            <p>打开摄像头后将在此显示识别结果</p>
          </div>

          <!-- 识别中指示 -->
          <div v-if="isRecognizing && !lastResult" class="result-loading">
            <div class="spinner"></div>
            <span>首次识别中...</span>
          </div>

          <!-- 识别结果 -->
          <div v-if="lastResult" class="result-body">
            <!-- ROI 截取预览 -->
            <div v-if="lastRoiPreview" class="roi-preview-wrap">
              <img :src="lastRoiPreview" alt="ROI 截取区域" class="roi-preview-img" />
              <span class="roi-preview-label">截取图像</span>
            </div>

            <div class="result-class">{{ lastResult.class_name }}</div>

            <div class="result-metrics">
              <div class="metric">
                <span class="metric-label">置信度</span>
                <span class="metric-value" :class="confidenceClass">
                  {{ (lastResult.confidence * 100).toFixed(1) }}%
                </span>
              </div>
              <div class="metric">
                <span class="metric-label">可信度</span>
                <span class="metric-badge" :class="{ reliable: lastResult.reliable, unreliable: !lastResult.reliable }">
                  {{ lastResult.reliable ? '可信' : '存疑' }}
                </span>
              </div>
              <div class="metric">
                <span class="metric-label">耗时</span>
                <span class="metric-value">{{ lastResult.processing_time }}s</span>
              </div>
            </div>

            <div class="result-model">
              使用模型：<strong>{{ lastResult.model_name }}</strong>
            </div>
          </div>
        </div>

        <!-- 反馈面板（暂停后显示） -->
        <Transition name="slide-up">
          <FeedbackPanel
            v-if="lastResult && !isRecognizing && stream"
            :result="feedbackResult"
            :image-file="lastCapturedFile"
            @feedback-submitted="onFeedbackSubmitted"
          />
        </Transition>
      </div>
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
import { ref, computed, onUnmounted } from 'vue'
import { predictImage } from '../api/predict.js'
import FeedbackPanel from '../components/FeedbackPanel.vue'

// ===== 状态 =====
const videoRef = ref(null)
const videoWrapperRef = ref(null)
const stream = ref(null)
const videoReady = ref(false)
const cameraError = ref('')
const errorMsg = ref('')

const isRecognizing = ref(false)
const recognitionCount = ref(0)
const lastResult = ref(null)
const lastRoiPreview = ref(null)  // 最近一次截取的 ROI 预览 dataURL
const lastCapturedFile = ref(null) // 最近一次截取的原始文件（用于反馈）
const isProcessing = ref(false)

let recTimerId = null
const RECOGNITION_INTERVAL = 1500 // 1.5秒

// ROI 坐标（在视频原始分辨率下的像素坐标）
const roiIntrinsic = ref(null)

// ===== 计算属性 =====

const statusBadgeClass = computed(() => ({
  'badge-on': isRecognizing.value,
  'badge-idle': stream.value && !isRecognizing.value,
}))

const statusBadgeText = computed(() => {
  if (!stream.value) return ''
  return isRecognizing.value ? '识别中' : '已就绪'
})

/** ROI 叠加层的样式（暗色遮罩） */
const roiOverlayStyle = computed(() => {
  if (!roiIntrinsic.value || !videoRef.value) return {}
  const b = getRoiDisplayRect()
  if (!b) return {}
  return {
    left: b.left + 'px',
    top: b.top + 'px',
    width: b.width + 'px',
    height: b.height + 'px',
    boxShadow: `0 0 0 9999px rgba(0,0,0,0.35)`,
  }
})

/** ROI 边框样式 */
const roiBoxStyle = computed(() => {
  if (!roiIntrinsic.value || !videoRef.value) return {}
  const b = getRoiDisplayRect()
  if (!b) return {}
  return {
    left: b.left + 'px',
    top: b.top + 'px',
    width: b.width + 'px',
    height: b.height + 'px',
  }
})

const confidenceClass = computed(() => {
  if (!lastResult.value) return ''
  const c = lastResult.value.confidence
  if (c >= 0.9) return 'conf-high'
  if (c >= 0.7) return 'conf-mid'
  return 'conf-low'
})

/** FeedbackPanel 所需的 result 格式 */
const feedbackResult = computed(() => {
  if (!lastResult.value) return null
  return {
    class_id: lastResult.value.class_id,
    class_name: lastResult.value.class_name,
    confidence: lastResult.value.confidence,
    reliable: lastResult.value.reliable,
    fileName: `camera_${Date.now()}.jpg`,
    timestamp: new Date().toLocaleString('zh-CN'),
  }
})

function onFeedbackSubmitted(data) {
  console.log('[摄像头反馈]', data)
}

// ===== ROI 坐标计算 =====

/** 获取 ROI 在显示区域的尺寸（用于覆盖层定位） */
function getRoiDisplayRect() {
  const video = videoRef.value
  const wr = videoWrapperRef.value
  if (!video || !wr || !roiIntrinsic.value) return null

  const displayW = wr.offsetWidth
  const displayH = wr.offsetHeight
  const scaleX = displayW / video.videoWidth
  const scaleY = displayH / video.videoHeight

  const r = roiIntrinsic.value
  return {
    left: r.x1 * scaleX,
    top: r.y1 * scaleY,
    width: (r.x2 - r.x1) * scaleX,
    height: (r.y2 - r.y1) * scaleY,
  }
}

/** 根据视频原始分辨率计算 ROI（居中正方形，50% 边长） */
function calculateRoi() {
  const video = videoRef.value
  if (!video || !video.videoWidth) return null
  const size = Math.round(Math.min(video.videoWidth, video.videoHeight) * 0.5)
  const cx = Math.round(video.videoWidth / 2)
  const cy = Math.round(video.videoHeight / 2)
  const half = Math.round(size / 2)
  return { x1: cx - half, y1: cy - half, x2: cx + half, y2: cy + half }
}

// ===== 摄像头控制 =====

async function startCamera() {
  cameraError.value = ''
  errorMsg.value = ''

  if (!navigator.mediaDevices?.getUserMedia) {
    cameraError.value = '您的浏览器不支持摄像头访问'
    return
  }

  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'environment',
      },
      audio: false,
    })
    stream.value = s
    videoReady.value = false

    // 等待下一帧确保标签已挂载
    await nextTick()
    if (videoRef.value) {
      videoRef.value.srcObject = s
    }
  } catch (e) {
    if (e.name === 'NotAllowedError') {
      cameraError.value = '摄像头权限被拒绝，请在浏览器设置中允许摄像头访问'
    } else if (e.name === 'NotFoundError') {
      cameraError.value = '未检测到摄像头设备'
    } else if (e.name === 'NotReadableError') {
      cameraError.value = '摄像头被其他应用占用，请关闭后重试'
    } else {
      cameraError.value = `摄像头启动失败: ${e.message}`
    }
  }
}

function onVideoReady() {
  const video = videoRef.value
  if (!video) return
  roiIntrinsic.value = calculateRoi()
  videoReady.value = true
}

function stopCamera() {
  stopRecognition()
  if (stream.value) {
    stream.value.getTracks().forEach(t => t.stop())
    stream.value = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  videoReady.value = false
  roiIntrinsic.value = null
  lastResult.value = null
  lastRoiPreview.value = null
  lastCapturedFile.value = null
  recognitionCount.value = 0
}

// ===== 识别控制 =====

function startRecognition() {
  if (isRecognizing.value || !stream.value || !videoReady.value) return
  isRecognizing.value = true
  lastResult.value = null
  scheduleNextRecognition()
}

function stopRecognition() {
  isRecognizing.value = false
  isProcessing.value = false
  if (recTimerId) {
    clearTimeout(recTimerId)
    recTimerId = null
  }
}

async function scheduleNextRecognition() {
  if (!isRecognizing.value) return
  recTimerId = setTimeout(doRecognition, RECOGNITION_INTERVAL)
}

async function doRecognition() {
  if (!isRecognizing.value || isProcessing.value || !stream.value) return

  isProcessing.value = true
  try {
    // 截取 ROI 预览（用于显示）
    lastRoiPreview.value = captureRoiPreview()

    const file = await captureFrame()
    lastCapturedFile.value = file
    const result = await predictImage(file, 'camera_roi', roiIntrinsic.value)
    lastResult.value = result
    recognitionCount.value++
  } catch (e) {
    errorMsg.value = e.response?.data?.detail || e.message || '识别请求失败'
  } finally {
    isProcessing.value = false
    scheduleNextRecognition()
  }
}

/** 截取 ROI 区域作为预览缩略图（dataURL） */
function captureRoiPreview() {
  const video = videoRef.value
  const roi = roiIntrinsic.value
  if (!video || !roi) return null
  const roiW = roi.x2 - roi.x1
  const roiH = roi.y2 - roi.y1
  if (roiW <= 0 || roiH <= 0) return null
  const canvas = document.createElement('canvas')
  canvas.width = roiW
  canvas.height = roiH
  const ctx = canvas.getContext('2d')
  ctx.drawImage(video, roi.x1, roi.y1, roiW, roiH, 0, 0, roiW, roiH)
  return canvas.toDataURL('image/jpeg', 0.85)
}

/** 捕获当前摄像头画面为 JPEG 文件 */
function captureFrame() {
  return new Promise((resolve, reject) => {
    try {
      const video = videoRef.value
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0)
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' }))
        } else {
          reject(new Error('无法捕获摄像头画面'))
        }
      }, 'image/jpeg', 0.85)
    } catch (e) {
      reject(e)
    }
  })
}

// ===== 生命周期 =====

onUnmounted(() => {
  stopCamera()
})

// 辅助
function nextTick() {
  return new Promise(resolve => setTimeout(resolve, 50))
}
</script>

<style scoped>
.camera-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ===== 页面标题 ===== */
.page-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--color-text);
}

.status-badge {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 3px 12px;
  border-radius: 12px;
}

.badge-on {
  background: #dcfce7;
  color: #16a34a;
}

.badge-idle {
  background: var(--color-primary-light);
  color: var(--color-primary);
}

/* ===== 摄像头区域 ===== */
.camera-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.video-container {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
  min-height: 280px;
  display: flex;
  align-items: center;
  justify-content: center;
  max-width: 560px;
  margin: 0 auto;
  width: 100%;
}

/* 错误状态 */
.camera-error {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-danger);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.camera-error p {
  font-size: 0.9rem;
  max-width: 360px;
}

/* 未启动 */
.camera-placeholder {
  text-align: center;
  padding: 60px 20px;
  color: var(--color-text-secondary);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.camera-placeholder svg {
  opacity: 0.4;
}

.camera-placeholder .hint {
  font-size: 0.8rem;
  color: #94a3b8;
}

/* 视频包装 */
.video-wrapper {
  position: relative;
  width: 100%;
  line-height: 0;
  background: #000;
}

.video-wrapper video {
  width: 100%;
  display: block;
  transform: scaleX(-1);  /* 镜像翻转，方便用户调整位置 */
}

/* ROI 暗色遮罩 */
.roi-overlay {
  position: absolute;
  pointer-events: none;
  z-index: 5;
  border-radius: 4px;
}

/* ROI 边框 */
.roi-box {
  position: absolute;
  pointer-events: none;
  z-index: 6;
}

.roi-corner {
  position: absolute;
  width: 16px;
  height: 16px;
  border-color: #4ade80;
  border-style: solid;
}

.roi-corner-nw { top: 0; left: 0; border-width: 3px 0 0 3px; }
.roi-corner-ne { top: 0; right: 0; border-width: 3px 3px 0 0; }
.roi-corner-sw { bottom: 0; left: 0; border-width: 0 0 3px 3px; }
.roi-corner-se { bottom: 0; right: 0; border-width: 0 3px 3px 0; }

/* 识别中指示 */
.rec-indicator {
  position: absolute;
  top: 12px;
  left: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  color: #4ade80;
  background: rgba(0, 0, 0, 0.55);
  padding: 4px 12px;
  border-radius: 6px;
  z-index: 10;
  pointer-events: none;
}

.rec-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #4ade80;
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* ===== 控制按钮 ===== */
.controls {
  display: flex;
  gap: 10px;
}

/* ===== 识别结果 ===== */
.result-section {
  width: 100%;
}

.result-card {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 20px 24px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.result-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text);
}

.rec-count {
  font-size: 0.78rem;
  color: var(--color-text-secondary);
  background: #f1f5f9;
  padding: 2px 10px;
  border-radius: 10px;
}

.result-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 24px;
  color: var(--color-text-secondary);
  font-size: 0.9rem;
}

.result-placeholder svg {
  opacity: 0.35;
}

.result-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 20px;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
}

.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 结果内容 */
.result-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.roi-preview-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.roi-preview-img {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  border: 2px solid var(--color-border);
  background: #f1f5f9;
}

.roi-preview-label {
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  opacity: 0.7;
}

.result-class {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text);
  text-align: center;
  padding: 12px;
  background: var(--color-primary-light);
  border-radius: 8px;
}

.result-metrics {
  display: flex;
  gap: 12px;
}

.metric {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 10px;
  background: #f8fafc;
  border-radius: 8px;
}

.metric-label {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-value {
  font-size: 1.1rem;
  font-weight: 700;
}

.metric-badge {
  font-size: 0.9rem;
  font-weight: 700;
  padding: 2px 12px;
  border-radius: 6px;
}

.metric-badge.reliable {
  background: #dcfce7;
  color: #16a34a;
}

.metric-badge.unreliable {
  background: #fff7ed;
  color: #ea580c;
}

.conf-high { color: #16a34a; }
.conf-mid  { color: var(--color-primary); }
.conf-low  { color: #ea580c; }

.result-model {
  font-size: 0.82rem;
  color: var(--color-text-secondary);
  text-align: center;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
}

.result-model strong {
  color: var(--color-text);
}

/* ===== 按钮 ===== */
.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 0.88rem;
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

.btn-warning {
  background: #ea580c;
  color: #fff;
}

.btn-warning:hover {
  background: #d4520a;
  box-shadow: 0 4px 12px rgba(234, 88, 12, 0.3);
}

.btn-outline {
  background: var(--color-card);
  border: 1.5px solid var(--color-border);
  color: var(--color-text-secondary);
}

.btn-outline:hover {
  border-color: var(--color-text-secondary);
  color: var(--color-text);
}

.btn-danger-text {
  color: var(--color-danger);
}

.btn-danger-text:hover {
  border-color: var(--color-danger);
  color: var(--color-danger);
  background: #fef2f2;
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
