<!--
  UploadPanel.vue - 图片上传面板
  支持点击上传、拖拽上传、图片预览
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
      <!-- 已选择图片 - 预览 -->
      <div v-else class="preview-container">
        <img :src="previewUrl" alt="预览图片" class="preview-image" />
        <!-- 换图遮罩 -->
        <div class="preview-overlay" @click.stop="triggerFileInput">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M23 4v6h-6"/><path d="M1 20v-6h6"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
          </svg>
          <span>更换图片</span>
        </div>
      </div>
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
        <!-- Loading 旋转图标 -->
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
import { ref } from 'vue'

const emit = defineEmits(['predict-start', 'predict-success', 'predict-error', 'reset', 'file-selected'])

// ===== 状态 =====
const fileInputRef = ref(null)
const previewUrl = ref(null)       // 图片预览 URL
const selectedFile = ref(null)     // 当前选中的文件
const isUploading = ref(false)     // 是否正在上传识别
const isDragover = ref(false)      // 拖拽悬停状态
const errorMsg = ref('')           // 错误消息

// 允许的文件类型
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

// ===== 方法 =====

/** 触发隐藏的文件选择器 */
function triggerFileInput() {
  if (isUploading.value) return
  fileInputRef.value?.click()
}

/** 处理文件选择 */
function handleFileSelect(e) {
  const file = e.target.files?.[0]
  if (file) processFile(file)
  // 清空 input 以允许重复选择同一文件
  e.target.value = ''
}

/** 处理拖拽上传 */
function handleDrop(e) {
  isDragover.value = false
  const file = e.dataTransfer.files?.[0]
  if (file) processFile(file)
}

/** 验证并处理文件 */
function processFile(file) {
  errorMsg.value = ''

  // 类型校验
  if (!ALLOWED_TYPES.includes(file.type)) {
    errorMsg.value = '仅支持 JPG、JPEG、PNG 格式的图片'
    return
  }

  // 大小校验（最大 20MB）
  if (file.size > 20 * 1024 * 1024) {
    errorMsg.value = '图片大小不能超过 20MB'
    return
  }

  selectedFile.value = file

  // 通知父组件文件已选择（用于反馈面板）
  emit('file-selected', file)

  // 生成预览 URL（释放旧 URL）
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
}

/** 开始识别 */
async function handlePredict() {
  if (!selectedFile.value || isUploading.value) return

  isUploading.value = true
  errorMsg.value = ''
  emit('predict-start')

  try {
    // 动态导入 API（避免循环依赖）
    const { predictImage } = await import('../api/predict.js')
    const result = await predictImage(selectedFile.value)
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

/** 重置选择 */
function handleReset() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  selectedFile.value = null
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

/* 上传区域 */
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

/* 预览 */
.preview-container {
  width: 100%;
  position: relative;
}

.preview-image {
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  display: block;
  background: #f8fafc;
}

.preview-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: #fff;
  font-size: 0.85rem;
  opacity: 0;
  transition: opacity var(--transition);
}

.preview-container:hover .preview-overlay {
  opacity: 1;
}

/* 按钮区 */
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

/* 加载旋转动画 */
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

/* 错误 Toast */
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
