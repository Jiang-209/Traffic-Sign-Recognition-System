<template>
  <div class="page-content">
    <div class="page-header">
      <h2 class="page-title">批量识别</h2>
      <p class="page-desc">一次上传多张图片，系统自动批量识别并返回结果</p>
    </div>

    <div class="upload-section">
      <div
        class="upload-zone"
        :class="{ 'is-dragover': isDragover }"
        @click="triggerFileInput"
        @dragover.prevent="isDragover = true"
        @dragleave.prevent="isDragover = false"
        @drop.prevent="handleDrop"
      >
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <p class="upload-hint">点击或拖拽图片到此处</p>
        <p class="upload-limit">支持 JPG、JPEG、PNG 格式，可多选</p>
        <span class="model-hint">使用 TSRD Model</span>
      </div>

      <div v-if="files.length > 0" class="file-preview-list">
        <div
          v-for="(item, index) in files"
          :key="item.id"
          class="file-preview-item"
          :class="{
            'is-processing': item.status === 'processing',
            'is-success': item.status === 'success',
            'is-error': item.status === 'error',
          }"
        >
          <img class="file-thumb" :src="item.thumb" alt="preview" />
          <div class="file-info">
            <span class="file-name" :title="item.file.name">{{ item.file.name }}</span>
            <span class="file-size">{{ formatSize(item.file.size) }}</span>
          </div>
          <span v-if="item.status === 'pending'" class="file-badge badge-pending">待识别</span>
          <span v-else-if="item.status === 'processing'" class="file-badge badge-processing">识别中...</span>
          <span v-else-if="item.status === 'success'" class="file-badge badge-success">{{ item.result.class_name }}</span>
          <span v-else-if="item.status === 'error'" class="file-badge badge-error">失败</span>
          <button
            v-if="item.status === 'pending' && !isProcessing"
            class="file-remove-btn"
            title="remove"
            @click.stop="removeFile(index)"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>

      <div v-if="files.length > 0" class="batch-actions">
        <button
          class="btn btn-primary"
          :disabled="isProcessing || recognisableCount === 0"
          @click="startBatch"
        >
          <span v-if="isProcessing" class="btn-spinner"></span>
          <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          {{ isProcessing ? '识别中 ' + (successCount + errorCount) + '/' + totalCount + ' ...' : '开始批量识别（' + totalCount + ' 张）' }}
        </button>
        <button
          v-if="!isProcessing"
          class="btn btn-clear"
          :disabled="totalCount === 0"
          @click="clearAll"
        >
          清空列表
        </button>
      </div>
    </div>

    <Transition name="fade">
      <div v-if="isProcessing" class="progress-section">
        <div class="progress-header">
          <span>识别进度</span>
          <span>{{ successCount + errorCount }} / {{ totalCount }}</span>
        </div>
        <div class="progress-bar-track">
          <div class="progress-bar-fill" :style="{ width: progressPercent + '%' }"></div>
        </div>
      </div>
    </Transition>

    <Transition name="slide-up">
      <div v-if="completedCount > 0" class="result-section">
        <h3 class="section-title">识别结果</h3>
        <div class="result-table-wrapper">
          <table class="result-table">
            <thead>
              <tr>
                <th class="col-index">#</th>
                <th class="col-file">文件名</th>
                <th class="col-result">识别结果</th>
                <th class="col-confidence">置信度</th>
                <th class="col-top5">Top-5</th>
                <th class="col-status">状态</th>
                <th class="col-feedback">反馈</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="(item, index) in completedItems" :key="item.id">
                <tr :class="{ 'row-error': item.status === 'error' }">
                  <td class="col-index">{{ index + 1 }}</td>
                  <td class="col-file" :title="item.file.name">
                    <div class="file-cell">
                      <img class="cell-thumb" :src="item.thumb" alt="" />
                      <span class="cell-name">{{ item.file.name }}</span>
                    </div>
                  </td>
                  <td class="col-result">
                    <template v-if="item.status === 'success'">
                      <span class="result-tag">{{ item.result.class_name }}</span>
                      <span class="result-id">#{{ item.result.class_id }}</span>
                    </template>
                    <span v-else class="error-text">{{ item.errorMsg }}</span>
                  </td>
                  <td class="col-confidence">
                    <template v-if="item.status === 'success'">
                      <div class="confidence-bar">
                        <div
                          class="confidence-fill"
                          :class="confidenceLevel(item.result.confidence)"
                          :style="{ width: (item.result.confidence * 100) + '%' }"
                        ></div>
                      </div>
                      <span class="confidence-num">{{ (item.result.confidence * 100).toFixed(1) }}%</span>
                    </template>
                  </td>
                  <td class="col-top5">
                    <button
                      v-if="item.status === 'success'"
                      class="batch-top5-btn"
                      @click.stop="openBatchTop5(item)"
                    >Top5</button>
                    <span v-else class="action-placeholder">--</span>
                  </td>
                  <td class="col-status">
                    <span v-if="item.status === 'success'" class="status-badge status-ok">成功</span>
                    <span v-else class="status-badge status-fail">失败</span>
                  </td>
                  <td class="col-feedback">
                    <button
                      v-if="item.status === 'success'"
                      class="feedback-btn"
                      :class="{ active: feedbackTarget && feedbackTarget.id === item.id }"
                      title="提交反馈"
                      type="button"
                      @click.stop="onFeedbackClick(item)"
                    >?</button>
                    <span v-else class="action-placeholder">--</span>
                  </td>
                </tr>
                <!-- 反馈行 -->
                <tr v-if="feedbackTarget && feedbackTarget.id === item.id" class="feedback-row">
                  <td colspan="7">
                    <div class="feedback-inline">
                      <div class="feedback-inner">
                        <span class="feedback-predicted">
                          模型预测：<strong>{{ item.result.class_name }}</strong>
                          #{{ item.result.class_id }}
                          （{{ (item.result.confidence * 100).toFixed(1) }}%）
                        </span>
                        <div class="feedback-form-row">
                          <select v-model="fbCorrectClass" class="fb-select">
                            <option :value="null" disabled>-- 请选择正确类别 --</option>
                            <option :value="-1">无正确类别</option>
                            <option
                              v-for="cls in CLASS_NAMES"
                              :key="cls.id"
                              :value="cls.id"
                              :disabled="cls.id === item.result.class_id"
                            >#{{ cls.id }} {{ cls.name }}</option>
                          </select>
                          <input v-model="fbRemark" class="fb-input" placeholder="备注（可选）" maxlength="200" />
                          <button class="fb-submit-btn" :disabled="fbSubmitting" @click="handleSubmitFeedback(item)">
                            <span v-if="fbSubmitting" class="btn-spinner"></span>
                            {{ fbSubmitting ? '提交中' : '提交' }}
                          </button>
                          <button class="fb-cancel-btn" @click="closeFeedback">取消</button>
                        </div>
                        <span v-if="fbError" class="fb-error">{{ fbError }}</span>
                        <span v-if="fbSuccess" class="fb-success">{{ fbSuccess }}</span>
                      </div>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>

        <div class="summary-bar">
          <span class="summary-item summary-ok">成功 {{ successCount }} 张</span>
          <span class="summary-item summary-fail">失败 {{ errorCount }} 张</span>
          <span class="summary-item summary-total">共 {{ completedCount }} 张</span>
        </div>
      </div>
    </Transition>

    <!-- Top5 模态框 -->
    <Top5Modal
      :visible="batchTop5Visible"
      :top5="batchTop5Data"
      :filename="batchTop5Filename"
      @close="batchTop5Visible = false"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { predictImage, submitFeedback } from '../api/predict.js'
import { CLASS_NAMES } from '../data/classNames.js'
import Top5Modal from '../components/Top5Modal.vue'

const files = ref([])
const isDragover = ref(false)
const isProcessing = ref(false)
let idCounter = 0

const ALLOWED_TYPES = ['image/jpeg', 'image/png']

function readFileAsDataURL(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target.result)
    reader.readAsDataURL(file)
  })
}

async function addFiles(fileList) {
  const newItems = []
  for (const file of fileList) {
    if (!ALLOWED_TYPES.includes(file.type)) continue
    const exists = files.value.some(
      (f) => f.file.name === file.name && f.file.size === file.size
    )
    if (exists) continue
    const thumb = await readFileAsDataURL(file)
    newItems.push({
      id: ++idCounter,
      file,
      thumb,
      status: 'pending',
      result: null,
      errorMsg: '',
    })
  }
  files.value.push(...newItems)
}

function removeFile(index) {
  files.value.splice(index, 1)
}

function clearAll() {
  files.value = []
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB'
  return (bytes / (1024 * 1024)).toFixed(1) + 'MB'
}

function triggerFileInput() {
  if (isProcessing.value) return
  const input = document.createElement('input')
  input.type = 'file'
  input.multiple = true
  input.accept = '.jpg,.jpeg,.png'
  input.onchange = (e) => {
    if (e.target.files.length > 0) {
      addFiles(e.target.files)
    }
  }
  input.click()
}

function handleDrop(e) {
  isDragover.value = false
  if (e.dataTransfer.files.length > 0) {
    addFiles(e.dataTransfer.files)
  }
}

const totalCount = computed(() => files.value.length)

const recognisableCount = computed(() =>
  files.value.filter((f) => f.status === 'pending').length
)

const successCount = computed(() =>
  files.value.filter((f) => f.status === 'success').length
)

const errorCount = computed(() =>
  files.value.filter((f) => f.status === 'error').length
)

const completedCount = computed(() => successCount.value + errorCount.value)

const completedItems = computed(() =>
  files.value.filter((f) => f.status === 'success' || f.status === 'error')
)

const progressPercent = computed(() => {
  if (totalCount.value === 0) return 0
  return (completedCount.value / totalCount.value) * 100
})

function confidenceLevel(confidence) {
  if (confidence >= 0.9) return 'level-high'
  if (confidence >= 0.7) return 'level-mid'
  return 'level-low'
}

async function startBatch() {
  if (isProcessing.value) return
  isProcessing.value = true
  const pendingItems = files.value.filter((f) => f.status === 'pending')

  for (const item of pendingItems) {
    item.status = 'processing'
    try {
      const result = await predictImage(item.file, 'batch')
      item.result = result
      item.status = 'success'
    } catch (err) {
      item.status = 'error'
      item.errorMsg = err.response?.data?.detail || err.message || '识别失败'
    }
  }

  isProcessing.value = false
}

// ===== Top5 =====
const batchTop5Visible = ref(false)
const batchTop5Data = ref([])
const batchTop5Filename = ref('')

function openBatchTop5(item) {
  batchTop5Data.value = item.result?.top5 || []
  batchTop5Filename.value = item.file.name
  batchTop5Visible.value = true
}

// ===== 反馈 =====
const feedbackTarget = ref(null)
const fbCorrectClass = ref(null)
const fbRemark = ref('')
const fbSubmitting = ref(false)
const fbError = ref('')
const fbSuccess = ref('')

function onFeedbackClick(item) {
  console.log('[Batch] feedback click', item?.id, item?.status, 'current:', feedbackTarget.value?.id)
  if (feedbackTarget.value && feedbackTarget.value.id === item.id) {
    closeFeedback()
  } else {
    feedbackTarget.value = item
    fbCorrectClass.value = null
    fbRemark.value = ''
    fbError.value = ''
    fbSuccess.value = ''
  }
}

function closeFeedback() {
  feedbackTarget.value = null
  fbCorrectClass.value = null
  fbRemark.value = ''
  fbSubmitting.value = false
  fbError.value = ''
  fbSuccess.value = ''
}

async function handleSubmitFeedback(item) {
  if (fbCorrectClass.value === null) {
    fbError.value = '请选择正确类别'
    return
  }

  fbError.value = ''
  fbSuccess.value = ''
  fbSubmitting.value = true

  try {
    const isNoCorrect = fbCorrectClass.value === -1
    const data = {
      image_name: item.file.name,
      predicted_class_id: item.result.class_id,
      predicted_class_name: item.result.class_name,
      correct_class_id: fbCorrectClass.value,
      correct_class_name: isNoCorrect
        ? '无正确类别'
        : (CLASS_NAMES.find(c => c.id === fbCorrectClass.value)?.name || 'Unknown'),
      confidence: item.result.confidence,
      remark: fbRemark.value.trim(),
    }

    await submitFeedback(data, item.file)
    fbSuccess.value = '反馈提交成功'
    setTimeout(() => closeFeedback(), 1500)
  } catch (err) {
    fbError.value = err.response?.data?.detail || err.message || '反馈提交失败'
  } finally {
    fbSubmitting.value = false
  }
}
</script>

<style scoped>
.page-content { display: flex; flex-direction: column; gap: 24px; }
.page-header { margin-bottom: 4px; }
.page-title { font-size: 1.3rem; font-weight: 700; color: var(--color-text); }
.page-desc { font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 4px; }
.upload-section { background: var(--color-card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.upload-zone { border: 2px dashed var(--color-border); border-radius: var(--radius); padding: 40px 20px; display: flex; flex-direction: column; align-items: center; gap: 8px; cursor: pointer; transition: all var(--transition); background: #fafbfc; }
.upload-zone:hover, .upload-zone.is-dragover { border-color: var(--color-primary); background: var(--color-primary-light); }
.upload-icon { width: 40px; height: 40px; color: var(--color-text-secondary); opacity: 0.5; }
.upload-hint { font-size: 0.95rem; color: var(--color-text-secondary); font-weight: 500; }
.upload-limit { font-size: 0.78rem; color: #94a3b8; }
.model-hint { display: inline-block; margin-top: 4px; padding: 2px 10px; background: var(--color-primary-light); color: var(--color-primary); border-radius: 10px; font-size: 0.75rem; font-weight: 500; }
.file-preview-list { display: flex; flex-wrap: wrap; gap: 10px; }
.file-preview-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: #f8fafc; border: 1px solid var(--color-border); border-radius: 8px; width: calc(50% - 5px); min-width: 0; position: relative; transition: all var(--transition); }
.file-preview-item.is-processing { border-color: var(--color-primary); background: var(--color-primary-light); }
.file-preview-item.is-success { border-color: #bbf7d0; background: #f0fdf4; }
.file-preview-item.is-error { border-color: #fecaca; background: #fef2f2; }
.file-thumb { width: 44px; height: 44px; border-radius: 6px; object-fit: cover; flex-shrink: 0; background: #e2e8f0; }
.file-info { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.file-name { font-size: 0.82rem; font-weight: 500; color: var(--color-text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-size { font-size: 0.72rem; color: #94a3b8; }
.file-badge { flex-shrink: 0; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge-pending { background: #e2e8f0; color: #64748b; }
.badge-processing { background: var(--color-primary-light); color: var(--color-primary); }
.badge-success { background: #dcfce7; color: var(--color-success); }
.badge-error { background: #fee2e2; color: var(--color-danger); }
.file-remove-btn { position: absolute; top: -6px; right: -6px; width: 20px; height: 20px; border-radius: 50%; border: none; background: var(--color-danger); color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; opacity: 0.85; transition: opacity var(--transition); }
.file-remove-btn:hover { opacity: 1; }
.batch-actions { display: flex; gap: 12px; align-items: center; }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 10px 22px; border-radius: 8px; font-size: 0.88rem; font-weight: 600; cursor: pointer; border: none; transition: all var(--transition); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--color-primary); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--color-primary-dark); }
.btn-clear { background: transparent; color: var(--color-text-secondary); border: 1.5px solid var(--color-border); }
.btn-clear:hover:not(:disabled) { border-color: var(--color-danger); color: var(--color-danger); }
.btn-spinner { width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.progress-section { background: var(--color-card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px 24px; display: flex; flex-direction: column; gap: 10px; }
.progress-header { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--color-text-secondary); }
.progress-bar-track { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--color-primary), #6366f1); transition: width 0.4s ease; }
.result-section { background: var(--color-card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px 24px; display: flex; flex-direction: column; gap: 12px; }
.section-title { font-size: 0.95rem; font-weight: 600; color: var(--color-text); }
.result-table-wrapper { overflow-x: auto; }
.result-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.result-table th { text-align: left; padding: 6px 8px; font-weight: 600; color: var(--color-text-secondary); border-bottom: 2px solid var(--color-border); white-space: nowrap; }
.result-table td { padding: 6px 8px; border-bottom: 1px solid var(--color-border); vertical-align: middle; }
.result-table tbody tr:hover { background: var(--color-hover); }
.result-table tbody tr.row-error { background: #fffbfb; }
.col-index { width: 32px; text-align: center; color: #94a3b8; font-size: 0.78rem; }
.col-file { min-width: 140px; }
.col-result { min-width: 120px; }
.col-confidence { min-width: 100px; white-space: nowrap; }
.col-status { width: 44px; text-align: center; }
.col-top5 { width: 56px; text-align: center; }
.col-feedback { width: 44px; text-align: center; }

.batch-top5-btn {
  padding: 2px 8px;
  border: 1.5px solid var(--color-primary);
  border-radius: 4px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition);
}
.batch-top5-btn:hover { background: var(--color-primary); color: #fff; }
.file-cell { display: flex; align-items: center; gap: 6px; }
.cell-thumb { width: 28px; height: 28px; border-radius: 4px; object-fit: cover; flex-shrink: 0; background: #e2e8f0; }
.cell-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text); font-size: 0.8rem; }
.result-tag { display: inline-block; padding: 1px 6px; background: var(--color-primary-light); color: var(--color-primary); border-radius: 4px; font-weight: 500; font-size: 0.78rem; margin-right: 4px; }
.result-id { color: #94a3b8; font-size: 0.72rem; }
.error-text { color: var(--color-danger); font-size: 0.76rem; }
.confidence-bar { display: inline-block; width: 44px; height: 5px; background: #e2e8f0; border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 6px; }
.confidence-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.confidence-fill.level-high { background: var(--color-success); }
.confidence-fill.level-mid { background: var(--color-primary); }
.confidence-fill.level-low { background: var(--color-warning); }
.confidence-num { font-size: 0.78rem; font-weight: 600; color: var(--color-text); vertical-align: middle; }
.status-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.72rem; font-weight: 600; }
.status-ok { background: #dcfce7; color: var(--color-success); }
.status-fail { background: #fee2e2; color: var(--color-danger); }
.action-placeholder { color: #d1d5db; font-size: 0.72rem; }
.summary-bar { display: flex; gap: 16px; padding-top: 6px; border-top: 1px solid var(--color-border); font-size: 0.8rem; }

/* 反馈按钮 */
.feedback-btn {
  width: 24px; height: 24px; border-radius: 50%;
  border: 1.5px solid var(--color-border);
  background: #f8fafc;
  color: var(--color-text-secondary);
  font-size: 0.8rem; font-weight: 700;
  cursor: pointer; display: inline-flex;
  align-items: center; justify-content: center;
  transition: all var(--transition);
  line-height: 1;
}
.feedback-btn:hover { border-color: var(--color-primary); color: var(--color-primary); background: var(--color-primary-light); }
.feedback-btn.active { border-color: var(--color-primary); color: #fff; background: var(--color-primary); }

/* 反馈展开行 */
.feedback-row td { padding: 0 !important; border-bottom: none; }
.feedback-inline { padding: 10px 16px 10px 48px; background: #f8fafc; border-bottom: 1px solid var(--color-border); }
.feedback-inner { display: flex; flex-direction: column; gap: 8px; }
.feedback-predicted { font-size: 0.78rem; color: var(--color-text-secondary); }
.feedback-predicted strong { color: var(--color-text); }
.feedback-form-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.fb-select {
  padding: 5px 8px; border: 1.5px solid var(--color-border); border-radius: 6px;
  font-size: 0.78rem; font-family: inherit; color: var(--color-text);
  background: #fff; min-width: 160px;
}
.fb-input {
  padding: 5px 8px; border: 1.5px solid var(--color-border); border-radius: 6px;
  font-size: 0.78rem; font-family: inherit; color: var(--color-text);
  background: #fff; min-width: 140px; flex: 1;
}
.fb-submit-btn {
  padding: 5px 14px; border: none; border-radius: 6px;
  background: var(--color-primary); color: #fff;
  font-size: 0.78rem; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;
}
.fb-submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.fb-cancel-btn {
  padding: 5px 10px; border: 1.5px solid var(--color-border); border-radius: 6px;
  background: #fff; color: var(--color-text-secondary);
  font-size: 0.78rem; cursor: pointer;
}
.fb-error { font-size: 0.76rem; color: var(--color-danger); }
.fb-success { font-size: 0.76rem; color: var(--color-success); }
.summary-item { font-weight: 500; }
.summary-ok { color: var(--color-success); }
.summary-fail { color: var(--color-danger); }
.summary-total { color: var(--color-text-secondary); }
</style>
