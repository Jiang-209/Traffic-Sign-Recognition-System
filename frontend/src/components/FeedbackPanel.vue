<!--
  FeedbackPanel.vue - 用户反馈面板
  当模型识别错误时，用户可选择正确类别并提交反馈
  反馈数据用于后续模型优化与数据集扩充
-->
<template>
  <div class="feedback-wrapper">
    <!-- 触发按钮 -->
    <button
      v-if="!showForm"
      class="feedback-trigger"
      @click="showForm = true"
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      识别有误？提交反馈
    </button>

    <!-- 反馈表单（展开/收起动画） -->
    <Transition name="expand">
      <div v-if="showForm" class="feedback-card">
        <h4 class="feedback-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          提交反馈
        </h4>

        <!-- 模型预测结果（只读） -->
        <div class="form-group">
          <label class="form-label">模型预测结果</label>
          <div class="readonly-field">
            <span class="tag predicted-tag">#{{ result.class_id }} {{ result.class_name }}</span>
            <span class="confidence-small">置信度 {{ (result.confidence * 100).toFixed(1) }}%</span>
          </div>
        </div>

        <!-- 正确类别选择 -->
        <div class="form-group">
          <label class="form-label" for="correct-class">
            正确类别 <span class="required">*</span>
          </label>
          <select
            id="correct-class"
            v-model="correctClassId"
            class="form-select"
            :class="{ 'has-error': validationError }"
          >
            <option :value="null" disabled>-- 请选择正确类别 --</option>
            <option :value="-1">无正确类别</option>
            <option
              v-for="cls in CLASS_NAMES"
              :key="cls.id"
              :value="cls.id"
              :disabled="cls.id === result.class_id"
            >
              #{{ cls.id }} {{ cls.name }}
            </option>
          </select>
          <span v-if="validationError" class="error-text">{{ validationError }}</span>
        </div>

        <!-- 备注 -->
        <div class="form-group">
          <label class="form-label" for="remark">备注（可选）</label>
          <textarea
            id="remark"
            v-model="remark"
            class="form-textarea"
            placeholder="例如：模型将「停止」误识别为「让行」"
            rows="2"
            maxlength="200"
          ></textarea>
          <span class="char-count">{{ remark.length }}/200</span>
        </div>

        <!-- 按钮区 -->
        <div class="form-actions">
          <button
            class="btn btn-submit"
            :disabled="isSubmitting"
            @click="handleSubmit"
          >
            <span v-if="isSubmitting" class="btn-spinner"></span>
            <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            {{ isSubmitting ? '提交中...' : '提交反馈' }}
          </button>
          <button
            class="btn btn-cancel"
            :disabled="isSubmitting"
            @click="handleCancel"
          >
            取消
          </button>
        </div>

        <!-- 成功提示 -->
        <Transition name="fade">
          <div v-if="successMsg" class="success-banner">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            {{ successMsg }}
          </div>
        </Transition>

        <!-- 错误提示 -->
        <Transition name="fade">
          <div v-if="errorMsg" class="error-banner">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            {{ errorMsg }}
          </div>
        </Transition>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { submitFeedback } from '../api/predict.js'
import { CLASS_NAMES } from '../data/classNames.js'

const props = defineProps({
  /** 当前识别结果 */
  result: {
    type: Object,
    required: true,
  },
  /** 用户上传的原始图片 File 对象 */
  imageFile: {
    type: File,
    default: null,
  },
})

const emit = defineEmits(['feedback-submitted'])

// ===== 表单状态 =====
const showForm = ref(false)          // 是否展开表单
const correctClassId = ref(null)     // 用户选择的正确类别 ID
const remark = ref('')              // 用户备注
const isSubmitting = ref(false)     // 是否正在提交
const validationError = ref('')     // 表单验证错误
const successMsg = ref('')          // 成功消息
const errorMsg = ref('')            // 错误消息

/** 提交反馈 */
async function handleSubmit() {
  // 清除旧消息
  validationError.value = ''
  successMsg.value = ''
  errorMsg.value = ''

  // 表单验证
  if (correctClassId.value === null) {
    validationError.value = '请选择正确类别'
    return
  }

  isSubmitting.value = true

  try {
    // 当选择"无正确类别"时，使用特殊名称
    const isNoCorrectClass = correctClassId.value === -1
    const feedbackData = {
      image_name: props.imageFile?.name || 'unknown.png',
      predicted_class_id: props.result.class_id,
      predicted_class_name: props.result.class_name,
      correct_class_id: correctClassId.value,
      correct_class_name: isNoCorrectClass
        ? '无正确类别'
        : (CLASS_NAMES.find(c => c.id === correctClassId.value)?.name || 'Unknown'),
      confidence: props.result.confidence,
      remark: remark.value.trim(),
    }

    await submitFeedback(feedbackData, props.imageFile)

    // 提交成功
    successMsg.value = '反馈提交成功，感谢您的参与！'
    emit('feedback-submitted', feedbackData)

    // 1.5 秒后自动关闭表单
    setTimeout(() => {
      showForm.value = false
      successMsg.value = ''
    }, 1800)
  } catch (err) {
    const msg = err.response?.data?.detail || err.message || '反馈提交失败，请稍后重试'
    errorMsg.value = msg
  } finally {
    isSubmitting.value = false
  }
}

/** 取消反馈 */
function handleCancel() {
  showForm.value = false
  correctClassId.value = null
  remark.value = ''
  validationError.value = ''
  errorMsg.value = ''
  successMsg.value = ''
}
</script>

<style scoped>
.feedback-wrapper {
  margin-top: 16px;
}

/* 触发按钮 */
.feedback-trigger {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
  background: none;
  border: none;
  color: var(--color-text-secondary);
  font-size: 0.82rem;
  cursor: pointer;
  transition: color var(--transition);
}

.feedback-trigger:hover {
  color: var(--color-primary);
}

/* 反馈卡片 */
.feedback-card {
  background: var(--color-card);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.feedback-title {
  font-size: 0.95rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
}

/* 表单组 */
.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-label {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--color-text-secondary);
}

.required {
  color: var(--color-danger);
}

/* 只读字段 */
.readonly-field {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: #f8fafc;
  border: 1px solid var(--color-border);
  border-radius: 8px;
}

.tag {
  display: inline-block;
  padding: 2px 10px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-radius: 10px;
  font-size: 0.8rem;
  font-weight: 500;
}

.confidence-small {
  font-size: 0.78rem;
  color: #94a3b8;
}

/* 下拉选择 */
.form-select {
  padding: 10px 12px;
  border: 1.5px solid var(--color-border);
  border-radius: 8px;
  font-size: 0.88rem;
  font-family: inherit;
  color: var(--color-text);
  background: #fff;
  cursor: pointer;
  transition: border-color var(--transition);
}

.form-select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.form-select.has-error {
  border-color: var(--color-danger);
}

/* 文本域 */
.form-textarea {
  padding: 10px 12px;
  border: 1.5px solid var(--color-border);
  border-radius: 8px;
  font-size: 0.88rem;
  font-family: inherit;
  color: var(--color-text);
  resize: vertical;
  transition: border-color var(--transition);
}

.form-textarea:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.char-count {
  text-align: right;
  font-size: 0.72rem;
  color: #94a3b8;
}

.error-text {
  font-size: 0.78rem;
  color: var(--color-danger);
}

/* 按钮区 */
.form-actions {
  display: flex;
  gap: 10px;
}

.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 9px 18px;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all var(--transition);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-submit {
  background: var(--color-primary);
  color: #fff;
  flex: 1;
}

.btn-submit:hover:not(:disabled) {
  background: var(--color-primary-dark);
}

.btn-cancel {
  background: var(--color-card);
  color: var(--color-text-secondary);
  border: 1.5px solid var(--color-border);
}

.btn-cancel:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

/* 加载旋转 */
.btn-spinner {
  width: 15px;
  height: 15px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 成功横幅 */
.success-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 8px;
  color: var(--color-success);
  font-size: 0.84rem;
}

/* 错误横幅 */
.error-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  color: var(--color-danger);
  font-size: 0.84rem;
}

/* 展开/收起动画 */
.expand-enter-active {
  transition: all 0.35s ease-out;
}
.expand-leave-active {
  transition: all 0.2s ease-in;
}
.expand-enter-from {
  opacity: 0;
  transform: translateY(-8px);
  max-height: 0;
}
.expand-enter-to {
  opacity: 1;
  transform: translateY(0);
  max-height: 500px;
}
.expand-leave-from {
  opacity: 1;
  transform: translateY(0);
  max-height: 500px;
}
.expand-leave-to {
  opacity: 0;
  transform: translateY(-8px);
  max-height: 0;
}
</style>
