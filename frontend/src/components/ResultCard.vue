<!--
  ResultCard.vue - 识别结果展示卡片
  展示类别名称、类别ID、置信度、处理时间、可靠性
-->
<template>
  <div class="result-panel">
    <h3 class="panel-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
      识别结果
    </h3>

    <!-- 空状态 -->
    <div v-if="!result" class="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="12" y1="18" x2="12" y2="12"/>
        <line x1="9" y1="15" x2="15" y2="15"/>
      </svg>
      <p>请上传图片并点击「开始识别」</p>
    </div>

    <!-- 结果内容 -->
    <Transition name="slide-up">
      <div v-if="result" class="result-content">
        <!-- 类别名称 - 最醒目 -->
        <div class="class-name-section">
          <span class="class-label">识别为</span>
          <span class="class-name">{{ result.class_name }}</span>
          <span class="class-id">#{{ result.class_id }}</span>
        </div>

        <!-- 置信度进度条 -->
        <div class="confidence-section">
          <div class="confidence-header">
            <span>置信度</span>
            <span class="confidence-value">{{ (result.confidence * 100).toFixed(2) }}%</span>
          </div>
          <div class="progress-bar-track">
            <div
              class="progress-bar-fill"
              :class="confidenceLevel"
              :style="{ width: (result.confidence * 100) + '%' }"
            ></div>
          </div>
        </div>

        <!-- 详情网格 -->
        <div class="detail-grid">
          <div class="detail-item">
            <span class="detail-label">类别编号</span>
            <span class="detail-value">{{ result.class_id }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">处理耗时</span>
            <span class="detail-value">{{ result.processing_time }}s</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">可靠性</span>
            <span class="detail-value" :class="{ reliable: result.reliable, unreliable: !result.reliable }">
              {{ result.reliable ? '可信' : '存疑' }}
            </span>
          </div>
        </div>

        <!-- 低置信度警告 -->
        <Transition name="fade">
          <div v-if="!result.reliable" class="warning-banner">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            当前识别结果可信度较低，请上传更清晰、更方正的交通标志图片
          </div>
        </Transition>
      </div>
    </Transition>

    <!-- Loading 骨架 -->
    <div v-if="isLoading" class="loading-skeleton">
      <div class="skeleton-line skeleton-name"></div>
      <div class="skeleton-line skeleton-bar"></div>
      <div class="skeleton-line skeleton-detail"></div>
      <div class="skeleton-line skeleton-detail"></div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  /** 识别结果对象，null 表示未识别 */
  result: {
    type: Object,
    default: null,
  },
  /** 是否正在加载 */
  isLoading: {
    type: Boolean,
    default: false,
  },
})

/**
 * 根据置信度返回进度条颜色等级
 */
const confidenceLevel = computed(() => {
  if (!props.result) return ''
  const c = props.result.confidence
  if (c >= 0.9) return 'level-high'
  if (c >= 0.7) return 'level-mid'
  return 'level-low'
})
</script>

<style scoped>
.result-panel {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 380px;
}

.panel-title {
  font-size: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
}

/* 空状态 */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #94a3b8;
  font-size: 0.9rem;
}

/* 结果内容 */
.result-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 类别名称区 */
.class-name-section {
  text-align: center;
  padding: 12px 0;
}

.class-label {
  display: block;
  font-size: 0.82rem;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.class-name {
  display: block;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--color-primary);
  line-height: 1.3;
}

.class-id {
  display: inline-block;
  margin-top: 4px;
  padding: 2px 10px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-radius: 12px;
  font-size: 0.78rem;
  font-weight: 600;
}

/* 置信度进度条 */
.confidence-section {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.confidence-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: var(--color-text-secondary);
}

.confidence-value {
  font-weight: 700;
  color: var(--color-text);
}

.progress-bar-track {
  height: 10px;
  background: #e2e8f0;
  border-radius: 5px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.6s ease;
}

.progress-bar-fill.level-high {
  background: linear-gradient(90deg, #22c55e, #16a34a);
}
.progress-bar-fill.level-mid {
  background: linear-gradient(90deg, #2563eb, #6366f1);
}
.progress-bar-fill.level-low {
  background: linear-gradient(90deg, #f59e0b, #ea580c);
}

/* 详情网格 */
.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}

.detail-item {
  text-align: center;
  padding: 10px 6px;
  background: #f8fafc;
  border-radius: 8px;
}

.detail-label {
  display: block;
  font-size: 0.75rem;
  color: #94a3b8;
  margin-bottom: 4px;
}

.detail-value {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--color-text);
}

.detail-value.reliable {
  color: var(--color-success);
}

.detail-value.unreliable {
  color: var(--color-warning);
}

/* 警告横幅 */
.warning-banner {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 12px 14px;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  color: var(--color-warning);
  font-size: 0.82rem;
  line-height: 1.5;
}

.warning-banner svg {
  flex-shrink: 0;
  margin-top: 1px;
}

/* Loading 骨架 */
.loading-skeleton {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px 0;
}

.skeleton-line {
  height: 14px;
  background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 7px;
}

.skeleton-name {
  width: 60%;
  height: 28px;
  margin: 0 auto;
}
.skeleton-bar {
  height: 10px;
}
.skeleton-detail {
  width: 80%;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
