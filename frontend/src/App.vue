<!--
  App.vue - 根组件
  组合 HeaderBar / UploadPanel / ResultCard / FeedbackPanel / HistoryTable
  管理全局状态：识别结果、加载状态、错误信息、当前文件
-->
<template>
  <div class="app">
    <HeaderBar />

    <main class="main-content">
      <!-- 主内容区：左右两栏 -->
      <div class="main-grid">
        <!-- 左侧：上传面板 -->
        <div class="col-left">
          <UploadPanel
            @predict-start="onPredictStart"
            @predict-success="onPredictSuccess"
            @predict-error="onPredictError"
            @reset="onReset"
            @file-selected="onFileSelected"
          />
        </div>

        <!-- 右侧：结果卡片 + 反馈面板 -->
        <div class="col-right">
          <ResultCard :result="result" :is-loading="isLoading" />

          <!-- 反馈面板：识别结果存在时显示 -->
          <Transition name="slide-up">
            <FeedbackPanel
              v-if="result"
              :result="result"
              :image-file="selectedFile"
              @feedback-submitted="onFeedbackSubmitted"
            />
          </Transition>
        </div>
      </div>

      <!-- 底部：历史记录 -->
      <Transition name="slide-up">
        <HistoryTable v-if="historyTrigger" :new-record="historyTrigger" />
      </Transition>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import HeaderBar from './components/HeaderBar.vue'
import UploadPanel from './components/UploadPanel.vue'
import ResultCard from './components/ResultCard.vue'
import FeedbackPanel from './components/FeedbackPanel.vue'
import HistoryTable from './components/HistoryTable.vue'

// ===== 全局状态 =====
const result = ref(null)          // 当前识别结果
const isLoading = ref(false)      // 加载状态
const historyTrigger = ref(null)  // 新历史记录（传给 HistoryTable）
const selectedFile = ref(null)    // 用户当前选择的文件（传给 FeedbackPanel）

// ===== 事件处理 =====

/** 用户选择了新文件 */
function onFileSelected(file) {
  selectedFile.value = file
}

/** 开始识别 */
function onPredictStart() {
  isLoading.value = true
  result.value = null
}

/** 识别成功 */
function onPredictSuccess(data) {
  result.value = data
  isLoading.value = false
  // 触发历史记录更新
  historyTrigger.value = { ...data }
}

/** 识别失败 */
function onPredictError() {
  isLoading.value = false
  result.value = null
}

/** 重置 */
function onReset() {
  result.value = null
  isLoading.value = false
  selectedFile.value = null
}

/** 反馈提交成功（可用于后续扩展，如统计反馈次数） */
function onFeedbackSubmitted(feedbackData) {
  // 预留：可在此处添加全局提示或统计
  console.log('[反馈] 用户提交反馈:', feedbackData)
}
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-content {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 28px 24px 48px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

/* 左右两栏布局 */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

/* 响应式：窄屏切换为单栏 */
@media (max-width: 860px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
</style>
