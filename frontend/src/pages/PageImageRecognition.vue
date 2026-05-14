<template>
  <div class="page-content">
    <div class="main-grid">
      <div class="col-left">
        <UploadPanel
          @predict-start="onPredictStart"
          @predict-success="onPredictSuccess"
          @predict-error="onPredictError"
          @reset="onReset"
          @file-selected="onFileSelected"
        />
      </div>

      <div class="col-right">
        <ResultCard :result="result" :is-loading="isLoading" />

        <Transition name="slide-up">
          <div v-if="result" class="top5-bar">
            <button class="top5-btn" @click="openTop5">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              查看 Top-5 预测
            </button>
          </div>
        </Transition>

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

    <Transition name="slide-up">
      <HistoryTable v-if="historyTrigger" :new-record="historyTrigger" />
    </Transition>

    <!-- Top5 模态框 -->
    <Top5Modal
      :visible="top5Visible"
      :top5="top5Data"
      :filename="top5Filename"
      @close="top5Visible = false"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadPanel from '../components/UploadPanel.vue'
import ResultCard from '../components/ResultCard.vue'
import FeedbackPanel from '../components/FeedbackPanel.vue'
import HistoryTable from '../components/HistoryTable.vue'
import Top5Modal from '../components/Top5Modal.vue'

const result = ref(null)
const isLoading = ref(false)
const historyTrigger = ref(null)
const selectedFile = ref(null)

const top5Visible = ref(false)
const top5Data = ref([])
const top5Filename = ref('')

function onFileSelected(file) {
  selectedFile.value = file
}

function onPredictStart() {
  isLoading.value = true
  result.value = null
}

function onPredictSuccess(data) {
  result.value = data
  isLoading.value = false
  historyTrigger.value = { ...data }
}

function onPredictError() {
  isLoading.value = false
  result.value = null
}

function onReset() {
  result.value = null
  isLoading.value = false
  selectedFile.value = null
  top5Visible.value = false
}

function onFeedbackSubmitted(feedbackData) {
  console.log('[反馈] 用户提交反馈:', feedbackData)
}

function openTop5() {
  if (!result.value || !result.value.top5) return
  top5Data.value = result.value.top5
  top5Filename.value = selectedFile.value?.name || ''
  top5Visible.value = true
}
</script>

<style scoped>
.page-content {
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

.top5-bar {
  display: flex;
  justify-content: center;
}
.top5-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border: 1.5px solid var(--color-primary);
  border-radius: 8px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition);
}
.top5-btn:hover {
  background: var(--color-primary);
  color: #fff;
}

@media (max-width: 860px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
</style>
