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
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadPanel from '../components/UploadPanel.vue'
import ResultCard from '../components/ResultCard.vue'
import FeedbackPanel from '../components/FeedbackPanel.vue'
import HistoryTable from '../components/HistoryTable.vue'

const result = ref(null)
const isLoading = ref(false)
const historyTrigger = ref(null)
const selectedFile = ref(null)

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
}

function onFeedbackSubmitted(feedbackData) {
  console.log('[反馈] 用户提交反馈:', feedbackData)
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

@media (max-width: 860px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
</style>
