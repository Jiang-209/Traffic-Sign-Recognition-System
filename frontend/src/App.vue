<template>
  <div class="app">
    <HeaderBar />

    <div class="app-body">
      <SideBar :current-page="currentPage" @navigate="currentPage = $event" />

      <main class="main-content">
        <PageImageRecognition v-if="currentPage === 'image-recognition'" />
        <PageCameraRecognition v-else-if="currentPage === 'camera-recognition'" />
        <PageBatchRecognition v-else-if="currentPage === 'batch-recognition'" />
        <PageSignDirectory v-else-if="currentPage === 'sign-directory'" />
        <PageHistory v-else-if="currentPage === 'history'" />
        <PageSystemInfo v-else-if="currentPage === 'system-info'" />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import HeaderBar from './components/HeaderBar.vue'
import SideBar from './components/SideBar.vue'
import PageImageRecognition from './pages/PageImageRecognition.vue'
import PageCameraRecognition from './pages/PageCameraRecognition.vue'
import PageBatchRecognition from './pages/PageBatchRecognition.vue'
import PageSignDirectory from './pages/PageSignDirectory.vue'
import PageHistory from './pages/PageHistory.vue'
import PageSystemInfo from './pages/PageSystemInfo.vue'

const currentPage = ref('image-recognition')
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-body {
  display: flex;
  flex: 1;
}

.main-content {
  flex: 1;
  padding: 28px 32px 48px;
  min-height: calc(100vh - 64px);
  overflow-y: auto;
}
</style>
