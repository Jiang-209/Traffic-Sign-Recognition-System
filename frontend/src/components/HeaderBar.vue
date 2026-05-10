<!--
  HeaderBar.vue - 顶部导航栏
  展示系统标题与副标题
-->
<template>
  <header class="header">
    <div class="header-inner">
      <!-- 图标 -->
      <div class="header-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
      </div>
      <!-- 标题 -->
      <div class="header-text">
        <h1 class="header-title">交通标志识别系统</h1>
        <p class="header-subtitle">Traffic Sign Recognition System · Based on CNN</p>
      </div>
      <!-- 状态指示 -->
      <div class="header-status" :class="{ online: backendOnline }">
        <span class="status-dot"></span>
        <span class="status-text">{{ backendOnline ? '后端已连接' : '后端未连接' }}</span>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { checkHealth } from '../api/predict.js'

const backendOnline = ref(false)

/**
 * 启动时检测后端健康状态
 */
onMounted(async () => {
  try {
    const res = await checkHealth()
    backendOnline.value = res.status === 'ok' && res.model_loaded === true
  } catch {
    backendOnline.value = false
  }
})
</script>

<style scoped>
.header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
  color: #fff;
  padding: 0 24px;
  box-shadow: 0 2px 12px rgba(37, 99, 235, 0.25);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-inner {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 16px;
  height: 64px;
}

.header-icon {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  opacity: 0.9;
}

.header-text {
  flex: 1;
}

.header-title {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.header-subtitle {
  font-size: 0.75rem;
  opacity: 0.75;
  font-weight: 400;
}

/* 状态指示器 */
.header-status {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.12);
  font-size: 0.8rem;
  transition: background var(--transition);
}

.header-status.online {
  background: rgba(34, 197, 94, 0.25);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f87171;
  transition: background var(--transition);
}

.header-status.online .status-dot {
  background: #4ade80;
  box-shadow: 0 0 6px #4ade80;
}

.status-text {
  white-space: nowrap;
}
</style>
