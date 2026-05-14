<template>
  <div class="directory-page">
    <div class="page-header">
      <h2 class="page-title">交通标志大全</h2>
      <p class="page-subtitle">TSRD 数据集 58 类中国交通标志</p>
    </div>

    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <span>加载中...</span>
    </div>

    <div v-else-if="error" class="error-state">
      加载失败：{{ error }}
    </div>

    <div v-else class="sign-grid">
      <div v-for="sign in signs" :key="sign.class_id" class="sign-card">
        <span class="sign-id">#{{ sign.class_id }}</span>
        <span class="sign-name">{{ sign.class_name }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { apiClient } from '../api/predict.js'

const signs = ref([])
const loading = ref(true)
const error = ref(null)

onMounted(async () => {
  try {
    const { data } = await apiClient.get('/classes')
    signs.value = data
  } catch (e) {
    error.value = e.message || '无法加载交通标志列表'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.directory-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  margin-bottom: 4px;
}

.page-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--color-text);
}

.page-subtitle {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
  margin-top: 4px;
}

.loading-state,
.error-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 60px 20px;
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 0.9rem;
  color: var(--color-text-secondary);
}

.error-state {
  color: var(--color-danger);
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.sign-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}

.sign-card {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  background: var(--color-card);
  border-radius: 8px;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--color-border);
  transition: all var(--transition);
}

.sign-card:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow);
}

.sign-id {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--color-primary);
  background: var(--color-primary-light);
  padding: 2px 6px;
  border-radius: 4px;
  flex-shrink: 0;
}

.sign-name {
  font-size: 0.85rem;
  color: var(--color-text);
  font-weight: 500;
  line-height: 1.3;
}
</style>
