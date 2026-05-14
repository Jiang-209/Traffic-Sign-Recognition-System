<template>
  <div class="history-panel" v-if="records.length > 0">
    <div class="history-header">
      <h3 class="panel-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
        识别历史
        <span class="record-count">{{ records.length }} 条</span>
      </h3>
      <button class="btn-clear" @click="handleClear">清空记录</button>
    </div>

    <div class="table-wrapper">
      <table class="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>图片名称</th>
            <th>识别结果</th>
            <th>置信度</th>
            <th>时间</th>
            <th>状态</th>
          </tr>
        </thead>
        <TransitionGroup name="fade" tag="tbody">
          <tr v-for="(record, index) in records" :key="record.id">
            <td class="td-index">{{ index + 1 }}</td>
            <td class="td-filename" :title="record.fileName">{{ record.fileName }}</td>
            <td>
              <span class="tag">{{ record.class_name }}</span>
            </td>
            <td class="td-confidence">
              <span :class="confidenceBadge(record.confidence)">
                {{ (record.confidence * 100).toFixed(1) }}%
              </span>
            </td>
            <td class="td-time">{{ record.timestamp }}</td>
            <td>
              <span class="status-badge" :class="{ reliable: record.reliable, unreliable: !record.reliable }">
                {{ record.reliable ? '可信' : '存疑' }}
              </span>
            </td>
          </tr>
        </TransitionGroup>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  newRecord: { type: Object, default: null },
})

const STORAGE_KEY = 'traffic_sign_history'
const records = ref(loadHistory())

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory() {
  const trimmed = records.value.slice(0, 50)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
}

watch(() => props.newRecord, (record) => {
  if (!record) return
  records.value.unshift({
    ...record,
    id: Date.now(),
    top5: record.top5 || [],
  })
  saveHistory()
}, { deep: true })

function handleClear() {
  records.value = []
  localStorage.removeItem(STORAGE_KEY)
}

function confidenceBadge(confidence) {
  if (confidence >= 0.9) return 'badge-high'
  if (confidence >= 0.7) return 'badge-mid'
  return 'badge-low'
}
</script>

<style scoped>
.history-panel {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 24px;
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.panel-title {
  font-size: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
}

.record-count {
  font-size: 0.78rem;
  font-weight: 400;
  color: #94a3b8;
  background: #f1f5f9;
  padding: 2px 10px;
  border-radius: 10px;
}

.btn-clear {
  background: none;
  border: 1px solid var(--color-border);
  padding: 5px 14px;
  border-radius: 6px;
  font-size: 0.8rem;
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all var(--transition);
}

.btn-clear:hover {
  border-color: var(--color-danger);
  color: var(--color-danger);
  background: #fef2f2;
}

.table-wrapper { overflow-x: auto; }

.history-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }

.history-table th {
  text-align: left;
  padding: 10px 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 2px solid var(--color-border);
  white-space: nowrap;
}

.history-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #f1f5f9;
  color: var(--color-text);
}

.history-table tbody tr { transition: background var(--transition); }
.history-table tbody tr:hover { background: var(--color-hover); }

.td-index { color: #94a3b8; font-weight: 500; width: 40px; }
.td-filename { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.td-time { color: var(--color-text-secondary); font-size: 0.8rem; white-space: nowrap; }

.tag {
  display: inline-block;
  padding: 2px 10px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-radius: 10px;
  font-size: 0.8rem;
  font-weight: 500;
  white-space: nowrap;
}

.badge-high { color: var(--color-success); font-weight: 700; }
.badge-mid  { color: var(--color-primary); font-weight: 700; }
.badge-low  { color: var(--color-warning); font-weight: 700; }

.status-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 0.78rem;
  font-weight: 500;
}

.status-badge.reliable { background: #f0fdf4; color: var(--color-success); }
.status-badge.unreliable { background: #fff7ed; color: var(--color-warning); }
</style>
