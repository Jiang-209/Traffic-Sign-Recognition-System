<template>
  <div class="history-page">
    <div class="page-header">
      <h2 class="page-title">历史记录</h2>
      <p class="page-subtitle">查看所有识别历史记录</p>
    </div>

    <div v-if="records.length === 0" class="empty-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"/>
        <polyline points="12 6 12 12 16 14"/>
      </svg>
      <p>暂无识别记录</p>
      <span class="empty-hint">完成一次图片识别后，记录将显示在这里</span>
    </div>

    <div v-else class="history-card">
      <div class="history-card-header">
        <span class="record-count">共 {{ records.length }} 条记录</span>
        <button class="btn-clear" @click="handleClear">清空全部</button>
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
          <tbody>
            <tr v-for="(record, index) in records" :key="record.id">
              <td class="td-index">{{ index + 1 }}</td>
              <td class="td-filename" :title="record.fileName">{{ record.fileName }}</td>
              <td><span class="tag">{{ record.class_name }}</span></td>
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
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

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
.history-page {
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

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 60px 20px;
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  color: #94a3b8;
}

.empty-state svg {
  opacity: 0.5;
}

.empty-state p {
  font-size: 0.95rem;
  color: var(--color-text-secondary);
}

.empty-hint {
  font-size: 0.8rem;
  color: #94a3b8;
}

.history-card {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 24px;
}

.history-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.record-count {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
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

.table-wrapper {
  overflow-x: auto;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

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

.history-table tbody tr {
  transition: background var(--transition);
}

.history-table tbody tr:hover {
  background: var(--color-hover);
}

.td-index {
  color: #94a3b8;
  font-weight: 500;
  width: 40px;
}

.td-filename {
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.td-time {
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  white-space: nowrap;
}

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

.badge-high { color: #16a34a; font-weight: 700; }
.badge-mid  { color: var(--color-primary); font-weight: 700; }
.badge-low  { color: #ea580c; font-weight: 700; }

.status-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 0.78rem;
  font-weight: 500;
}

.status-badge.reliable {
  background: #f0fdf4;
  color: #16a34a;
}

.status-badge.unreliable {
  background: #fff7ed;
  color: #ea580c;
}
</style>
