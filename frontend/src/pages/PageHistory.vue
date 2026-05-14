<template>
  <div class="history-page">
    <div class="page-header">
      <h2 class="page-title">历史记录</h2>
      <p class="page-subtitle">查看所有识别历史，支持检索与筛选</p>
    </div>

    <!-- 筛选栏 -->
    <div class="filter-bar">
      <div class="filter-row">
        <div class="filter-group">
          <label class="filter-label">标志名称</label>
          <input v-model="filters.keyword" class="filter-input" placeholder="搜索标志名称..." />
        </div>
        <div class="filter-group">
          <label class="filter-label">状态</label>
          <select v-model="filters.status" class="filter-select">
            <option value="all">全部</option>
            <option value="reliable">可信</option>
            <option value="unreliable">存疑</option>
          </select>
        </div>
        <div class="filter-group">
          <label class="filter-label">最低置信度</label>
          <div class="range-group">
            <input v-model.number="filters.minConf" type="range" min="0" max="100" class="filter-range" />
            <span class="range-value">{{ filters.minConf }}%</span>
          </div>
        </div>
        <div class="filter-group">
          <label class="filter-label">时间</label>
          <select v-model="filters.timeRange" class="filter-select">
            <option value="all">全部</option>
            <option value="today">今天</option>
            <option value="week">最近一周</option>
            <option value="month">最近一个月</option>
          </select>
        </div>
        <button class="btn-reset" @click="resetFilters">重置</button>
      </div>
    </div>

    <div v-if="filteredRecords.length === 0" class="empty-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"/>
        <polyline points="12 6 12 12 16 14"/>
      </svg>
      <p>{{ records.length === 0 ? '暂无识别记录' : '没有匹配的记录' }}</p>
      <span class="empty-hint" v-if="records.length > 0">尝试调整筛选条件</span>
    </div>

    <div v-else class="history-card">
      <div class="history-card-header">
        <span class="record-count">共 {{ records.length }} 条，筛选后 {{ filteredRecords.length }} 条</span>
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
            <tr v-for="(record, index) in filteredRecords" :key="record.id">
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
import { ref, computed } from 'vue'

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

// ===== 筛选 =====
const filters = ref({
  keyword: '',
  status: 'all',
  minConf: 0,
  timeRange: 'all',
})

function resetFilters() {
  filters.value = { keyword: '', status: 'all', minConf: 0, timeRange: 'all' }
}

const filteredRecords = computed(() => {
  return records.value.filter((r) => {
    // 名称检索
    if (filters.value.keyword) {
      const kw = filters.value.keyword.toLowerCase()
      const nameMatch = r.class_name && r.class_name.toLowerCase().includes(kw)
      const fileNameMatch = r.fileName && r.fileName.toLowerCase().includes(kw)
      if (!nameMatch && !fileNameMatch) return false
    }
    // 状态
    if (filters.value.status === 'reliable' && !r.reliable) return false
    if (filters.value.status === 'unreliable' && r.reliable) return false
    // 最低置信度
    if ((r.confidence * 100) < filters.value.minConf) return false
    // 时间范围
    if (filters.value.timeRange !== 'all') {
      const now = Date.now()
      const ts = r.timestamp ? new Date(r.timestamp).getTime() : 0
      if (!ts) return false
      if (filters.value.timeRange === 'today') {
        const today = new Date()
        today.setHours(0, 0, 0, 0)
        if (ts < today.getTime()) return false
      } else if (filters.value.timeRange === 'week') {
        if (now - ts > 7 * 24 * 60 * 60 * 1000) return false
      } else if (filters.value.timeRange === 'month') {
        if (now - ts > 30 * 24 * 60 * 60 * 1000) return false
      }
    }
    return true
  })
})

function confidenceBadge(confidence) {
  if (confidence >= 0.9) return 'badge-high'
  if (confidence >= 0.7) return 'badge-mid'
  return 'badge-low'
}
</script>

<style scoped>
.history-page { display: flex; flex-direction: column; gap: 20px; }
.page-header { margin-bottom: 4px; }
.page-title { font-size: 1.2rem; font-weight: 700; color: var(--color-text); }
.page-subtitle { font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 4px; }

/* 筛选栏 */
.filter-bar {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 16px 20px;
}

.filter-row {
  display: flex;
  gap: 16px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.filter-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.filter-input, .filter-select {
  padding: 7px 10px;
  border: 1.5px solid var(--color-border);
  border-radius: 6px;
  font-size: 0.82rem;
  font-family: inherit;
  color: var(--color-text);
  background: #fff;
  min-width: 120px;
}

.filter-input:focus, .filter-select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.range-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-range {
  width: 100px;
  accent-color: var(--color-primary);
}

.range-value {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-text);
  min-width: 36px;
}

.btn-reset {
  padding: 7px 16px;
  border: 1.5px solid var(--color-border);
  border-radius: 6px;
  background: #fff;
  color: var(--color-text-secondary);
  font-size: 0.82rem;
  cursor: pointer;
  transition: all var(--transition);
}

.btn-reset:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

/* 空状态 */
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
.empty-state svg { opacity: 0.5; }
.empty-state p { font-size: 0.95rem; color: var(--color-text-secondary); }
.empty-hint { font-size: 0.8rem; color: #94a3b8; }

/* 表格卡片 */
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

.record-count { font-size: 0.85rem; color: var(--color-text-secondary); }

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
