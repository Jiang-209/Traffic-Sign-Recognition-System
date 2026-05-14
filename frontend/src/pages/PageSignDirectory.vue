<template>
  <div class="directory-page">
    <div class="page-header">
      <h2 class="page-title">交通标志大全</h2>
      <p class="page-subtitle">中国道路交通标志 —— 基于 GB 5768.2-2022</p>
    </div>

    <!-- 搜索栏 -->
    <div class="search-bar">
      <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <input
        v-model="searchQuery"
        class="search-input"
        placeholder="搜索标志名称..."
      />
      <span v-if="searchQuery" class="search-clear" @click="searchQuery = ''">&times;</span>
      <span class="search-count">共 {{ filteredSigns.length }} 个</span>
    </div>

    <!-- 加载 / 错误 -->
    <div v-if="loading" class="loading-state">
      <div class="spinner"></div><span>加载中...</span>
    </div>
    <div v-else-if="error" class="error-state">{{ error }}</div>

    <!-- 分类分组 -->
    <template v-else>
      <div
        v-for="group in groupedSigns"
        :key="group.category"
        class="category-section"
      >
        <div class="category-header" @click="toggleCategory(group.category)">
          <svg
            class="collapse-arrow"
            :class="{ expanded: !collapsed[group.category] }"
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
          ><polyline points="9 18 15 12 9 6"/></svg>
          <h3 class="category-title">{{ group.category }}</h3>
          <span class="category-count">{{ group.items.length }} 个</span>
        </div>

        <Transition name="collapse">
          <div v-if="!collapsed[group.category]" class="sign-grid">
            <div
              v-for="sign in group.items"
              :key="sign.category + sign.name"
              class="sign-card"
            >
              <div class="sign-img-wrapper">
                <img
                  class="sign-img"
                  :src="signImageUrl(sign)"
                  :alt="sign.name"
                  @error="onImgError($event)"
                />
                <span v-if="sign.in_tsrd" class="badge-tsrd" title="模型支持识别">TSRD</span>
              </div>
              <div class="sign-info">
                <span class="sign-name">{{ sign.name }}</span>
                <p class="sign-desc">{{ sign.description }}</p>
              </div>
            </div>
          </div>
        </Transition>
      </div>

      <div v-if="filteredSigns.length === 0" class="empty-state">
        <p>没有匹配的标志</p>
        <span class="empty-hint">尝试其他搜索词</span>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { apiClient } from '../api/predict.js'

const signs = ref([])
const loading = ref(true)
const error = ref(null)
const searchQuery = ref('')
const collapsed = ref({})

onMounted(async () => {
  try {
    const { data } = await apiClient.get('/signs-data')
    signs.value = data
    // 默认展开第一个分类
    const cats = [...new Set(data.map((s) => s.category))]
    cats.forEach((c) => (collapsed.value[c] = false))
  } catch (e) {
    error.value = e.message || '加载失败'
  } finally {
    loading.value = false
  }
})

function toggleCategory(cat) {
  collapsed.value[cat] = !collapsed.value[cat]
}

function signImageUrl(sign) {
  return `/api/signs-images/${encodeURIComponent(sign.image_file)}`
}

function onImgError(e) {
  e.target.style.display = 'none'
  e.target.parentNode.classList.add('no-image')
}

const filteredSigns = computed(() => {
  if (!searchQuery.value) return signs.value
  const q = searchQuery.value.toLowerCase()
  return signs.value.filter(
    (s) => s.name.toLowerCase().includes(q) || s.category.toLowerCase().includes(q)
  )
})

const groupedSigns = computed(() => {
  const map = {}
  for (const s of filteredSigns.value) {
    if (!map[s.category]) map[s.category] = []
    map[s.category].push(s)
  }
  return Object.entries(map).map(([category, items]) => ({ category, items }))
})
</script>

<style scoped>
.directory-page { display: flex; flex-direction: column; gap: 16px; }
.page-header { margin-bottom: 2px; }
.page-title { font-size: 1.2rem; font-weight: 700; color: var(--color-text); }
.page-subtitle { font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 4px; }

/* 搜索栏 */
.search-bar {
  display: flex; align-items: center; gap: 8px;
  background: var(--color-card); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 10px 16px;
  border: 1.5px solid var(--color-border);
  transition: border-color var(--transition);
}
.search-bar:focus-within { border-color: var(--color-primary); }
.search-icon { color: #94a3b8; flex-shrink: 0; }
.search-input {
  flex: 1; border: none; outline: none; font-size: 0.88rem; font-family: inherit;
  color: var(--color-text); background: transparent;
}
.search-clear {
  font-size: 1.2rem; cursor: pointer; color: #94a3b8; line-height: 1;
}
.search-clear:hover { color: var(--color-text); }
.search-count { font-size: 0.78rem; color: #94a3b8; white-space: nowrap; }

.loading-state, .error-state, .empty-state {
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  padding: 60px 20px; background: var(--color-card); border-radius: var(--radius);
  box-shadow: var(--shadow); font-size: 0.9rem; color: var(--color-text-secondary);
}
.error-state { color: var(--color-danger); }
.empty-hint { font-size: 0.8rem; color: #94a3b8; }
.spinner { width: 20px; height: 20px; border: 2px solid var(--color-border); border-top-color: var(--color-primary); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* 分类 */
.category-section {
  background: var(--color-card); border-radius: var(--radius); box-shadow: var(--shadow);
  overflow: hidden;
}
.category-header {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 20px; cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--color-border);
  transition: background var(--transition);
}
.category-header:hover { background: var(--color-hover); }
.collapse-arrow { color: #94a3b8; transition: transform 0.2s; flex-shrink: 0; }
.collapse-arrow.expanded { transform: rotate(90deg); }
.category-title { font-size: 1rem; font-weight: 700; color: var(--color-text); flex: 1; }
.category-count { font-size: 0.78rem; color: #94a3b8; background: #f1f5f9; padding: 2px 10px; border-radius: 10px; }

/* 标志网格 */
.sign-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px; padding: 16px 20px;
}
.sign-card {
  display: flex; gap: 12px; padding: 12px;
  border: 1px solid var(--color-border); border-radius: 8px;
  transition: all var(--transition);
}
.sign-card:hover { border-color: var(--color-primary); box-shadow: var(--shadow-sm); }
.sign-img-wrapper {
  width: 56px; height: 56px; flex-shrink: 0;
  border-radius: 6px; overflow: hidden;
  background: #f8fafc; display: flex; align-items: center; justify-content: center;
  position: relative;
}
.sign-img-wrapper.no-image::after {
  content: '🪧'; font-size: 1.5rem; opacity: 0.3;
}
.sign-img { width: 100%; height: 100%; object-fit: contain; }
.badge-tsrd {
  position: absolute; top: -2px; right: -2px;
  font-size: 0.6rem; font-weight: 700;
  background: var(--color-primary); color: #fff;
  padding: 1px 5px; border-radius: 4px; line-height: 1.3;
}
.sign-info { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.sign-name { font-size: 0.85rem; font-weight: 600; color: var(--color-text); line-height: 1.3; }
.sign-desc {
  font-size: 0.76rem; color: var(--color-text-secondary); line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}

/* 折叠动画 */
.collapse-enter-active, .collapse-leave-active { transition: all 0.25s ease; overflow: hidden; }
.collapse-enter-from, .collapse-leave-to { opacity: 0; max-height: 0; padding: 0 20px; }
.collapse-enter-to, .collapse-leave-from { opacity: 1; max-height: 2000px; padding: 16px 20px; }

@media (max-width: 700px) {
  .sign-grid { grid-template-columns: 1fr; }
}
</style>
