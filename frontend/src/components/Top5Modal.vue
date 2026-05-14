<template>
  <Transition name="modal-fade">
    <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
      <Transition name="modal-slide" appear>
        <div v-if="visible" class="modal-panel">
          <div class="modal-header">
            <h3 class="modal-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Top-5 预测结果
            </h3>
            <button class="modal-close" @click="$emit('close')">&times;</button>
          </div>

          <div class="modal-body">
            <div class="file-info" v-if="filename">
              <span class="file-label">图片：</span>
              <span class="file-value">{{ filename }}</span>
            </div>

            <div class="top5-list">
              <div
                v-for="(item, index) in top5"
                :key="item.class_id"
                class="top5-item"
                :class="{ 'is-top1': index === 0 }"
              >
                <span class="top5-rank">{{ index + 1 }}</span>
                <div class="top5-bar-track">
                  <div
                    class="top5-bar-fill"
                    :class="barLevel(item.confidence)"
                    :style="{ width: (item.confidence * 100) + '%' }"
                  ></div>
                </div>
                <span class="top5-name">{{ item.class_name }}</span>
                <span class="top5-id">#{{ item.class_id }}</span>
                <span class="top5-conf">{{ (item.confidence * 100).toFixed(1) }}%</span>
              </div>
            </div>

            <!-- 无 top5 数据时的提示 -->
            <div v-if="!top5 || top5.length === 0" class="empty-top5">
              <p>暂无 Top-5 数据</p>
            </div>
          </div>
        </div>
      </Transition>
    </div>
  </Transition>
</template>

<script setup>
defineProps({
  visible: { type: Boolean, default: false },
  top5: { type: Array, default: () => [] },
  filename: { type: String, default: '' },
})

defineEmits(['close'])

function barLevel(confidence) {
  if (confidence >= 0.9) return 'level-high'
  if (confidence >= 0.7) return 'level-mid'
  return 'level-low'
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-panel {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
  width: 420px;
  max-width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 20px 12px;
  border-bottom: 1px solid var(--color-border);
}

.modal-title {
  font-size: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
}

.modal-close {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: none;
  background: #f1f5f9;
  color: var(--color-text-secondary);
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}

.modal-close:hover {
  background: var(--color-danger);
  color: #fff;
}

.modal-body {
  padding: 16px 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.file-info {
  font-size: 0.82rem;
  color: var(--color-text-secondary);
}

.file-label { color: #94a3b8; }
.file-value { color: var(--color-text); font-weight: 500; }

.top5-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.top5-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 8px;
  background: #f8fafc;
  transition: background var(--transition);
}

.top5-item.is-top1 {
  background: var(--color-primary-light);
  border: 1px solid var(--color-primary);
}

.top5-rank {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #e2e8f0;
  color: var(--color-text-secondary);
  font-size: 0.72rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.is-top1 .top5-rank {
  background: var(--color-primary);
  color: #fff;
}

.top5-bar-track {
  flex: 1;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  min-width: 80px;
}

.top5-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}

.top5-bar-fill.level-high { background: var(--color-success); }
.top5-bar-fill.level-mid { background: var(--color-primary); }
.top5-bar-fill.level-low { background: var(--color-warning); }

.top5-name {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--color-text);
  min-width: 60px;
}

.top5-id {
  font-size: 0.72rem;
  color: #94a3b8;
}

.top5-conf {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--color-text);
  min-width: 48px;
  text-align: right;
}

.empty-top5 {
  text-align: center;
  color: #94a3b8;
  font-size: 0.85rem;
  padding: 20px;
}

/* 模态框动画 */
.modal-fade-enter-active,
.modal-fade-leave-active { transition: opacity 0.25s ease; }
.modal-fade-enter-from,
.modal-fade-leave-to { opacity: 0; }

.modal-slide-enter-active { transition: all 0.25s ease-out; }
.modal-slide-leave-active { transition: all 0.15s ease-in; }
.modal-slide-enter-from { opacity: 0; transform: translateY(-20px) scale(0.96); }
.modal-slide-leave-to { opacity: 0; transform: translateY(-10px) scale(0.98); }
</style>
