<template>
  <div class="placeholder-page">
    <div class="placeholder-card">
      <div class="placeholder-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
      </div>
      <h2 class="placeholder-title">批量识别</h2>
      <p class="placeholder-desc">上传多张图片进行批量交通标志识别</p>
      <div class="placeholder-badge">开发中</div>
    </div>
  </div>
</template>

<style scoped>
.placeholder-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 500px;
}

.placeholder-card {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 60px 40px;
  text-align: center;
  max-width: 420px;
  width: 100%;
}

.placeholder-icon {
  color: var(--color-primary);
  opacity: 0.6;
  margin-bottom: 20px;
}

.placeholder-title {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--color-text);
  margin-bottom: 8px;
}

.placeholder-desc {
  font-size: 0.9rem;
  color: var(--color-text-secondary);
  margin-bottom: 20px;
}

.placeholder-badge {
  display: inline-block;
  padding: 4px 16px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
}
</style>
