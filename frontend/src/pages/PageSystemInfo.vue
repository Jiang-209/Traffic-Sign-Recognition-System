<template>
  <div class="info-page">
    <div class="page-header">
      <h2 class="page-title">系统说明</h2>
    </div>

    <div class="info-grid">
      <div class="info-card">
        <h3 class="info-card-title">关于本系统</h3>
        <p class="info-text">
          基于卷积神经网络（CNN）的中国交通标志识别系统，使用 TSRD 数据集进行训练，
          可识别 58 类常见中国交通标志。
        </p>
      </div>

      <div class="info-card">
        <h3 class="info-card-title">技术栈</h3>
        <ul class="info-list">
          <li><strong>后端：</strong>FastAPI + TensorFlow 1.x</li>
          <li><strong>前端：</strong>Vue 3 + Vite</li>
          <li><strong>模型：</strong>CNN 卷积神经网络</li>
          <li><strong>数据集：</strong>TSRD 58 类交通标志</li>
        </ul>
      </div>

      <div class="info-card">
        <h3 class="info-card-title">使用说明</h3>
        <ul class="info-list">
          <li><strong>图片识别：</strong>上传单张图片进行识别</li>
          <li><strong>摄像头识别：</strong>通过摄像头实时识别（开发中）</li>
          <li><strong>批量识别：</strong>上传多张图片批量处理（开发中）</li>
        </ul>
      </div>

      <div class="info-card">
        <h3 class="info-card-title">模型信息</h3>
        <ul class="info-list">
          <li><strong>TSRD Model：</strong>全图 Resize 基线模型，适用于批量识别</li>
          <li><strong>TSRD-ROI Model：</strong>ROI 裁剪预训练模型，适用于框选区域识别</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
.info-page {
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

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

@media (max-width: 860px) {
  .info-grid {
    grid-template-columns: 1fr;
  }
}

.info-card {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 20px 24px;
}

.info-card-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--color-text);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-primary-light);
}

.info-text {
  font-size: 0.88rem;
  color: var(--color-text-secondary);
  line-height: 1.7;
}

.info-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-list li {
  font-size: 0.88rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
  padding-left: 14px;
  position: relative;
}

.info-list li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 8px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--color-primary);
  opacity: 0.5;
}

.info-list li strong {
  color: var(--color-text);
  font-weight: 600;
}
</style>
