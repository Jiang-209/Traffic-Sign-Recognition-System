<template>
  <div class="info-page">
    <div class="page-header">
      <h2 class="page-title">系统说明</h2>
      <p class="page-subtitle">使用说明、模型介绍与注意事项</p>
    </div>

    <div class="info-grid">
      <!-- 使用说明 -->
      <div class="info-card span-2">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
          </svg>
          使用说明
        </h3>
        <div class="info-text">
          <!-- TODO: 请替换为实际使用说明 -->
          <p>本系统基于卷积神经网络（CNN）实现中国交通标志的自动识别，支持单张图片识别、摄像头实时识别和批量识别三种模式。</p>
          <br/>
          <p><strong>图片识别：</strong>上传包含交通标志的图片，系统自动识别并返回最可能的类别及置信度。支持上传 JPG/PNG 格式图片，可选择 ROI 区域进行精确识别。</p>
          <br/>
          <p><strong>摄像头识别：</strong>使用电脑摄像头实时捕捉画面，自动检测并识别交通标志。</p>
          <br/>
          <p><strong>批量识别：</strong>一次上传多张图片，系统按顺序逐一识别，汇总结果展示。</p>
        </div>
      </div>

      <!-- 模型介绍 -->
      <div class="info-card">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
          </svg>
          TSRD Model
        </h3>
        <ul class="info-list">
          <!-- TODO: 请替换为实际模型说明 -->
          <li><strong>类型：</strong>全图 Resize 基线模型</li>
          <li><strong>输入：</strong>整张图片缩放到固定尺寸</li>
          <li><strong>适用场景：</strong>批量识别、常规图片识别</li>
          <li><strong>特点：</strong>处理速度快，适用于标志占比较大的图片</li>
        </ul>
      </div>

      <div class="info-card">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
          </svg>
          TSRD-ROI Model
        </h3>
        <ul class="info-list">
          <!-- TODO: 请替换为实际模型说明 -->
          <li><strong>类型：</strong>ROI 裁剪预训练模型</li>
          <li><strong>输入：</strong>用户框选的 ROI 区域（裁剪后缩放）</li>
          <li><strong>适用场景：</strong>图片框选识别、摄像头 ROI 识别</li>
          <li><strong>特点：</strong>对小型标志更敏感，减少背景干扰</li>
        </ul>
      </div>

      <!-- 数据说明 -->
      <div class="info-card">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
          </svg>
          数据集说明
        </h3>
        <ul class="info-list">
          <!-- TODO: 请替换为实际数据集说明 -->
          <li><strong>数据集：</strong>TSRD（Traffic Sign Recognition Dataset）</li>
          <li><strong>类别数：</strong>58 类中国交通标志</li>
          <li><strong>覆盖类型：</strong>禁令标志、指示标志、警告标志等</li>
        </ul>
      </div>

      <!-- 注意事项 -->
      <div class="info-card">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          注意事项
        </h3>
        <div class="info-text">
          <!-- TODO: 请替换为实际注意事项 -->
          <p>1. 本系统识别结果仅供参考，不应作为驾驶决策的唯一依据。</p>
          <br/>
          <p>2. 识别准确率受拍摄角度、光照条件、标志清晰度等因素影响。</p>
          <br/>
          <p>3. 摄像头识别模式需要浏览器授权摄像头权限。</p>
          <br/>
          <p>4. 批量识别时，文件数量过多可能会增加处理时间。</p>
          <br/>
          <p>5. 交通标志大全数据来源：北京市交通管理局官方网站。</p>
        </div>
      </div>

      <!-- 技术信息 -->
      <div class="info-card">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
          </svg>
          技术栈
        </h3>
        <ul class="info-list">
          <li><strong>后端：</strong>FastAPI + TensorFlow</li>
          <li><strong>前端：</strong>Vue 3 + Vite</li>
          <li><strong>模型架构：</strong>CNN 卷积神经网络</li>
        </ul>
      </div>

      <!-- 帮助反馈 -->
      <div class="info-card span-2">
        <h3 class="info-card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>
          </svg>
          帮助与反馈
        </h3>
        <div class="help-feedback">
          <p class="feedback-desc">遇到问题或有改进建议？请填写以下表单，我们会尽快处理。</p>
          <div class="feedback-form">
            <div class="fb-field">
              <label class="fb-label">问题描述 <span class="required">*</span></label>
              <textarea
                v-model="helpContent"
                class="fb-textarea"
                placeholder="请详细描述您遇到的问题或改进建议..."
                rows="3"
                maxlength="1000"
              ></textarea>
              <span class="fb-char-count">{{ helpContent.length }}/1000</span>
            </div>
            <div class="fb-row">
              <div class="fb-field fb-field-sm">
                <label class="fb-label">联系方式</label>
                <input v-model="helpContact" class="fb-input" placeholder="邮箱 / QQ / 微信（选填）" />
              </div>
              <div class="fb-field fb-field-sm">
                <label class="fb-label">所在页面</label>
                <select v-model="helpPage" class="fb-select">
                  <option value="">-- 请选择（选填） --</option>
                  <option value="图片识别">图片识别</option>
                  <option value="摄像头识别">摄像头识别</option>
                  <option value="批量识别">批量识别</option>
                  <option value="交通标志大全">交通标志大全</option>
                  <option value="历史记录">历史记录</option>
                  <option value="系统说明">系统说明</option>
                </select>
              </div>
            </div>
            <div class="fb-field">
              <label class="fb-label">截图（选填）</label>
              <div class="fb-upload" @click="triggerImageUpload">
                <span v-if="!helpImage">点击上传截图</span>
                <span v-else class="fb-upload-name">{{ helpImage.name }}</span>
                <button v-if="helpImage" type="button" class="fb-remove-img" @click.stop="helpImage = null">&times;</button>
              </div>
            </div>
            <div class="fb-actions">
              <span v-if="helpStatus" class="fb-status" :class="helpStatus.type">{{ helpStatus.text }}</span>
              <button class="fb-submit" :disabled="!helpContent.trim() || helpSubmitting" @click="submitHelp">
                {{ helpSubmitting ? '提交中...' : '提交反馈' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { apiClient } from '../api/predict.js'

const helpContent = ref('')
const helpContact = ref('')
const helpPage = ref('')
const helpImage = ref(null)
const helpSubmitting = ref(false)
const helpStatus = ref(null)

function triggerImageUpload() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.jpg,.jpeg,.png'
  input.onchange = (e) => {
    if (e.target.files[0]) helpImage.value = e.target.files[0]
  }
  input.click()
}

async function submitHelp() {
  if (!helpContent.value.trim() || helpSubmitting.value) return
  helpSubmitting.value = true
  helpStatus.value = null

  try {
    const form = new FormData()
    form.append('content', helpContent.value.trim())
    form.append('contact', helpContact.value)
    form.append('page', helpPage.value)
    if (helpImage.value) form.append('image', helpImage.value)

    await apiClient.post('/help-feedback', form)
    helpStatus.value = { type: 'success', text: '反馈已提交，谢谢！' }
    helpContent.value = ''
    helpContact.value = ''
    helpPage.value = ''
    helpImage.value = null
  } catch (e) {
    helpStatus.value = { type: 'error', text: '提交失败，请稍后重试' }
  } finally {
    helpSubmitting.value = false
  }
}
</script>

<style scoped>
.info-page { display: flex; flex-direction: column; gap: 20px; }
.page-header { margin-bottom: 2px; }
.page-title { font-size: 1.2rem; font-weight: 700; color: var(--color-text); }
.page-subtitle { font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 4px; }

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.info-card {
  background: var(--color-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 20px 24px;
}

.info-card.span-2 {
  grid-column: 1 / -1;
}

.info-card-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--color-text);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-primary-light);
  display: flex;
  align-items: center;
  gap: 8px;
}

.info-card-title svg {
  color: var(--color-primary);
  flex-shrink: 0;
}

.info-text {
  font-size: 0.88rem;
  color: var(--color-text-secondary);
  line-height: 1.7;
}

.info-text p {
  margin: 0;
}

.info-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
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

@media (max-width: 860px) {
  .info-grid { grid-template-columns: 1fr; }
}

/* 帮助反馈 */
.help-feedback { font-size: 0.88rem; }
.feedback-desc { color: var(--color-text-secondary); margin: 0 0 14px; line-height: 1.6; }

.feedback-form { display: flex; flex-direction: column; gap: 12px; }
.fb-field { display: flex; flex-direction: column; gap: 4px; position: relative; }
.fb-label { font-size: 0.8rem; font-weight: 600; color: var(--color-text); }
.required { color: var(--color-danger); }
.fb-textarea {
  padding: 8px 10px; border: 1.5px solid var(--color-border); border-radius: 6px;
  font-size: 0.88rem; font-family: inherit; color: var(--color-text);
  resize: vertical; min-height: 60px;
  transition: border-color var(--transition);
}
.fb-textarea:focus { outline: none; border-color: var(--color-primary); }
.fb-char-count { position: absolute; right: 6px; bottom: -18px; font-size: 0.72rem; color: #94a3b8; }

.fb-row { display: flex; gap: 12px; }
.fb-field-sm { flex: 1; }
.fb-input, .fb-select {
  padding: 7px 10px; border: 1.5px solid var(--color-border); border-radius: 6px;
  font-size: 0.85rem; font-family: inherit; color: var(--color-text); background: #fff;
  transition: border-color var(--transition);
}
.fb-input:focus, .fb-select:focus { outline: none; border-color: var(--color-primary); }

.fb-upload {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; border: 1.5px dashed var(--color-border); border-radius: 6px;
  cursor: pointer; font-size: 0.82rem; color: #94a3b8;
  transition: all var(--transition);
}
.fb-upload:hover { border-color: var(--color-primary); color: var(--color-primary); }
.fb-upload-name { color: var(--color-text); font-weight: 500; flex: 1; }
.fb-remove-img {
  background: none; border: none; color: var(--color-danger); font-size: 1.1rem;
  cursor: pointer; line-height: 1; padding: 0 2px;
}

.fb-actions { display: flex; align-items: center; gap: 12px; margin-top: 2px; }
.fb-submit {
  padding: 8px 22px; border: none; border-radius: 6px;
  background: var(--color-primary); color: #fff;
  font-size: 0.85rem; font-weight: 600; cursor: pointer;
  transition: background var(--transition);
}
.fb-submit:hover:not(:disabled) { background: var(--color-primary-dark); }
.fb-submit:disabled { opacity: 0.5; cursor: not-allowed; }

.fb-status { font-size: 0.82rem; padding: 4px 10px; border-radius: 4px; }
.fb-status.success { background: #f0fdf4; color: var(--color-success); }
.fb-status.error { background: #fef2f2; color: var(--color-danger); }
</style>
