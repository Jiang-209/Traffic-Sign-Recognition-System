/**
 * API 请求模块
 * 封装与 FastAPI 后端的通信
 */
import axios from 'axios'

// 后端接口基础地址
// 开发环境：通过 Vite 代理转发 /api -> http://127.0.0.1:8000
// 生产环境：可改为实际后端地址
const BASE_URL = '/api'

// 创建 axios 实例，统一配置
export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,       // 30 秒超时
  headers: {
    'Accept': 'application/json',
  },
})

/**
 * 健康检查 - 确认后端服务是否在线
 * @returns {Promise<Object>} { status, model_loaded }
 */
export async function checkHealth() {
  const { data } = await apiClient.get('/health')
  return data
}

/**
 * 上传图片进行交通标志识别
 * @param {File} file - 图片文件对象
 * @param {string} mode - 识别模式: 'batch' | 'upload_roi' | 'camera_roi'
 * @param {Object|null} roiBox - ROI 边界框 { x1, y1, x2, y2 }（upload_roi / camera_roi 模式必需）
 * @returns {Promise<Object>} 识别结果
 *   { class_id, class_name, confidence, processing_time, reliable, mode, model_name }
 */
export async function predictImage(file, mode = 'batch', roiBox = null) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('mode', mode)

  if (roiBox) {
    formData.append('roi_x1', roiBox.x1)
    formData.append('roi_y1', roiBox.y1)
    formData.append('roi_x2', roiBox.x2)
    formData.append('roi_y2', roiBox.y2)
  }

  const { data } = await apiClient.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return data
}

/**
 * 提交用户反馈
 * 当模型识别错误时，用户可选择正确类别进行反馈
 * @param {Object} feedbackData - 反馈数据
 *   { image_name, predicted_class_id, predicted_class_name,
 *     correct_class_id, correct_class_name, confidence, remark }
 * @param {File} imageFile - 原始图片文件（可选）
 * @returns {Promise<Object>} { success, message }
 */
export async function submitFeedback(feedbackData, imageFile = null) {
  const formData = new FormData()

  // 将 JSON 数据作为字符串字段传递
  formData.append('feedback_data', JSON.stringify(feedbackData))

  // 如果有图片，一同上传
  if (imageFile) {
    formData.append('image', imageFile)
  }

  const { data } = await apiClient.post('/feedback', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return data
}
