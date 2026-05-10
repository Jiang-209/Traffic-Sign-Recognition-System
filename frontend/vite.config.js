import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],

  // 开发服务器配置
  server: {
    port: 5173,
    host: '127.0.0.1',

    // 代理配置：将 /api 请求转发到 FastAPI 后端
    // 这样前端无需处理跨域问题（开发时）
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
