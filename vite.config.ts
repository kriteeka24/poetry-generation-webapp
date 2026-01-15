import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API requests to Flask backend at port 5000
    proxy: {
      '/api': 'http://localhost:5000'
    }
  }
})