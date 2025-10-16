import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'

export default defineConfig({
  plugins: [solid()],
  server: {
    proxy: {
      // Proxy WebSocket to NBA_API
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
        changeOrigin: true
      }
    }
  },
  build: {
    // Optimize for production
    minify: 'terser',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['solid-js']
        }
      }
    }
  }
})
