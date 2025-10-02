import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Optimized for ML dashboard with heavy dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      '@mui/material',
      '@mui/icons-material',
      '@emotion/react',
      '@emotion/styled',
      'three',
      'recharts',
      'axios'
    ]
  },
  
  // Build configuration optimized for production
  build: {
    target: 'esnext',
    minify: 'esbuild',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate chunks for better caching
          'react-vendor': ['react', 'react-dom'],
          'mui-vendor': ['@mui/material', '@mui/icons-material', '@emotion/react', '@emotion/styled'],
          'three-vendor': ['three'],
          'charts-vendor': ['recharts']
        }
      }
    },
    // Increase chunk size warning limit for ML libraries
    chunkSizeWarningLimit: 1000
  },
  
  // Development server configuration
  server: {
    port: 3000,
    open: true,
    cors: true,
    // Proxy API calls to backend
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: 'ws://localhost:8002',
        ws: true,
        changeOrigin: true
      }
    }
  },
  
  // Path resolution
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@services': resolve(__dirname, 'src/services'),
      '@types': resolve(__dirname, 'src/types')
    }
  },
  
  // Environment variables
  envPrefix: 'VITE_',
  
  // ESBuild configuration for better performance
  esbuild: {
    target: 'esnext',
    format: 'esm'
  }
})
