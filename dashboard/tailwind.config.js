/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'f1-bg': '#0b0e14',
        'f1-panel': '#121722',
        'f1-border': '#1e2533',
        'f1-text': '#e6e9f2',
        'f1-muted': '#a0a7b8',
        'f1-accent': '#00d1ff',
        'f1-accent-2': '#ff7a1a',
        'f1-good': '#22c55e',
        'f1-bad': '#ef4444',
      },
      fontFamily: {
        'inter': ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'f1': '0 10px 30px rgba(0,0,0,0.35)',
      },
      borderRadius: {
        'xl': '16px',
        '2xl': '20px',
      }
    },
  },
  plugins: [],
}
