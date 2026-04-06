import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0c1320",
        surface: "#0c1320",
        "surface-container": "#19202d",
        "surface-container-high": "#232a38",
        "surface-container-highest": "#2e3543",
        "on-surface": "#dce2f5",
        "on-surface-variant": "#bbcbb2",
        primary: "#3de530",
        "primary-container": "#00c805",
        secondary: "#ffb3ae",
        "secondary-container": "#a00118",
        outline: "#86957e",
        "outline-variant": "#3d4b37",
      },
      fontFamily: {
        sans: ['Manrope', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg: "0.25rem",
        xl: "0.5rem",
        full: "0.75rem"
      },
    },
  },
  plugins: [],
} satisfies Config
