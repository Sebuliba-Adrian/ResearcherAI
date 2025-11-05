/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Cyber blue palette
        cyber: {
          50: '#e6f1ff',
          100: '#b3d9ff',
          200: '#80c1ff',
          300: '#4da9ff',
          400: '#1a91ff',
          500: '#0070f3',
          600: '#0057c2',
          700: '#003f91',
          800: '#002760',
          900: '#001a40',
        },
        // Purple palette
        neon: {
          purple: '#a855f7',
          pink: '#ec4899',
          blue: '#3b82f6',
          cyan: '#06b6d4',
          green: '#10b981',
          yellow: '#fbbf24',
        },
        // Glassmorphism backgrounds
        glass: {
          white: 'rgba(255, 255, 255, 0.1)',
          black: 'rgba(0, 0, 0, 0.3)',
          dark: 'rgba(15, 23, 42, 0.6)',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-cyber': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-neon': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'gradient-blue': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'gradient-purple': 'linear-gradient(135deg, #a855f7 0%, #ec4899 100%)',
        'gradient-dark': 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'float-delayed': 'float 8s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'gradient': 'gradient 15s ease infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'shimmer': 'shimmer 2s linear infinite',
        'blob': 'blob 7s infinite',
        'spin-slow': 'spin 20s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center',
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center',
          },
        },
        glow: {
          '0%': {
            'box-shadow': '0 0 5px rgba(168, 85, 247, 0.5), 0 0 20px rgba(168, 85, 247, 0.3)',
          },
          '100%': {
            'box-shadow': '0 0 20px rgba(168, 85, 247, 0.8), 0 0 40px rgba(168, 85, 247, 0.5)',
          },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        blob: {
          '0%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
          '33%': {
            transform: 'translate(30px, -50px) scale(1.1)',
          },
          '66%': {
            transform: 'translate(-20px, 20px) scale(0.9)',
          },
          '100%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
        },
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        'glass-sm': '0 4px 16px 0 rgba(31, 38, 135, 0.2)',
        'glass-lg': '0 16px 64px 0 rgba(31, 38, 135, 0.5)',
        'neon': '0 0 10px rgba(168, 85, 247, 0.5)',
        'neon-lg': '0 0 20px rgba(168, 85, 247, 0.8)',
      },
      borderRadius: {
        'glass': '16px',
      },
    },
  },
  plugins: [],
}
