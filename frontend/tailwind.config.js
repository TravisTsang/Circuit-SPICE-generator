/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#17211f",
        panel: "#f7f8f5",
        line: "#d9dfd8",
        accent: "#0f766e",
        signal: "#c2410c",
      },
      boxShadow: {
        tool: "0 18px 60px rgba(23, 33, 31, 0.12)",
      },
    },
  },
  plugins: [],
};
