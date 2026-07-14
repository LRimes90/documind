/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0a0a0f",       // sfondo quasi-nero
        panel: "#14141c",
        accent: "#34d399",    // verde (label)
        violet: "#a78bfa",    // viola (citazioni/tag)
      },
    },
  },
  plugins: [],
};
