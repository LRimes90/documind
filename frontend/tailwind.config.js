/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0a0a0f",       // sfondo quasi-nero
        panel: "#14141c",
        accent: "#34d399",    // verde (sistema / label)
        violet: "#a78bfa",    // viola (citazioni / riferimenti)
      },
      fontFamily: {
        // font-sans di default (corpo/UI): IBM Plex Sans (umanista, tecnico)
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        // font-mono (metadati, label, chip, coordinate): IBM Plex Mono
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
        // font-display (wordmark, titoli): Instrument Serif (editoriale)
        display: ['"Instrument Serif"', "ui-serif", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
