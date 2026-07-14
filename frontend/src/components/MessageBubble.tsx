import { splitCitations } from "../lib/citations";
import { CitationChip } from "./CitationChip";
import type { ChatMessage, Citation } from "../types";

// Bolla di messaggio. L'assistant rende il testo spezzato in segmenti con chip
// [n] cliccabili; l'utente rende testo semplice. Se la risposta è ancora vuota
// (streaming appena iniziato) mostra un cursore pulsante.
export function MessageBubble({ msg, onCite }: { msg: ChatMessage; onCite: (c: Citation) => void }) {
  const mine = msg.role === "user";
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"} animate-[fadeUp_.35s_ease-out]`}>
      <div
        className={
          "max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-[0.95rem] leading-relaxed shadow-sm " +
          (mine
            ? "rounded-tr-sm border border-violet/25 bg-violet/10 text-gray-100"
            : "rounded-tl-sm border border-white/5 bg-panel text-gray-200")
        }
      >
        {msg.role === "assistant" ? (
          msg.text === "" ? (
            <span className="inline-block h-4 w-[3px] animate-pulse rounded-full bg-accent/80 align-middle" />
          ) : (
            splitCitations(msg.text).map((s, i) =>
              s.type === "text" ? (
                <span key={i}>{s.value}</span>
              ) : (
                <CitationChip
                  key={i}
                  n={s.n}
                  onClick={() => {
                    const c = msg.citations?.find((x) => x.n === s.n);
                    if (c) onCite(c);
                  }}
                />
              ),
            )
          )
        ) : (
          msg.text
        )}
      </div>
    </div>
  );
}
