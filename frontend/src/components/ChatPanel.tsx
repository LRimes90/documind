import { useState, useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage, Citation } from "../types";

// Pannello conversazione: lista messaggi con auto-scroll, indicatore di
// generazione e composer. La logica di invio è delegata al parent via onAsk.
export function ChatPanel({ messages, busy, onAsk, onCite }: {
  messages: ChatMessage[]; busy: boolean; onAsk: (q: string) => void; onCite: (c: Citation) => void;
}) {
  const [q, setQ] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  // Scrolla in fondo a ogni nuovo token/messaggio.
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="grid h-full place-items-center px-6 text-center">
            <div className="max-w-xs">
              <div className="mb-2 font-mono text-[0.7rem] uppercase tracking-[0.2em] text-accent">Pronto</div>
              <p className="text-sm text-gray-500">
                Carica un PDF e fai una domanda. Le risposte citano la fonte con marcatori
                <span className="mx-1 rounded bg-violet/10 px-1 font-mono text-xs text-violet">[n]</span>
                cliccabili.
              </p>
            </div>
          </div>
        )}
        {messages.map((m, i) => <MessageBubble key={i} msg={m} onCite={onCite} />)}
        {busy && (
          <div className="flex items-center gap-1.5 pl-1 text-gray-500">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent" />
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form
        className="flex gap-2 border-t border-white/10 bg-ink/60 p-3 backdrop-blur"
        onSubmit={(e) => { e.preventDefault(); onAsk(q); setQ(""); }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Fai una domanda sui documenti…"
          className="flex-1 rounded-lg border border-white/10 bg-panel px-3.5 py-2.5 text-sm text-gray-100 outline-none transition-colors placeholder:text-gray-600 focus:border-accent/50"
        />
        <button
          disabled={busy}
          className="rounded-lg bg-accent px-4 text-sm font-semibold text-ink transition-all hover:brightness-110 active:scale-95 disabled:opacity-40"
        >
          Invia
        </button>
      </form>
    </div>
  );
}
