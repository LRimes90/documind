import { useState } from "react";
import { Uploader } from "./components/Uploader";
import { ChatPanel } from "./components/ChatPanel";
import { SourceViewer } from "./components/SourceViewer";
import { useChat } from "./hooks/useChat";
import type { Citation, DocumentInfo } from "./types";

// Layout a due colonne: a sinistra brand + upload + chat, a destra il viewer
// della fonte citata. Lo stato `active` (citazione cliccata) collega le due.
export default function App() {
  const { messages, ask, busy, error } = useChat();
  const [active, setActive] = useState<Citation | null>(null);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);

  return (
    <div className="grid h-full grid-cols-1 lg:grid-cols-2">
      {/* Colonna sinistra */}
      <div className="flex min-h-0 flex-col border-r border-white/10">
        <header className="border-b border-white/5 px-6 py-5">
          <div className="font-mono text-[0.7rem] uppercase tracking-[0.35em] text-accent">DocuMind</div>
          <h1 className="mt-1.5 font-display text-[2rem] leading-none text-gray-100">
            RAG multi-documento<span className="text-violet">.</span>
          </h1>
          <p className="mt-2 text-sm text-gray-500">Risposte con citazioni verificabili alla fonte.</p>
        </header>

        <div className="space-y-2.5 px-6 pt-4">
          <Uploader
            onUploaded={(d) => setDocs((prev) => [...prev.filter((x) => x.doc_id !== d.doc_id), d])}
          />
          {docs.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {docs.map((d) => (
                <span
                  key={d.doc_id}
                  className="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[0.7rem] text-gray-400"
                >
                  {d.doc_name}<span className="text-gray-600"> · {d.pages}p</span>
                </span>
              ))}
            </div>
          )}
        </div>

        {error && (
          <div className="mx-6 mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="mt-3 min-h-0 flex-1">
          <ChatPanel messages={messages} busy={busy} onAsk={ask} onCite={setActive} />
        </div>
      </div>

      {/* Colonna destra */}
      <div className="hidden min-h-0 lg:block">
        <SourceViewer citation={active} />
      </div>
    </div>
  );
}
