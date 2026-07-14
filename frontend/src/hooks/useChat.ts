import { useState, useCallback } from "react";
import { streamQuery } from "../api";
import type { ChatMessage, Citation } from "../types";

/**
 * Hook di chat: mantiene la cronologia messaggi e gestisce una query in
 * streaming. Al momento della domanda inserisce il messaggio utente + un
 * messaggio assistant vuoto, poi accumula token e citazioni sull'ultimo.
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = useCallback(async (question: string) => {
    if (!question.trim() || busy) return;
    setError(null);
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }, { role: "assistant", text: "" }]);
    const patchLast = (fn: (msg: ChatMessage) => ChatMessage) =>
      setMessages((m) => m.map((msg, i) => (i === m.length - 1 ? fn(msg) : msg)));
    await streamQuery(question, {
      onToken: (t) => patchLast((msg) => ({ ...msg, text: msg.text + t })),
      onCitations: (c: Citation[]) => patchLast((msg) => ({ ...msg, citations: c })),
      onDone: () => setBusy(false),
      onError: (e) => { setError(e); setBusy(false); },
    });
  }, [busy]);

  return { messages, ask, busy, error };
}
