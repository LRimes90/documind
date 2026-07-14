import type { Citation, DocumentInfo } from "./types";

/**
 * Parser puro di blocchi SSE (`event:`/`data:` separati da riga vuota).
 *
 * Semantica SSE: un evento è completo solo quando terminato da `\n\n`.
 * `split("\n\n")` produce come ultimo elemento il residuo che segue l'ultimo
 * terminatore (stringa vuota se il buffer finiva con `\n\n`, oppure un blocco
 * ancora incompleto): lo si scarta con `pop()` e si parsano solo i blocchi
 * effettivamente chiusi. `streamQuery` passa sempre buffer terminati da `\n\n`.
 */
export function parseSSE(buffer: string): { event: string; data: string }[] {
  const out: { event: string; data: string }[] = [];
  const blocks = buffer.split("\n\n");
  blocks.pop(); // residuo incompleto dopo l'ultimo \n\n → non è un evento chiuso
  for (const block of blocks) {
    if (!block.trim()) continue;
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) out.push({ event, data });
  }
  return out;
}

/** Carica un PDF sul backend e ritorna i metadati di indicizzazione. */
export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/documents", { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Upload fallito (${r.status})`);
  return r.json();
}

/**
 * Esegue una query in streaming SSE via `fetch` (EventSource fa solo GET).
 * Legge lo stream, bufferizza fino all'ultimo `\n\n` e inoltra ogni evento
 * completo alle callback (`token`/`citations`/`done`).
 */
export async function streamQuery(
  question: string,
  cb: {
    onToken: (t: string) => void;
    onCitations: (c: Citation[]) => void;
    onDone: () => void;
    onError: (msg: string) => void;
  },
): Promise<void> {
  const r = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!r.ok || !r.body) {
    cb.onError((await r.json().catch(() => ({}))).detail || `Errore query (${r.status})`);
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const idx = buf.lastIndexOf("\n\n");
    if (idx === -1) continue;
    const ready = buf.slice(0, idx + 2);
    buf = buf.slice(idx + 2);
    for (const ev of parseSSE(ready)) {
      if (ev.event === "token") cb.onToken(JSON.parse(ev.data));
      else if (ev.event === "citations") cb.onCitations(JSON.parse(ev.data));
      else if (ev.event === "done") cb.onDone();
    }
  }
}
