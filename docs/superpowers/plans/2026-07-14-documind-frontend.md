# DocuMind Frontend — Implementation Plan (Piano 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`). Usare la skill `frontend-design` per la qualità visiva durante i task UI.

**Goal:** UI React per DocuMind — upload PDF, chat con risposta in streaming, e citazioni `[n]` cliccabili che aprono la pagina PDF citata evidenziando lo snippet.

**Architecture:** SPA Vite + React 18 + TypeScript + Tailwind. Consuma l'API backend (`/documents`, `/query` SSE) via proxy Vite. Il PDF per il viewer resta **lato client** (blob del File caricato, tenuto in `docStore`), renderizzato con **react-pdf**. Nessuna libreria di stato: hook + props.

**Tech Stack:** Vite · React 18 · TypeScript · TailwindCSS · react-pdf (pdfjs-dist) · Vitest + @testing-library/react.

## Global Constraints

- Cartella: `frontend/` nella root del repo `documind`.
- Node già presente (v24). Package manager: **npm**.
- API base via **proxy Vite** `/api → http://localhost:8000` (nessun CORS in dev). `api.ts` chiama `/api/...`.
- SSE: `EventSource` fa solo GET → si usa `fetch` + parsing manuale dello stream (`event:`/`data:`).
- PDF: il file originale NON è servito dal backend → il viewer usa il **blob in memoria** (scelta MVP). Il viewer mostra i documenti caricati nella sessione corrente.
- Tema **dark** che riecheggia la card: sfondo quasi-nero, accent verde + viola. Usare `frontend-design` per evitare estetica generica.
- TDD sui helper puri (parser SSE, parser citazioni). Componenti: test leggeri di rendering.
- Il backend deve girare in locale (`cd backend && uv run uvicorn app.main:app`) con `GEMINI_API_KEY` in `.env` (già presente) per provare `/query`.
- Branch dedicato `feat/frontend`. Commit frequenti, uno per task. Ogni commit termina con:
  `Co-Authored-By: claude-flow <ruv@ruv.net>`

---

### Task 1: Scaffold Vite + React + TS + Tailwind + Vitest

**Files:**
- Create: `frontend/` (via scaffold) + `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `src/index.css`, `src/setupTests.ts`
- Test: `frontend/src/smoke.test.ts`

**Interfaces:** nessuna (setup).

- [ ] **Step 1: Scaffold + dipendenze**

```bash
cd C:/Users/Rimes/Projects/documind
git checkout -b feat/frontend
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer vitest @testing-library/react @testing-library/jest-dom jsdom
npm install react-pdf
npx tailwindcss init -p
```

- [ ] **Step 2: Config Tailwind** — `tailwind.config.js`

```js
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
```

- [ ] **Step 3: `src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
body { @apply bg-ink text-gray-200; }
```

- [ ] **Step 4: Config Vitest + proxy** — aggiungi a `vite.config.ts`

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  test: { environment: "jsdom", setupFiles: "./src/setupTests.ts", globals: true },
});
```
`src/setupTests.ts`:
```ts
import "@testing-library/jest-dom";
```
In `package.json` scripts aggiungi: `"test": "vitest run"`.

- [ ] **Step 5: Smoke test** — `src/smoke.test.ts`

```ts
import { describe, it, expect } from "vitest";
describe("smoke", () => {
  it("somma", () => expect(1 + 1).toBe(2));
});
```

- [ ] **Step 6: Verifica**

Run: `npm run test`
Expected: 1 test PASS.
Run: `npm run build`
Expected: build senza errori TS.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Rimes/Projects/documind
git add frontend/
git commit -m "feat(frontend): scaffold Vite+React+TS+Tailwind+Vitest

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 2: Tipi + client API con parser SSE (TDD)

**Files:**
- Create: `frontend/src/types.ts`, `frontend/src/api.ts`
- Test: `frontend/src/api.test.ts`

**Interfaces:**
- Produces: `Citation {n,doc_name,page,snippet}`, `DocumentInfo {doc_id,doc_name,n_chunks,pages}`, `ChatMessage {role,text,citations}`.
- `parseSSE(chunk: string): {event:string,data:string}[]` — parser puro di blocchi SSE.
- `uploadDocument(file: File): Promise<DocumentInfo>`
- `streamQuery(question, {onToken, onCitations, onDone, onError}): Promise<void>`

- [ ] **Step 1: Test del parser SSE (fallisce)** — `src/api.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { parseSSE } from "./api";

describe("parseSSE", () => {
  it("estrae eventi token e citations", () => {
    const raw =
      'event: token\ndata: "Par"\n\n' +
      'event: token\ndata: "igi"\n\n' +
      'event: citations\ndata: [{"n":1,"doc_name":"a.pdf","page":1,"snippet":"x"}]\n\n' +
      "event: done\ndata: {}\n\n";
    const evs = parseSSE(raw);
    expect(evs.map((e) => e.event)).toEqual(["token", "token", "citations", "done"]);
    expect(JSON.parse(evs[0].data)).toBe("Par");
  });

  it("ignora blocchi incompleti", () => {
    expect(parseSSE("event: token\ndata: \"x\"")).toEqual([]); // niente \n\n finale
  });
});
```

- [ ] **Step 2: Run → FAIL** — `npm run test -- api` → `parseSSE` non definita.

- [ ] **Step 3: `src/types.ts`**

```ts
export interface Citation { n: number; doc_name: string; page: number; snippet: string; }
export interface DocumentInfo { doc_id: string; doc_name: string; n_chunks: number; pages: number; }
export interface ChatMessage { role: "user" | "assistant"; text: string; citations?: Citation[]; }
```

- [ ] **Step 4: `src/api.ts`**

```ts
import type { Citation, DocumentInfo } from "./types";

export function parseSSE(buffer: string): { event: string; data: string }[] {
  const out: { event: string; data: string }[] = [];
  for (const block of buffer.split("\n\n")) {
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

export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/documents", { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Upload fallito (${r.status})`);
  return r.json();
}

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
```

- [ ] **Step 5: Run → PASS** — `npm run test -- api` → 2 test verdi.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/api.ts frontend/src/api.test.ts
git commit -m "feat(frontend): tipi + client API con parser SSE (TDD)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 3: Parser citazioni inline (TDD) — segmenti testo/[n]

**Files:**
- Create: `frontend/src/lib/citations.ts`
- Test: `frontend/src/lib/citations.test.ts`

**Interfaces:**
- Produces: `splitCitations(text: string): Array<{type:"text",value:string} | {type:"cite",n:number}>` — spezza una risposta in segmenti di testo e marcatori `[n]`.

- [ ] **Step 1: Test (fallisce)** — `src/lib/citations.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { splitCitations } from "./citations";

describe("splitCitations", () => {
  it("separa testo e marcatori", () => {
    const seg = splitCitations("Parigi [1] e Roma [2].");
    expect(seg).toEqual([
      { type: "text", value: "Parigi " },
      { type: "cite", n: 1 },
      { type: "text", value: " e Roma " },
      { type: "cite", n: 2 },
      { type: "text", value: "." },
    ]);
  });
  it("testo senza citazioni", () => {
    expect(splitCitations("ciao")).toEqual([{ type: "text", value: "ciao" }]);
  });
});
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: `src/lib/citations.ts`**

```ts
export type Segment = { type: "text"; value: string } | { type: "cite"; n: number };

export function splitCitations(text: string): Segment[] {
  const segs: Segment[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) segs.push({ type: "text", value: text.slice(last, m.index) });
    segs.push({ type: "cite", n: Number(m[1]) });
    last = m.index + m[0].length;
  }
  if (last < text.length) segs.push({ type: "text", value: text.slice(last) });
  return segs;
}
```

- [ ] **Step 4: Run → PASS**.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/
git commit -m "feat(frontend): parser citazioni inline [n] (TDD)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 4: docStore (blob dei PDF) + hook useChat

**Files:**
- Create: `frontend/src/store/docStore.ts`, `frontend/src/hooks/useChat.ts`
- Test: `frontend/src/hooks/useChat.test.tsx`

**Interfaces:**
- `docStore`: `putFile(doc_name,file)`, `getFile(doc_name): File | undefined` (Map in modulo).
- `useChat()` → `{ messages, ask(question), busy, error }`; usa `streamQuery`.

- [ ] **Step 1: Test hook (fallisce)** — mock di `api.streamQuery`. `src/hooks/useChat.test.tsx`

```tsx
import { describe, it, expect, vi } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
vi.mock("../api", () => ({
  streamQuery: async (_q: string, cb: any) => {
    cb.onToken("Parigi "); cb.onToken("[1]");
    cb.onCitations([{ n: 1, doc_name: "a.pdf", page: 1, snippet: "x" }]);
    cb.onDone();
  },
}));
import { useChat } from "./useChat";

describe("useChat", () => {
  it("accumula token e citazioni", async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => { await result.current.ask("Capitale?"); });
    await waitFor(() => {
      const last = result.current.messages.at(-1)!;
      expect(last.role).toBe("assistant");
      expect(last.text).toContain("Parigi");
      expect(last.citations?.[0].page).toBe(1);
    });
  });
});
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: `src/store/docStore.ts`**

```ts
const files = new Map<string, File>();
export const docStore = {
  putFile: (name: string, f: File) => files.set(name, f),
  getFile: (name: string) => files.get(name),
};
```

- [ ] **Step 4: `src/hooks/useChat.ts`**

```ts
import { useState, useCallback } from "react";
import { streamQuery } from "../api";
import type { ChatMessage, Citation } from "../types";

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
```

- [ ] **Step 5: Run → PASS**.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/store/ frontend/src/hooks/
git commit -m "feat(frontend): docStore blob + hook useChat (streaming)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 5: Componenti chat — CitationChip, MessageBubble, ChatPanel, Uploader

**Files:**
- Create: `frontend/src/components/{CitationChip,MessageBubble,ChatPanel,Uploader}.tsx`
- Test: `frontend/src/components/MessageBubble.test.tsx`

**Interfaces:**
- `CitationChip({ n, onClick })` — bottone `[n]` viola.
- `MessageBubble({ msg, onCite })` — rende testo con chip via `splitCitations`.
- `ChatPanel({ messages, busy, onAsk, onCite })`.
- `Uploader({ onUploaded })` — drag-drop, chiama `uploadDocument` + `docStore.putFile`.

- [ ] **Step 1: Test MessageBubble (fallisce)** — `src/components/MessageBubble.test.tsx`

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MessageBubble } from "./MessageBubble";

describe("MessageBubble", () => {
  it("rende i chip [n] e chiama onCite", () => {
    const onCite = vi.fn();
    render(
      <MessageBubble
        msg={{ role: "assistant", text: "Parigi [1].", citations: [{ n: 1, doc_name: "a.pdf", page: 1, snippet: "x" }] }}
        onCite={onCite}
      />,
    );
    const chip = screen.getByRole("button", { name: "1" });
    fireEvent.click(chip);
    expect(onCite).toHaveBeenCalledWith(expect.objectContaining({ n: 1, page: 1 }));
  });
});
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Scrivi i componenti**

`CitationChip.tsx`:
```tsx
export function CitationChip({ n, onClick }: { n: number; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="mx-0.5 rounded bg-violet/20 px-1.5 text-xs font-semibold text-violet hover:bg-violet/40"
    >
      {n}
    </button>
  );
}
```

`MessageBubble.tsx`:
```tsx
import { splitCitations } from "../lib/citations";
import { CitationChip } from "./CitationChip";
import type { ChatMessage, Citation } from "../types";

export function MessageBubble({ msg, onCite }: { msg: ChatMessage; onCite: (c: Citation) => void }) {
  const mine = msg.role === "user";
  return (
    <div className={`my-2 flex ${mine ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] rounded-2xl px-4 py-2 ${mine ? "bg-violet/20" : "bg-panel"}`}>
        {msg.role === "assistant"
          ? splitCitations(msg.text).map((s, i) =>
              s.type === "text" ? (
                <span key={i}>{s.value}</span>
              ) : (
                <CitationChip key={i} n={s.n} onClick={() => {
                  const c = msg.citations?.find((x) => x.n === s.n);
                  if (c) onCite(c);
                }} />
              ),
            )
          : msg.text}
      </div>
    </div>
  );
}
```

`ChatPanel.tsx`:
```tsx
import { useState } from "react";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage, Citation } from "../types";

export function ChatPanel({ messages, busy, onAsk, onCite }: {
  messages: ChatMessage[]; busy: boolean; onAsk: (q: string) => void; onCite: (c: Citation) => void;
}) {
  const [q, setQ] = useState("");
  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((m, i) => <MessageBubble key={i} msg={m} onCite={onCite} />)}
        {busy && <div className="text-sm text-gray-500">…</div>}
      </div>
      <form
        className="flex gap-2 border-t border-white/10 p-3"
        onSubmit={(e) => { e.preventDefault(); onAsk(q); setQ(""); }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Fai una domanda sui documenti…"
          className="flex-1 rounded-lg bg-panel px-3 py-2 outline-none"
        />
        <button disabled={busy} className="rounded-lg bg-accent px-4 font-semibold text-ink disabled:opacity-40">
          Invia
        </button>
      </form>
    </div>
  );
}
```

`Uploader.tsx`:
```tsx
import { useState } from "react";
import { uploadDocument } from "../api";
import { docStore } from "../store/docStore";
import type { DocumentInfo } from "../types";

export function Uploader({ onUploaded }: { onUploaded: (d: DocumentInfo) => void }) {
  const [msg, setMsg] = useState("");
  const handle = async (file: File) => {
    setMsg(`Indicizzo ${file.name}…`);
    try {
      const info = await uploadDocument(file);
      docStore.putFile(info.doc_name, file);
      onUploaded(info);
      setMsg(`${info.doc_name}: ${info.n_chunks} chunk, ${info.pages} pagine`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Upload fallito");
    }
  };
  return (
    <label
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handle(f); }}
      className="block cursor-pointer rounded-lg border border-dashed border-white/20 p-4 text-center text-sm text-gray-400 hover:border-accent"
    >
      Trascina un PDF o clicca per caricarlo
      <input
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
      />
      {msg && <div className="mt-2 text-xs text-gray-500">{msg}</div>}
    </label>
  );
}
```

- [ ] **Step 4: Run → PASS** — `npm run test -- MessageBubble`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/
git commit -m "feat(frontend): componenti chat (CitationChip, MessageBubble, ChatPanel, Uploader)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 6: SourceViewer (react-pdf) con highlight dello snippet

**Files:**
- Create: `frontend/src/components/SourceViewer.tsx`
- Modify: `frontend/src/main.tsx` (worker pdfjs)

**Interfaces:**
- `SourceViewer({ citation })` — se `citation` è presente, carica il blob da `docStore`, mostra `citation.page`, evidenzia lo snippet.

- [ ] **Step 1: Worker pdfjs in `main.tsx`**

```tsx
import { pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();
```

- [ ] **Step 2: `SourceViewer.tsx`**

```tsx
import { Document, Page } from "react-pdf";
import { docStore } from "../store/docStore";
import type { Citation } from "../types";

export function SourceViewer({ citation }: { citation: Citation | null }) {
  if (!citation) {
    return <div className="grid h-full place-items-center text-gray-600">Clicca una citazione [n] per vedere la fonte</div>;
  }
  const file = docStore.getFile(citation.doc_name);
  if (!file) {
    return <div className="grid h-full place-items-center text-gray-500">PDF "{citation.doc_name}" non disponibile in questa sessione</div>;
  }
  const needle = citation.snippet.slice(0, 40).toLowerCase();
  return (
    <div className="h-full overflow-y-auto bg-panel p-3">
      <div className="mb-2 text-xs text-violet">{citation.doc_name} — pag. {citation.page}</div>
      <Document file={file}>
        <Page
          pageNumber={citation.page}
          width={520}
          customTextRenderer={({ str }) =>
            needle && str.toLowerCase().includes(needle.slice(0, 12))
              ? `<mark class="bg-accent/40">${str}</mark>`
              : str
          }
        />
      </Document>
    </div>
  );
}
```
> Nota: l'highlight è euristico (match del prefisso dello snippet sui text-item della pagina). Per l'MVP è sufficiente evidenziare la zona; aprire la pagina giusta è il requisito primario.

- [ ] **Step 3: Verifica build** — `npm run build` → nessun errore TS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/SourceViewer.tsx frontend/src/main.tsx
git commit -m "feat(frontend): SourceViewer react-pdf con highlight snippet

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 7: App layout + wiring + tema

**Files:**
- Modify: `frontend/src/App.tsx`

**Interfaces:** compone tutto.

- [ ] **Step 1: `App.tsx`**

```tsx
import { useState } from "react";
import { Uploader } from "./components/Uploader";
import { ChatPanel } from "./components/ChatPanel";
import { SourceViewer } from "./components/SourceViewer";
import { useChat } from "./hooks/useChat";
import type { Citation } from "./types";

export default function App() {
  const { messages, ask, busy, error } = useChat();
  const [active, setActive] = useState<Citation | null>(null);
  return (
    <div className="grid h-full grid-cols-2">
      <div className="flex flex-col border-r border-white/10">
        <header className="p-4">
          <div className="text-xs font-bold tracking-widest text-accent">DOCUMIND</div>
          <div className="text-lg font-bold">RAG multi-documento con citazioni</div>
        </header>
        <div className="px-4"><Uploader onUploaded={() => {}} /></div>
        {error && <div className="mx-4 mt-2 rounded bg-red-500/20 p-2 text-sm text-red-300">{error}</div>}
        <div className="min-h-0 flex-1"><ChatPanel messages={messages} busy={busy} onAsk={ask} onCite={setActive} /></div>
      </div>
      <SourceViewer citation={active} />
    </div>
  );
}
```

- [ ] **Step 2: Applicare `frontend-design`** per rifinire tema/tipografia/spacing (dark, accent verde+viola come la card), senza cambiare la logica.

- [ ] **Step 3: Verifica** — `npm run test` (tutti verdi) + `npm run build`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/**/*.css 2>/dev/null; git add frontend/src/App.tsx
git commit -m "feat(frontend): App layout (chat + source viewer) e tema dark

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 8: Verifica visiva end-to-end con Playwright

**Files:** nessuno (verifica manuale/assistita).

- [ ] **Step 1: Avvia backend + frontend**

```bash
# terminale A
cd backend && uv run uvicorn app.main:app
# terminale B
cd frontend && npm run dev    # http://localhost:5173
```

- [ ] **Step 2: Con Playwright (MCP)** naviga su `http://localhost:5173`, carica `backend/sample_docs/geografia.pdf`, chiedi "Qual è la capitale della Francia?", verifica: la risposta streamma, compare `[1]`, il click apre `geografia.pdf` pag.1 nel viewer. Cattura screenshot.

- [ ] **Step 3:** Se emergono bug visivi/funzionali, correggi e ricommetti.

---

### Task 9: README (sezione frontend + GIF) e chiusura

**Files:** Modify `README.md`

- [ ] **Step 1:** Aggiorna il README: sezione "Frontend" (stack, `npm run dev`, proxy verso backend), sostituisci la riga "frontend planned" nella Roadmap, aggiungi screenshot/GIF della demo.
- [ ] **Step 2:** `npm run test` + `npm run build` verdi.
- [ ] **Step 3: Commit**

```bash
git add README.md frontend/
git commit -m "docs: README sezione frontend + demo

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

- [ ] **Step 4:** `superpowers:finishing-a-development-branch` → merge `feat/frontend` su `main`, push.

---

## Self-Review

**1. Spec coverage (§7 del design):** Uploader/ChatPanel/MessageBubble/CitationChip/SourceViewer → Task 5-6; useChat SSE → Task 4; citazioni cliccabili con highlight+scroll → Task 3+5+6; tema dark → Task 1+7; blob client-side → Task 4 (docStore) + Task 6. ✓

**2. Placeholder scan:** nessun TODO/placeholder; ogni step ha codice completo. ✓

**3. Type consistency:** `Citation`/`DocumentInfo`/`ChatMessage` da `types.ts` usati coerentemente; `streamQuery` callback (`onToken/onCitations/onDone/onError`) coerenti tra `api.ts` (Task 2) e `useChat` (Task 4); `splitCitations`/`Segment` coerenti tra Task 3 e `MessageBubble` (Task 5); `docStore.putFile/getFile` coerenti tra Task 4 e Task 6. ✓

**Rischi noti:** versione worker `pdfjs-dist` deve combaciare con quella richiesta da `react-pdf` (se mismatch, allineare la versione in package.json); highlight euristico (accettato per MVP).
