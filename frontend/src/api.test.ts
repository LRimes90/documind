import { describe, it, expect, vi, afterEach } from "vitest";
import { parseSSE, streamQuery } from "./api";

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
    expect(parseSSE('event: token\ndata: "x"')).toEqual([]); // niente \n\n finale
  });
});

// Gli eventi SSE vengono composti da un helper: le newline non compaiono come
// escape nel sorgente, cosi' i test restano leggibili e non ambigui.
const NL = String.fromCharCode(10);
const sse = (event: string, data: string) => `event: ${event}${NL}data: ${data}${NL}${NL}`;

/** Monta un fetch finto che restituisce i chunk dati come corpo in streaming. */
function stubStream(chunks: string[]) {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
}

function spies() {
  return { onToken: vi.fn(), onCitations: vi.fn(), onDone: vi.fn(), onError: vi.fn() };
}

describe("streamQuery - chiusura dello stream", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("inoltra event: error a onError e non segnala successo", async () => {
    stubStream([
      sse("token", JSON.stringify("La capitale ")),
      sse("error", JSON.stringify({ detail: "Il modello non risponde." })),
    ]);
    const cb = spies();
    await streamQuery("q", cb);
    expect(cb.onToken).toHaveBeenCalledWith("La capitale ");
    expect(cb.onError).toHaveBeenCalledWith("Il modello non risponde.");
    expect(cb.onDone).not.toHaveBeenCalled();
  });

  it("segnala errore se lo stream si chiude senza evento terminale", async () => {
    // Il caso reale: il provider muore, la connessione cade, nessun done ne error.
    stubStream([sse("token", JSON.stringify("La capitale "))]);
    const cb = spies();
    await streamQuery("q", cb);
    expect(cb.onDone).not.toHaveBeenCalled();
    expect(cb.onError).toHaveBeenCalledTimes(1);
    expect(cb.onError.mock.calls[0][0]).toMatch(/interrotta/i);
  });

  it("uno stream completo chiama onDone senza errori", async () => {
    stubStream([
      sse("token", JSON.stringify("Parigi")),
      sse("citations", JSON.stringify([{ n: 1, doc_name: "a.pdf", page: 1, snippet: "x" }])),
      sse("done", "{}"),
    ]);
    const cb = spies();
    await streamQuery("q", cb);
    expect(cb.onDone).toHaveBeenCalledTimes(1);
    expect(cb.onError).not.toHaveBeenCalled();
  });
});
