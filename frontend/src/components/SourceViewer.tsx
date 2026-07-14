import { Document, Page } from "react-pdf";
import { docStore } from "../store/docStore";
import type { Citation } from "../types";

/**
 * Decide se un frammento di testo della pagina PDF appartiene allo snippet
 * citato dal backend, e quindi va evidenziato nel viewer.
 *
 * - `itemText`: testo di un singolo text-item del layer di pdf.js. pdf.js
 *   spezza la pagina in molti item (spesso una riga, a volte una parola).
 * - `snippet`: l'estratto testuale restituito insieme alla citazione.
 *
 * L'obiettivo dell'MVP è far risaltare la zona citata, non un match perfetto:
 * bisogna scegliere un'euristica robusta al rumore (maiuscole/minuscole,
 * spazi, item molto corti che darebbero falsi positivi).
 */
function shouldHighlight(itemText: string, snippet: string): boolean {
  const item = itemText.trim().toLowerCase();
  if (item.length < 4) return false;           // item troppo corti → falsi positivi ("e", "di", ".")
  return snippet.toLowerCase().includes(item); // lo snippet citato contiene il frammento della pagina
}

/**
 * Prepara il testo di un item per il text-layer: escaping HTML sempre (il
 * contenuto viene dal PDF e finisce in innerHTML), e wrapping in <mark> se
 * l'euristica lo evidenzia.
 */
function renderItem(itemText: string, snippet: string): string {
  const safe = itemText.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return shouldHighlight(itemText, snippet)
    ? `<mark class="rounded-sm bg-accent/40 text-inherit">${safe}</mark>`
    : safe;
}

// Pannello di destra: mostra la pagina PDF citata (blob dal docStore) con lo
// snippet evidenziato. Stati: nessuna citazione / PDF non disponibile / render.
export function SourceViewer({ citation }: { citation: Citation | null }) {
  if (!citation) {
    return (
      <div className="grid h-full place-items-center px-8 text-center text-gray-600">
        <div>
          <div className="mb-2 font-mono text-[0.7rem] uppercase tracking-[0.2em] text-violet/60">Fonte</div>
          Clicca una citazione{" "}
          <span className="mx-0.5 rounded bg-violet/10 px-1 font-mono text-violet">[n]</span>{" "}
          per aprire la pagina citata
        </div>
      </div>
    );
  }
  const file = docStore.getFile(citation.doc_name);
  if (!file) {
    return (
      <div className="grid h-full place-items-center px-8 text-center text-gray-500">
        PDF “{citation.doc_name}” non disponibile in questa sessione
      </div>
    );
  }
  return (
    <div className="h-full overflow-y-auto bg-panel p-4">
      <div className="mb-3 flex items-center gap-2 font-mono text-xs text-violet">
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet" />
        {citation.doc_name} · pag. {citation.page}
      </div>
      <div className="mx-auto w-fit overflow-hidden rounded-lg shadow-2xl ring-1 ring-white/10">
        <Document
          file={file}
          loading={<div className="p-6 text-sm text-gray-500">Carico il PDF…</div>}
          error={<div className="p-6 text-sm text-red-400">Impossibile aprire il PDF</div>}
        >
          <Page
            pageNumber={citation.page}
            width={520}
            customTextRenderer={({ str }) => renderItem(str, citation.snippet)}
          />
        </Document>
      </div>
    </div>
  );
}
