import { useState } from "react";
import { uploadDocument } from "../api";
import { docStore } from "../store/docStore";
import type { DocumentInfo } from "../types";

// Zona drag-and-drop: carica il PDF sul backend (indicizzazione) e ne conserva
// il blob nel docStore per il viewer. Espone stato idle/indicizzo/ok/errore.
export function Uploader({ onUploaded }: { onUploaded: (d: DocumentInfo) => void }) {
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);

  const handle = async (file: File) => {
    setBusy(true);
    setMsg(`Indicizzo ${file.name}…`);
    try {
      const info = await uploadDocument(file);
      docStore.putFile(info.doc_name, file);
      onUploaded(info);
      setMsg(`✓ ${info.doc_name} · ${info.n_chunks} chunk · ${info.pages} pagine`);
    } catch (e) {
      setMsg(`✗ ${e instanceof Error ? e.message : "Upload fallito"}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) handle(f); }}
      className={
        "block cursor-pointer rounded-xl border border-dashed p-5 text-center text-sm transition-all " +
        (drag ? "border-accent bg-accent/5" : "border-white/15 hover:border-accent/60 hover:bg-white/[0.02]") +
        (busy ? " animate-pulse" : "")
      }
    >
      <div className="font-mono text-[0.7rem] uppercase tracking-[0.15em] text-gray-500">Documento</div>
      <div className="mt-1 text-gray-300">Trascina un PDF o clicca per caricarlo</div>
      <input
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
      />
      {msg && (
        <div className={"mt-2.5 font-mono text-xs " + (msg.startsWith("✗") ? "text-red-400" : msg.startsWith("✓") ? "text-accent" : "text-gray-500")}>
          {msg}
        </div>
      )}
    </label>
  );
}
