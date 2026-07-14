// Marcatore di citazione in stile "nota a piè di pagina": pulsante monospace
// viola. Il contenuto testuale è il solo numero → accessible name = "1"
// (il test cerca getByRole("button", { name: "1" })).
export function CitationChip({ n, onClick }: { n: number; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`Apri la fonte [${n}]`}
      className="group relative mx-0.5 inline-flex h-5 min-w-[1.25rem] -translate-y-px items-center justify-center rounded-md border border-violet/30 bg-violet/10 px-1.5 font-mono text-[0.7rem] font-semibold leading-none text-violet transition-all duration-150 hover:-translate-y-0.5 hover:border-violet/70 hover:bg-violet/25 hover:shadow-[0_4px_12px_-4px] hover:shadow-violet/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-violet/60"
    >
      {n}
    </button>
  );
}
