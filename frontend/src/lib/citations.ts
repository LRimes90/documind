export type Segment = { type: "text"; value: string } | { type: "cite"; n: number };

/**
 * Spezza una risposta in segmenti di testo e marcatori di citazione `[n]`,
 * così il renderer può sostituire i marcatori con chip cliccabili.
 */
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
