// Tipi condivisi del frontend, allineati alla risposta del backend.
export interface Citation { n: number; doc_name: string; page: number; snippet: string; }
export interface DocumentInfo { doc_id: string; doc_name: string; n_chunks: number; pages: number; }
export interface ChatMessage { role: "user" | "assistant"; text: string; citations?: Citation[]; }
