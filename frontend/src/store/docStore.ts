// Store in-memory dei PDF caricati nella sessione: il backend non serve il file
// originale, quindi il viewer usa il blob (File) tenuto qui, indicizzato per nome.
const files = new Map<string, File>();
export const docStore = {
  putFile: (name: string, f: File) => files.set(name, f),
  getFile: (name: string) => files.get(name),
};
