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
    expect(parseSSE('event: token\ndata: "x"')).toEqual([]); // niente \n\n finale
  });
});
