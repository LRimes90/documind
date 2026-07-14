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
