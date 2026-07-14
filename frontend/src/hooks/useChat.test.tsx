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
