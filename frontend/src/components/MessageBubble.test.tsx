import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MessageBubble } from "./MessageBubble";

describe("MessageBubble", () => {
  it("rende i chip [n] e chiama onCite", () => {
    const onCite = vi.fn();
    render(
      <MessageBubble
        msg={{ role: "assistant", text: "Parigi [1].", citations: [{ n: 1, doc_name: "a.pdf", page: 1, snippet: "x" }] }}
        onCite={onCite}
      />,
    );
    const chip = screen.getByRole("button", { name: "1" });
    fireEvent.click(chip);
    expect(onCite).toHaveBeenCalledWith(expect.objectContaining({ n: 1, page: 1 }));
  });
});
