from fastapi import FastAPI

app = FastAPI(title="DocuMind API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
