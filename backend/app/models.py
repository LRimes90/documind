from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    text: str
    chunk_id: str


class Citation(BaseModel):
    n: int
    doc_name: str
    page: int
    snippet: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    n_chunks: int
    pages: int
