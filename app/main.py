import os
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag import answer, ingest

app=FastAPI(title="Doc Q&A")

class AskRequest(BaseModel):
    question:str

@app.post("/upload")
def upload(files: list[UploadFile]):
    results=[]
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename} is not a PDF")
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
            shutil.copyfileobj(f.file,tmp)
        try:
            pages= ingest(tmp.name,name=f.filename)
        finally:
            os.remove(tmp.name)
        results.append({"file": f.filename, "pages": pages})
    return {"ingested": results}

@app.post("/ask")
def ask(req: AskRequest):
    text, sources = answer(req.question)
    return {
        "answer": text,
        "citations": [{"id": i, "file": src, "page": page} for i, src, page in sources],
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def home():
    return FileResponse("static/index.html")
