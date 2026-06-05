from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag import load_pdf, answer
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile, os
from threading import Lock
from typing import Dict


limiter = Limiter(key_func=get_remote_address)
MAX_FILE_SIZE = 20 * 1024 * 1024
app = FastAPI()
app.state.limiter = limiter
chains: Dict[str, object] = {}
chains_lock = Lock()


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests. Try again later."})


@app.get("/")
def hello():
    return {"message": "This is a Document based question answering chatbot API"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload-pdf")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, session_id: str = Form(...), file: UploadFile = File(...)):
    if not session_id:
        return {"success": False, "message": "Session ID is required."}

    if not file.filename.endswith(".pdf"):
        return {"success": False, "message": "Only PDF files are allowed!"}
    
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        return {"success": False, "message": "File too large. Maximum size is 20MB."}


    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    chain = load_pdf(tmp_path)
    os.unlink(tmp_path)

    with chains_lock:
        chains[session_id] = chain

    return {"success": True, "message": "PDF loaded successfully!"}


@app.post("/ask")
@limiter.limit("10/minute")
def ask_question(request: Request, payload: AskRequest):
    if not payload.session_id:
        return {"answer": "Session ID is required."}

    with chains_lock:
        chain = chains.get(payload.session_id)

    if chain is None:
        return {"answer": "Please upload a PDF first."}

    result = answer(payload.question, chain)
    return {"answer": result}