from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag import load_pdf, answer
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile, os


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
chain = None


class QuestionRequest(BaseModel):
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
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    global chain

    if not file.filename.endswith(".pdf"):
        return {"success": False, "message": "Only PDF files are allowed!"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    chain = load_pdf(tmp_path)
    os.unlink(tmp_path)

    return {"success": True, "message": "PDF loaded successfully!"}


@app.post("/ask")
@limiter.limit("10/minute")
def ask_question(request: Request, qustion: QuestionRequest):
    global chain

    if chain is None:
        return {"answer": "Please upload a PDF first."}

    result = answer(qustion.question, chain)
    return {"answer": result}