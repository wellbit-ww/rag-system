import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form

from ingestion.ingest import ingest_file
from retrieval.query_engine import ask_question
from models.schemas import QuestionRequest


app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Advanced RAG API running"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    project: str = Form(...)
):

    project_dir = os.path.join(
        UPLOAD_DIR,
        project
    )

    os.makedirs(project_dir, exist_ok=True)

    file_path = os.path.join(
        project_dir,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_file(
        file_path=file_path,
        project=project
    )

    return {
        "status": "success",
        "filename": file.filename,
        "project": project
    }


@app.post("/ask")
async def ask(request: QuestionRequest):

    response = ask_question(
        question=request.question,
        project=request.project
    )

    return response