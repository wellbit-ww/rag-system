from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    project: Optional[str] = None