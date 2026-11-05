
import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Project Awaaz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    language: str = "auto"


@app.get("/")
def home():
    return {
        "project": "Awaaz",
        "status": "running",
        "message": "Awaaz AI backend is live"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(request: ChatRequest):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured on the server."
        )

    try:
        client = OpenAI(api_key=api_key)

        conversation = []

        for item in request.history[-12:]:
            conversation.append({
                "role": item.role,
                "content": item.content
            })

        conversation.append({
            "role": "user",
            "content": request.message
        })

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=(
                "You are Awaaz, a helpful personal AI assistant. "
                "Reply naturally in the same language as the user. "
                "Support Hindi and English. Be clear, friendly and useful. "
                "Never claim to have completed an action unless it was actually done."
            ),
            input=conversation
        )

        return {
            "reply": response.output_text,
            "status": "success"
        }

    except Exception:
        logging.exception("Awaaz OpenAI ERROR:")

        raise HTTPException(
            status_code=502,
            detail="Awaaz AI service encountered an error."
        )
