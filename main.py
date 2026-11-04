import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME = "Project Awaaz API"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    # The server can still start so /health works; /chat will explain the missing key.
    client = None
else:
    client = OpenAI(api_key=API_KEY)

app = FastAPI(title=APP_NAME, version="0.1.0")

# For the first test only. Once your website URL is fixed, set CORS_ORIGINS
# to that exact URL in your hosting provider's environment variables.
cors_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins = ["*"] if cors_origins == "*" else [
    item.strip() for item in cors_origins.split(",") if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    language: Literal["auto", "en", "hi"] = "auto"

class ChatResponse(BaseModel):
    reply: str
    model: str

SYSTEM_PROMPT = """
You are Awaaz, a helpful personal AI assistant being built for users in India.
Be practical, clear, respectful, and concise. Understand Hindi, English, and Hinglish.
Reply in the language the user uses unless they request another language.
Help users organize their digital life: explain, plan, summarize, draft, and reason.
Do not claim you sent messages, booked things, changed accounts, made payments,
accessed private data, or completed actions unless a connected tool actually did it.
Ask for confirmation before any consequential action. Protect personal information.
This early version has no access to the user's email, WhatsApp, calendar, files,
bank accounts, or device. Be transparent about that limitation.
"""

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "project-awaaz-backend"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the backend host."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in payload.history:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": payload.message})

    if payload.language == "hi":
        messages[0]["content"] += "\nPrefer Hindi in Devanagari unless the user writes Hinglish."
    elif payload.language == "en":
        messages[0]["content"] += "\nPrefer English."

    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=700,
        )
        reply = (result.choices[0].message.content or "").strip()
        if not reply:
            raise HTTPException(status_code=502, detail="The AI returned an empty response.")
        return ChatResponse(reply=reply, model=MODEL)
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid exposing provider secrets or raw internal errors to the browser.
        print(f"Chat provider error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Awaaz could not get a reply right now. Check the backend logs and API configuration."
        )
