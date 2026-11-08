import os
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

# -----------------------------
# CONFIGURATION
# -----------------------------

APP_NAME = "Project Awaaz API"

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("awaaz")

client = (
    OpenAI(
        api_key=API_KEY,
        timeout=30.0,
        max_retries=1
    )
    if API_KEY else None
)

# -----------------------------
# FASTAPI APPLICATION
# -----------------------------

app = FastAPI(
    title=APP_NAME,
    description="Personal and Business AI Assistant",
    version="2.0.0"
)

# -----------------------------
# CORS CONFIGURATION
# -----------------------------

cors_origins = os.getenv("CORS_ORIGINS", "*")

allowed_origins = (
    ["*"]
    if cors_origins == "*"
    else [
        origin.strip()
        for origin in cors_origins.split(",")
        if origin.strip()
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# -----------------------------
# REQUEST MODELS
# -----------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)

    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20
    )

    language: Literal["auto", "en", "hi"] = "auto"

    mode: Literal["personal", "business"] = "personal"


class ChatResponse(BaseModel):
    reply: str
    model: str


# -----------------------------
# AI PERSONALITY
# -----------------------------

PERSONAL_PROMPT = """
You are Awaaz, a helpful personal AI companion.

Your responsibilities:

1. Help users with everyday tasks.
2. Answer questions naturally.
3. Help with planning and organization.
4. Draft messages and documents.
5. Explain complicated topics simply.
6. Support Hindi, English and Hinglish.

Be friendly, intelligent, practical and respectful.

Never pretend to access private accounts or perform actions
that have not actually been completed.

Ask for confirmation before consequential actions.

You currently do not have direct access to email,
WhatsApp, calendars, contacts or personal files.
Be transparent about this.
"""


BUSINESS_PROMPT = """
You are Awaaz Business, an intelligent business assistant.

Your responsibilities:

1. Business planning and strategy.
2. Customer communication drafts.
3. Marketing ideas and campaigns.
4. Sales planning.
5. Customer service workflows.
6. Order management guidance.
7. Invoice and documentation drafts.
8. Business productivity.
9. Market research and analysis.
10. Professional email writing.

Understand the user's business requirements.

Give practical, professional and actionable suggestions.

Use clear formatting when explaining business processes.

Support Hindi, English and Hinglish.

Never claim to have contacted customers,
processed payments, created actual invoices,
or accessed business databases unless a connected
tool has actually performed that action.

Ask for confirmation before consequential actions.
"""


# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "status": "running",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "project-awaaz-backend",
        "model": MODEL,
        "api_configured": client is not None
    }


# -----------------------------
# CHAT ENGINE
# -----
