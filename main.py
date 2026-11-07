import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field

APP_NAME = "Project Awaaz API"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, timeout=30.0, max_retries=1) if API_KEY else None

app = FastAPI(title=APP_NAME, version="0.2.0")

cors_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins = ["*"] if cors_origins.strip() == "*" else [
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
    mode: Literal["personal", "business"] = "personal"

class ChatResponse(BaseModel):
    reply: str
    model: str

BASE_PROMPT = """
You are Awaaz, a helpful AI assistant for users in India.
Be practical, clear, respectful, and concise. Understand Hindi, English, and Hinglish.
Reply in the language the user uses unless they request another language.
Never claim you completed an action, accessed private data, sent a message, booked,
paid, or changed an account unless a connected tool actually performed it.
Ask for confirmation before consequential actions. Protect personal information.
You currently have no access to email, WhatsApp, calendars, contacts, files, bank
accounts, customer records, orders, invoices, or the user's device. Be transparent.
You do not have persistent memory; you can use only the conversation history supplied
with the current request.
"""

MODE_PROMPTS = {
    "personal": """
The user selected Personal Awaaz. Focus on everyday assistance, explanations,
planning, learning, drafting, and personal organization. Do not claim to create
actual reminders unless a reminder integration is connected.
""",
    "business": """
The user selected Business Awaaz. Act as a practical business productivity assistant.
Help with planning, customer-service wording, order-process design, invoice templates,
and business analysis. You cannot see actual customer/order/payment records and cannot
create invoices or update orders in this prototype. Clearly distinguish advice and
drafts from completed business actions.
""",
}

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "project-awaaz-backend", "model": MODEL}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the backend host."
        )

    system_prompt = BASE_PROMPT + MODE_PROMPTS[payload.mode]
    if payload.language == "hi":
        system_prompt += "\nPrefer Hindi in Devanagari unless the user writes Hinglish."
    elif payload.language == "en":
        system_prompt += "\nPrefer English."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in payload.history
    )
    messages.append({"role": "user", "content": payload.message})

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
    except AuthenticationError:
        print("OpenAI authentication failed. Verify OPENAI_API_KEY in the hosting environment.")
        raise HTTPException(
            status_code=502,
            detail="AI authentication failed. The backend owner must verify the OpenAI API key in Render."
        )
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI usage limit or rate limit reached. Please try again later.")
    except (APIConnectionError, APITimeoutError):
        raise HTTPException(status_code=503, detail="The AI provider is temporarily unreachable. Please try again.")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Chat provider error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Awaaz could not get a reply right now. Check the backend logs and model configuration."
        )
