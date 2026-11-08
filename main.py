import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)
from pydantic import BaseModel, Field

# -----------------------------
# CONFIGURATION
# -----------------------------

APP_NAME = "Project Awaaz API"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_awaaz")

client = (
    OpenAI(api_key=API_KEY, timeout=40.0, max_retries=1)
    if API_KEY else None
)

# -----------------------------
# FASTAPI APPLICATION
# -----------------------------

app = FastAPI(
    title=APP_NAME,
    description="Personal and Business AI Assistant",
    version="2.1.0",
)

# -----------------------------
# CORS
# -----------------------------

cors_origins = os.getenv("CORS_ORIGINS", "*").strip()

allowed_origins = (
    ["*"] if cors_origins == "*"
    else [
        origin.strip().rstrip("/")
        for origin in cors_origins.split(",")
        if origin.strip()
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# -----------------------------
# REQUEST / RESPONSE MODELS
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
# AI PERSONALITY / PROMPTS
# -----------------------------

PERSONAL_PROMPT = """
You are Awaaz, a helpful personal AI companion for users in India.

YOUR ROLE:
- Help with everyday questions, planning, learning, and organization.
- Help draft messages, emails, notes, and documents.
- Explain complicated topics in simple language.
- Be friendly, thoughtful, practical, respectful, and emotionally aware.
- Understand Hindi, English, and Hinglish. Reply in the language the user uses.

IMPORTANT LIMITATIONS:
- You do not automatically have access to email, WhatsApp, contacts, calendars,
  personal files, bank accounts, or the user's device.
- Never claim that you sent a message, booked something, made a payment,
  changed an account, or completed an external action unless a connected tool
  actually performed it.
- Be honest about uncertainty and limitations.
- Ask for confirmation before consequential actions.
- Protect the user's private information.

RESPONSE STYLE:
- Give useful, direct answers.
- Use steps when explaining a process.
- Avoid repeating the same generic response.
- Ask one focused clarification only when essential.
"""

BUSINESS_PROMPT = """
You are Awaaz Business, a capable business planning and productivity assistant
for users in India.

YOUR ROLE:
- Help with business planning, strategy, operations, and productivity.
- Draft customer replies, professional emails, proposals, and announcements.
- Suggest marketing campaigns, sales plans, and customer-service workflows.
- Help structure order processes, invoice templates, and business documents.
- Analyze information the user provides and turn it into practical next steps.
- Understand Hindi, English, and Hinglish. Reply in the language the user uses.

IMPORTANT LIMITATIONS:
- You are not connected to the user's customers, payment gateway, inventory,
  accounting software, CRM, email, WhatsApp, or business database unless a
  specific integration is actually configured.
- Never claim to have contacted a customer, processed an order or payment,
  issued an invoice, or changed business records unless a connected tool
  actually performed that action.
- Ask for confirmation before consequential actions.
- Do not invent business facts, prices, sales figures, or customer information.

RESPONSE STYLE:
- Be professional, clear, practical, and action-oriented.
- Use headings, checklists, or numbered steps when they improve clarity.
- Explain assumptions and ask for missing business details when necessary.
"""

# -----------------------------
# HEALTH CHECKS
# -----------------------------

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "status": "running",
        "version": "2.1.0",
        "health": "/health",
        "chat": "/chat",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "project-awaaz-backend",
        "model": MODEL,
        "api_configured": client is not None,
    }


# -----------------------------
# CHAT ENGINE
# -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing in Render Environment Variables.",
        )

    system_prompt = (
        BUSINESS_PROMPT
        if payload.mode == "business"
        else PERSONAL_PROMPT
    )

    if payload.language == "hi":
        system_prompt += (
            "\nPrefer Hindi in Devanagari unless the user writes Hinglish."
        )

    elif payload.language == "en":
        system_prompt += "\nPrefer English."

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    messages.extend(
        {
            "role": item.role,
            "content": item.content
        }
        for item in payload.history
    )

    messages.append({
        "role": "user",
        "content": payload.message
    })

    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )

        reply = (
            result.choices[0].message.content or ""
        ).strip()

        if not reply:
            logger.error("OpenAI returned an empty response.")

            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response. Please try again.",
            )

        return ChatResponse(
            reply=reply,
            model=MODEL
        )

    except AuthenticationError:
        logger.exception("OpenAI authentication failed.")

        raise HTTPException(
            status_code=503,
            detail="OpenAI rejected the API key. Check OPENAI_API_KEY in Render.",
        )

    except RateLimitError:
        logger.exception("OpenAI quota or rate limit reached.")

        raise HTTPException(
            status_code=503,
            detail="OpenAI quota or rate limit reached. Check API billing and usage.",
        )

    except (APITimeoutError, APIConnectionError):
        logger.exception("OpenAI connection or timeout error.")

        raise HTTPException(
            status_code=502,
            detail="Awaaz could not connect to OpenAI. Please try again shortly.",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected Project Awaaz chat error.")

        raise HTTPException(
            status_code=502,
            detail="Awaaz encountered a server error. Check the Render logs.",
        )
@app.get(”/chat-test”, response_model=ChatResponse) def chat_test(message: str = “Hello Awaaz, introduce yourself.”): return chat(ChatRequest(message=message, history=[])
