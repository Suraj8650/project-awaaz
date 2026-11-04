# Project Awaaz — Backend starter

A small FastAPI backend for the first Awaaz chat prototype.

## What it does
- `GET /` — service status
- `GET /health` — health check
- `POST /chat` — sends a message and optional recent chat history to the AI model
- Keeps the AI provider key on the server, never in website JavaScript

This is a starter backend, not the full digital-life assistant yet. It does **not**
connect to WhatsApp, email, calendar, contacts, files, payments, or device data.
It does not save long-term memory. Those need separate integrations, consent,
database storage, and security work.

## Run locally
1. Install Python 3.11 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Set `OPENAI_API_KEY` in your environment.
4. Start: `uvicorn main:app --reload`
5. Open `http://127.0.0.1:8000/docs`

Never commit your real API key or `.env` file to GitHub.

## Deploy (example: Render)
1. Push these files to a GitHub repository.
2. Create a new **Web Service** on Render and connect that repository.
3. Runtime: Python.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable `OPENAI_API_KEY` with your secret API key.
7. Optional: set `OPENAI_MODEL` (default `gpt-4o-mini`).
8. After deployment, test `https://YOUR-SERVICE.onrender.com/health`.

For the first prototype, `CORS_ORIGINS` defaults to `*` so your GitHub Pages
site can reach it. After you know the exact website URL, set `CORS_ORIGINS`
to that exact origin (for example `https://yourname.github.io`) and redeploy.

## Chat API example
POST `/chat` with JSON:
```json
{
  "message": "Namaste Awaaz",
  "history": [],
  "language": "auto"
}
```

Response:
```json
{
  "reply": "Namaste! Main Awaaz hoon...",
  "model": "gpt-4o-mini"
}
```

The frontend must call your deployed backend URL. Do not put `OPENAI_API_KEY`
in `index.html` or any browser-side JavaScript.
