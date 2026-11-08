# Project Awaaz — Complete package

## Render settings

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Render Environment Variables

- `OPENAI_API_KEY`: your secret OpenAI API key
- `OPENAI_MODEL`: `gpt-4o-mini` (optional)
- `CORS_ORIGINS`: `*` for initial testing; later replace with your exact website origin

Never put the API key in `index.html` or commit it to GitHub.

## API checks

- `GET /health` should return `status: ok`.
- `POST /chat` accepts `message`, `history`, `language`, and `mode`.
- `mode` can be `personal` or `business`.

The frontend `index.html` uses:
`https://awaaz-backend-j68d.onrender.com`

If your Render service URL is different, update `API_BASE` in `index.html`.

This prototype does not yet connect to email, WhatsApp, calendars, payment systems,
customer databases, or persistent long-term memory.
