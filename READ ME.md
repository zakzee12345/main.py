# Telegram Bot Deployment on Render

This is a basic Telegram bot built with the `python-telegram-bot` library (v20+) and designed to run continuously on Render.

## Features
- Responds to the `/start` command with a welcome message.
- Uses `asyncio.Event()` to keep running forever.
- Securely handles your bot token via environment variables.
- Easy to deploy on Render.

## Setup

1. Add all files to your GitHub repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your bot token in an environment variable `BOT_TOKEN`.
4. Run locally with `python main.py` or deploy to Render.

## Deploying to Render

1. Push your code to GitHub.
2. On [https://render.com](https://render.com), create a new Web Service.
3. Connect your GitHub repo.
4. Set start command: `python main.py`
5. Add environment variable: `BOT_TOKEN=your_telegram_bot_token_here`
6. 
