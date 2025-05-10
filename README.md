# Telegram Bot Deployment on Render

This is a basic Telegram bot built with the `python-telegram-bot` library (v20+) and designed to run continuously on [Render](https://render.com).

## Features

- Responds to the `/start` command with a welcome message.
- Uses `asyncio.Event()` to keep running forever.
- Securely handles your bot token via environment variables.
- Easy to deploy on Render.

---

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/your-telegram-bot.git
cd your-telegram-bot
