import os
import requests
import threading
import google.generativeai as genai
from fastapi import FastAPI
import uvicorn
from telegram.request import HTTPXRequest
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ENV
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# GEMINI
genai.configure(api_key=GEMINI_API_KEY)

# USER DB (IN-MEMORY)
users = {}

def check_limit(user_id):
    u = users.setdefault(user_id, {"count": 0, "plan": "free"})
    if u["plan"] == "free" and u["count"] >= 20:
        return False
    u["count"] += 1
    return True

# FASTAPI
app_api = FastAPI()

@app_api.get("/")
def home():
    return {"status": "ok"}

@app_api.post("/webhook")
def webhook(data: dict):
    user_id = int(data.get("user_id", 0))
    if user_id in users:
        users[user_id]["plan"] = "pro"
    return {"ok": True}

# AI
def chat_nvidia(messages):
    try:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + NVIDIA_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.1-8b-instruct",
                "messages": messages
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "❌ " + str(e)

def gemini_image(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-image")
        res = model.generate_content(
            prompt,
            generation_config={"response_modalities": ["IMAGE"]}
        )
        for p in res.candidates[0].content.parts:
            if p.inline_data:
                return p.inline_data.data
        return None
    except:
        return None

def hf_image(prompt):
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            headers={"Authorization": "Bearer " + HF_API_KEY},
            json={"inputs": prompt},
            timeout=60
        )
        if r.status_code == 200:
            return r.content
        return None
    except:
        return None

# BOT
user_memory = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 BOT SaaS
"
        "💬 Chat
"
        "🎨 /imagine prompt
"
        "💳 /upgrade
"
        "♻️ /reset
"
    )

async def upgrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    link = f"https://your-domain.com/pay?user_id={user_id}"
    await update.message.reply_text("💎 Nang cap PRO:
" + link)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_memory[update.message.from_user.id] = []
    await update.message.reply_text("♻️ Reset xong")

async def imagine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("/imagine a dragon")
        return

    prompt = " ".join(context.args)
    await update.message.reply_text("🎨 Dang tao...")

    img = gemini_image(prompt)
    if isinstance(img, bytes):
        await update.message.reply_photo(photo=img, caption="✨ Gemini")
        return

    img = hf_image(prompt)
    if isinstance(img, bytes):
        await update.message.reply_photo(photo=img, caption="🎨 HF")
    else:
        await update.message.reply_text("❌ Loi")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text

    if not check_limit(user_id):
        await update.message.reply_text("🚫 Het luot free! /upgrade")
        return

    mem = user_memory.setdefault(user_id, [])
    mem.append({"role": "user", "content": text})

    await update.message.reply_chat_action("typing")

    reply = chat_nvidia(mem)
    mem.append({"role": "assistant", "content": reply})

    await update.message.reply_text(reply[:4000])

# RUN BOTH
http_request = HTTPXRequest(connect_timeout=10.0, read_timeout=10.0)

def run_api():
    uvicorn.run("bot:app_api", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")

threading.Thread(target=run_api, daemon=True).start()

app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(http_request).build()
# ... add_handler giống cũ ...

# Set webhook (chạy 1 lần hoặc manual)
WEBHOOK_URL = f"https://{os.environ.get('RAILWAY_STATIC_URL', 'your-app.railway.app')}/webhook_telegram"
app.bot.set_webhook(WEBHOOK_URL)

@app_api.post("/webhook_telegram")
async def telegram_webhook(request: Request):
    update = Update.de_json(await request.json(), app.bot)
    await app.process_update(update)
    return {"ok": True}

print("🚀 Bot on Railway webhook")
app.run_polling()  # Backup, nhưng webhook chính
