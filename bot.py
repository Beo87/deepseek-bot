import os 
import requests 
import threading 
import google.generativeai as genai 
from fastapi import FastAPI 
import uvicorn 
from telegram import Update 
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

===== ENV =====

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") HF_API_KEY = os.environ.get("HF_API_KEY") GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

===== GEMINI =====

genai.configure(api_key=GEMINI_API_KEY)

===== USER DB (IN-MEMORY) =====

users = {}

def check_limit(user_id): u = users.setdefault(user_id, {"count": 0, "plan": "free"})

if u["plan"] == "free" and u["count"] >= 20:
    return False

u["count"] += 1
return True

===== FASTAPI =====

app_api = FastAPI()

@app_api.get("/") def home(): return {"status": "ok"}

@app_api.post("/webhook") def webhook(data: dict): user_id = int(data.get("user_id", 0)) if user_id in users: users[user_id]["plan"] = "pro" return {"ok": True}

===== AI =====

def chat_nvidia(messages): try: r = requests.post( "https://integrate.api.nvidia.com/v1/chat/completions", headers={ "Authorization": "Bearer " + NVIDIA_API_KEY, "Content-Type": "application/json" }, json={"model": "meta/llama-3.1-8b-instruct", "messages": messages}, timeout=30 ) return r.json()["choices"][0]["message"]["content"] except Exception as e: return "❌ " + str(e)

def gemini_image(prompt): try: model = genai.GenerativeModel("gemini-2.5-flash-image") res = model.generate_content(prompt, generation_config={"response_modalities": ["IMAGE"]}) for p in res.candidates[0].content.parts: if p.inline_data: return p.inline_data.data except: return None

def hf_image(prompt): try: r = requests.post( "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell", headers={"Authorization": "Bearer " + HF_API_KEY}, json={"inputs": prompt}, timeout=60 ) if r.status_code == 200: return r.content except: return None

===== BOT =====

user_memory = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE): await update.message.reply_text( "🔥 BOT SaaS\n" "💬 Chat\n" "🎨 /imagine prompt\n" "💳 /upgrade\n" "♻️ /reset" )

async def upgrade(update: Update, context: ContextTypes.DEFAULT_TYPE): user_id = update.message.from_user.id link = f"https://your-domain.com/pay?user_id={user_id}" await update.message.reply_text("💎 Nang cap PRO:\n" + link)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE): user_memory[update.message.from_user.id] = [] await update.message.reply_text("♻️ Reset xong")

async def imagine(update: Update, context: ContextTypes.DEFAULT_TYPE): if not context.args: await update.message.reply_text("/imagine a dragon") return

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

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE): user_id = update.message.from_user.id text = update.message.text

if not check_limit(user_id):
    await update.message.reply_text("🚫 Het luot free! /upgrade")
    return

mem = user_memory.setdefault(user_id, [])
mem.append({"role": "user", "content": text})

await update.message.reply_chat_action("typing")

reply = chat_nvidia(mem)
mem.append({"role": "assistant", "content": reply})

await update.message.reply_text(reply[:4000])

===== RUN BOTH =====

def run_api(): uvicorn.run(app_api, host="0.0.0.0", port=8000)

threading.Thread(target=run_api).start()

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(CommandHandler("start", start)) app.add_handler(CommandHandler("upgrade", upgrade)) app.add_handler(CommandHandler("reset", reset)) app.add_handler(CommandHandler("imagine", imagine)) app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

print("🚀 SaaS BOT running (1 service)") app.run_polling()
