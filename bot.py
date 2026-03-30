import requests
import re
import time
import os
from Crypto.Cipher import AES
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

models = [
    "DeepSeek-V1", "DeepSeek-V2", "DeepSeek-V2.5", "DeepSeek-V3", "DeepSeek-V3-0324",
    "DeepSeek-V3.1", "DeepSeek-V3.2", "DeepSeek-R1", "DeepSeek-R1-0528", "DeepSeek-R1-Distill",
    "DeepSeek-Prover-V1", "DeepSeek-Prover-V1.5", "DeepSeek-Prover-V2", "DeepSeek-VL",
    "DeepSeek-Coder", "DeepSeek-Coder-V2", "DeepSeek-Coder-6.7B-base", "DeepSeek-Coder-6.7B-instruct"
]

user_data = {}

def tao_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Android)"})
    r = s.get("https://asmodeus.free.nf/")
    nums = re.findall(r"toNumbers\(\"([a-f0-9]+)\"\)", r.text)
    key, iv, data = [bytes.fromhex(n) for n in nums[:3]]
    s.cookies.set("__test", AES.new(key, AES.MODE_CBC, iv).decrypt(data).hex(), domain="asmodeus.free.nf")
    s.get("https://asmodeus.free.nf/index.php?i=1")
    time.sleep(0.5)
    return s

def goi_deepseek(session, model, question):
    r = session.post(
        "https://asmodeus.free.nf/deepseek.php",
        params={"i": "1"},
        data={"model": model, "question": question}
    )
    reply = re.search(r"<div class=\"response-content\">(.*?)</div>", r.text, re.DOTALL)
    return reply.group(1).strip() if reply else "Khong co phan hoi"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot DeepSeek\n\n"
        "/model - Chon model AI\n"
        "/reset - Tao session moi\n"
        "/mymodel - Xem model dang dung"
    )

async def my_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        model = user_data[user_id]["model"]
        await update.message.reply_text("Model hien tai: " + model)
    else:
        await update.message.reply_text("Chua chon model. Dung /model de chon.")

async def chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for i, m in enumerate(models, 1):
        row.append(InlineKeyboardButton(str(i) + ". " + m, callback_data="model_" + str(i-1)))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    await update.message.reply_text(
        "Chon model DeepSeek:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def xu_ly_chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    model_index = int(query.data.split("_")[1])
    model = models[model_index]
    await query.edit_message_text("Dang khoi tao " + model + "...")
    try:
        session = tao_session()
        user_data[user_id] = {"session": session, "model": model}
        await query.edit_message_text("Da chon: " + model + "\nNhan tin de bat dau chat!")
    except Exception as e:
        await query.edit_message_text("Loi: " + str(e))

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        del user_data[user_id]
    await update.message.reply_text("Da reset! Dung /model de chon lai.")

async def xu_ly_tin_nhan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    tin_nhan = update.message.text
    if user_id not in user_data:
        await update.message.reply_text("Chua chon model! Dung /model truoc.")
        return
    model = user_data[user_id]["model"]
    session = user_data[user_id]["session"]
    await update.message.reply_chat_action("typing")
    try:
        tra_loi = goi_deepseek(session, model, tin_nhan)
        if len(tra_loi) > 4096:
            tra_loi = tra_loi[:4090] + "..."
        await update.message.reply_text(model + ":\n\n" + tra_loi)
    except Exception as e:
        await update.message.reply_text("Loi: " + str(e) + "\nDung /reset roi /model de thu lai.")

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", chon_model))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CommandHandler("mymodel", my_model))
app.add_handler(CallbackQueryHandler(xu_ly_chon_model, pattern="^model_"))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, xu_ly_tin_nhan))

print("Bot dang chay...")
app.run_polling()
