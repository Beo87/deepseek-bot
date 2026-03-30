import os
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

models = [
    {"name": "DeepSeek R1", "id": "deepseek/deepseek-r1:free"},
    {"name": "DeepSeek V3", "id": "deepseek/deepseek-chat-v3-0324:free"},
    {"name": "Gemini 2.0 Flash", "id": "google/gemini-2.0-flash-exp:free"},
    {"name": "Gemini 2.5 Pro", "id": "google/gemini-2.5-pro-exp-03-25:free"},
    {"name": "Llama 4 Maverick", "id": "meta-llama/llama-4-maverick:free"},
    {"name": "Mistral Small 3.1", "id": "mistralai/mistral-small-3.1-24b-instruct:free"},
    {"name": "Auto Free", "id": "openrouter/auto"},
]

user_data = {}

def goi_openrouter(model_id, messages):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + OPENROUTER_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
        }
    )
    data = response.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    elif "error" in data:
        return "Loi: " + str(data["error"].get("message", "Khong ro"))
    return "Khong co phan hoi"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot AI nhieu model\n\n"
        "/model - Chon model AI\n"
        "/mymodel - Xem model dang dung\n"
        "/reset - Xoa lich su chat\n\n"
        "Bat dau bang /model !"
    )

async def chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for i, m in enumerate(models):
        row.append(InlineKeyboardButton(m["name"], callback_data="model_" + str(i)))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    await update.message.reply_text(
        "Chon model AI (tat ca mien phi):",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def xu_ly_chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    model_index = int(query.data.split("_")[1])
    model = models[model_index]
    user_data[user_id] = {
        "model_id": model["id"],
        "model_name": model["name"],
        "messages": []
    }
    await query.edit_message_text(
        "Da chon: " + model["name"] + "\nNhan tin de bat dau chat!"
    )

async def my_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        await update.message.reply_text("Model hien tai: " + user_data[user_id]["model_name"])
    else:
        await update.message.reply_text("Chua chon model. Dung /model de chon.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        user_data[user_id]["messages"] = []
    await update.message.reply_text("Da xoa lich su chat!")

async def xu_ly_tin_nhan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    tin_nhan = update.message.text

    if user_id not in user_data:
        await update.message.reply_text("Chua chon model! Dung /model truoc.")
        return

    user_data[user_id]["messages"].append({"role": "user", "content": tin_nhan})

    await update.message.reply_chat_action("typing")

    try:
        tra_loi = goi_openrouter(
            user_data[user_id]["model_id"],
            user_data[user_id]["messages"]
        )
        user_data[user_id]["messages"].append({"role": "assistant", "content": tra_loi})

        if len(tra_loi) > 4096:
            tra_loi = tra_loi[:4090] + "..."

        await update.message.reply_text(
            user_data[user_id]["model_name"] + ":\n\n" + tra_loi
        )
    except Exception as e:
        await update.message.reply_text("Loi: " + str(e))

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", chon_model))
app.add_handler(CommandHandler("mymodel", my_model))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CallbackQueryHandler(xu_ly_chon_model, pattern="^model_"))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, xu_ly_tin_nhan))

print("Bot dang chay...")
app.run_polling()
