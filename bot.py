import os
import requests
import base64
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # vd: "tenban/deepseek-bot"

# ===== DANH SACH MODEL =====
# provider: "openrouter" hoac "github"
models = [
    # OpenRouter free
    {"name": "DeepSeek R1 (OR)", "id": "deepseek/deepseek-r1:free", "provider": "openrouter"},
    {"name": "DeepSeek V3 (OR)", "id": "deepseek/deepseek-chat-v3-0324:free", "provider": "openrouter"},
    {"name": "Gemini 2.0 Flash (OR)", "id": "google/gemini-2.0-flash-exp:free", "provider": "openrouter"},
    {"name": "Gemini 2.5 Pro (OR)", "id": "google/gemini-2.5-pro-exp-03-25:free", "provider": "openrouter"},
    {"name": "Llama 4 Maverick (OR)", "id": "meta-llama/llama-4-maverick:free", "provider": "openrouter"},
    {"name": "Mistral Small (OR)", "id": "mistralai/mistral-small-3.1-24b-instruct:free", "provider": "openrouter"},
    # GitHub Models free
    {"name": "GPT-4o (GH)", "id": "openai/gpt-4o-mini", "provider": "github"},
    {"name": "DeepSeek R1 (GH)", "id": "deepseek/DeepSeek-R1", "provider": "github"},
    {"name": "DeepSeek V3 (GH)", "id": "deepseek/DeepSeek-V3-0324", "provider": "github"},
    {"name": "Grok 3 Mini (GH)", "id": "xai/grok-3-mini", "provider": "github"},
    {"name": "Llama 4 Maverick (GH)", "id": "meta/Llama-4-Maverick-17B-128E-Instruct-FP8", "provider": "github"},
    {"name": "Llama 3.3 70B (GH)", "id": "meta/Llama-3.3-70B-Instruct", "provider": "github"},
]

user_data = {}
edit_state = {}

# ===== AI FUNCTIONS =====

def parse_error(data, prefix):
    err = data.get("error", "Khong ro")
    if isinstance(err, dict):
        msg = err.get("message", str(err))
    else:
        msg = str(err)
    return prefix + ": " + msg

def goi_openrouter(model_id, messages):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + OPENROUTER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"model": model_id, "messages": messages},
            timeout=30
        )
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return parse_error(data, "Loi OpenRouter")
        return "Khong co phan hoi (status: " + str(response.status_code) + ")"
    except requests.exceptions.Timeout:
        return "Loi: Qua thoi gian cho (timeout)"
    except Exception as e:
        return "Loi ket noi OpenRouter: " + str(e)

def goi_github_models(model_id, messages):
    try:
        response = requests.post(
            "https://models.inference.ai.azure.com/chat/completions",
            headers={
                "Authorization": "Bearer " + GITHUB_TOKEN,
                "Content-Type": "application/json",
            },
            json={"model": model_id, "messages": messages},
            timeout=30
        )
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return parse_error(data, "Loi GitHub Models")
        return "Khong co phan hoi (status: " + str(response.status_code) + ")"
    except requests.exceptions.Timeout:
        return "Loi: Qua thoi gian cho (timeout)"
    except Exception as e:
        return "Loi ket noi GitHub Models: " + str(e)

def goi_ai(provider, model_id, messages):
    if provider == "github":
        return goi_github_models(model_id, messages)
    else:
        return goi_openrouter(model_id, messages)

# ===== GITHUB REPO FUNCTIONS =====

def github_headers():
    return {
        "Authorization": "token " + GITHUB_TOKEN,
        "Accept": "application/vnd.github.v3+json"
    }

def list_files(path=""):
    url = "https://api.github.com/repos/" + GITHUB_REPO + "/contents/" + path
    r = requests.get(url, headers=github_headers())
    if r.status_code == 200:
        return r.json()
    return None

def get_file(path):
    url = "https://api.github.com/repos/" + GITHUB_REPO + "/contents/" + path
    r = requests.get(url, headers=github_headers())
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    return None, None

def update_file(path, content, sha, message="Update via Telegram bot"):
    url = "https://api.github.com/repos/" + GITHUB_REPO + "/contents/" + path
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    r = requests.put(url, headers=github_headers(), json={
        "message": message,
        "content": encoded,
        "sha": sha
    })
    return r.status_code == 200

def create_file_github(path, content, message="Create via Telegram bot"):
    url = "https://api.github.com/repos/" + GITHUB_REPO + "/contents/" + path
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    r = requests.put(url, headers=github_headers(), json={
        "message": message,
        "content": encoded
    })
    return r.status_code == 201

# ===== BOT COMMANDS =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot AI + GitHub\n\n"
        "CHAT AI:\n"
        "/model - Chon model (OpenRouter + GitHub)\n"
        "/mymodel - Xem model dang dung\n"
        "/reset - Xoa lich su chat\n\n"
        "GITHUB REPO:\n"
        "/files - Xem danh sach file\n"
        "/read ten_file - Doc file\n"
        "/edit ten_file - Sua file\n"
        "/push ten_file - Tao file moi\n"
        "/cancel - Huy thao tac"
    )

async def chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for i, m in enumerate(models):
        label = m["name"]
        row.append(InlineKeyboardButton(label, callback_data="model_" + str(i)))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    await update.message.reply_text(
        "Chon model AI:\n(OR = OpenRouter | GH = GitHub Models)",
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
        "provider": model["provider"],
        "messages": []
    }
    provider_label = "GitHub Models" if model["provider"] == "github" else "OpenRouter"
    await query.edit_message_text(
        "Da chon: " + model["name"] + "\n"
        "Nguon: " + provider_label + "\n\n"
        "Nhan tin de bat dau chat!"
    )

async def my_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        d = user_data[user_id]
        provider_label = "GitHub Models" if d["provider"] == "github" else "OpenRouter"
        await update.message.reply_text(
            "Model: " + d["model_name"] + "\nNguon: " + provider_label
        )
    else:
        await update.message.reply_text("Chua chon model. Dung /model")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        user_data[user_id]["messages"] = []
    await update.message.reply_text("Da xoa lich su chat!")

# ===== GITHUB COMMANDS =====

async def files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = list_files()
    if not result:
        await update.message.reply_text("Loi: Khong the lay danh sach file")
        return
    text = "File trong repo:\n\n"
    for f in result:
        icon = "Folder" if f["type"] == "dir" else "File"
        text += icon + ": " + f["name"] + "\n"
    await update.message.reply_text(text)

async def read_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /read ten_file\nVi du: /read bot.py")
        return
    path = context.args[0]
    content, sha = get_file(path)
    if content is None:
        await update.message.reply_text("Loi: Khong tim thay file " + path)
        return
    if len(content) > 4000:
        content = content[:4000] + "\n...(con tiep)"
    await update.message.reply_text("Noi dung " + path + ":\n\n" + content)

async def edit_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Dung: /edit ten_file\nVi du: /edit bot.py")
        return
    path = context.args[0]
    content, sha = get_file(path)
    if content is None:
        await update.message.reply_text("Loi: Khong tim thay file " + path)
        return
    edit_state[user_id] = {"mode": "edit", "path": path, "sha": sha}
    preview = content[:1500] + "\n...(con tiep)" if len(content) > 1500 else content
    await update.message.reply_text(
        "Dang sua: " + path + "\n\n"
        "Noi dung hien tai:\n" + preview + "\n\n"
        "Gui noi dung MOI de thay the\n"
        "Hoac /cancel de huy"
    )

async def push_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Dung: /push ten_file\nVi du: /push newfile.py")
        return
    path = context.args[0]
    edit_state[user_id] = {"mode": "create", "path": path}
    await update.message.reply_text(
        "Tao file moi: " + path + "\n\n"
        "Gui noi dung file\n"
        "Hoac /cancel de huy"
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in edit_state:
        del edit_state[user_id]
    await update.message.reply_text("Da huy!")

# ===== XU LY TIN NHAN =====

async def xu_ly_tin_nhan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    tin_nhan = update.message.text

    # Che do edit/create file
    if user_id in edit_state:
        state = edit_state[user_id]
        if state["mode"] == "edit":
            success = update_file(state["path"], tin_nhan, state["sha"])
            del edit_state[user_id]
            if success:
                await update.message.reply_text("Da cap nhat " + state["path"] + " thanh cong!")
            else:
                await update.message.reply_text("Loi cap nhat file!")
        elif state["mode"] == "create":
            success = create_file_github(state["path"], tin_nhan)
            del edit_state[user_id]
            if success:
                await update.message.reply_text("Da tao " + state["path"] + " thanh cong!")
            else:
                await update.message.reply_text("Loi tao file!")
        return

    # Chat AI
    if user_id not in user_data:
        await update.message.reply_text("Chua chon model! Dung /model truoc.")
        return

    user_data[user_id]["messages"].append({"role": "user", "content": tin_nhan})
    await update.message.reply_chat_action("typing")

    try:
        tra_loi = goi_ai(
            user_data[user_id]["provider"],
            user_data[user_id]["model_id"],
            user_data[user_id]["messages"]
        )
        user_data[user_id]["messages"].append({"role": "assistant", "content": tra_loi})
        if len(tra_loi) > 4096:
            tra_loi = tra_loi[:4090& + "..."
        await update.message.reply_text(user_data[user_id]["model_name"] + ":\n\n" + tra_loi)
    except Exception as e:
        await update.message.reply_text("Loi: " + str(e))

# ===== CHAY BOT =====
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", chon_model))
app.add_handler(CommandHandler("mymodel", my_model))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CommandHandler("files", files))
app.add_handler(CommandHandler("read", read_file))
app.add_handler(CommandHandler("edit", edit_file))
app.add_handler(CommandHandler("push", push_file))
app.add_handler(CommandHandler("cancel", cancel))
app.add_handler(CallbackQueryHandler(xu_ly_chon_model, pattern="^model_"))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, xu_ly_tin_nhan))

print("Bot dang chay...")
app.run_polling()
