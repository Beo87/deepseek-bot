import os
import json
import requests
import base64
from html.parser import HTMLParser
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# ===== ENV VARIABLES =====
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN")
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY")
GITHUB_TOKEN    = os.environ.get("GIHUB_TOKEN")
GITHUB_REPO     = os.environ.get("GIHUB_REPO")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")

# ===== FILES TREN GITHUB =====
MEMORY_FILE    = "memory.json"
USER_DATA_FILE = "user_data.json"

# ===== DANH SACH MODEL NVIDIA NIM =====
models = [
    # Meta Llama
    {"name": "Llama 4 Maverick",     "id": "meta/llama-4-maverick-17b-128e-instruct",    "provider": "nvidia"},
    {"name": "Llama 4 Scout",        "id": "meta/llama-4-scout-17b-16e-instruct",         "provider": "nvidia"},
    {"name": "Llama 3.3 70B",        "id": "meta/llama-3.3-70b-instruct",                 "provider": "nvidia"},
    {"name": "Llama 3.1 405B",       "id": "meta/llama-3.1-405b-instruct",                "provider": "nvidia"},
    {"name": "Llama 3.1 70B",        "id": "meta/llama-3.1-70b-instruct",                 "provider": "nvidia"},
    {"name": "Llama 3.1 8B",         "id": "meta/llama-3.1-8b-instruct",                  "provider": "nvidia"},
    # DeepSeek
    {"name": "DeepSeek R1",          "id": "deepseek-ai/deepseek-r1",                     "provider": "nvidia"},
    {"name": "DeepSeek V3.2",        "id": "deepseek-ai/deepseek-v3.2",                   "provider": "nvidia"},
    {"name": "DeepSeek V3.1",        "id": "deepseek-ai/deepseek-v3.1",                   "provider": "nvidia"},
    {"name": "DeepSeek R1 Qwen 32B", "id": "deepseek-ai/deepseek-r1-distill-qwen-32b",   "provider": "nvidia"},
    {"name": "DeepSeek R1 Llama 8B", "id": "deepseek-ai/deepseek-r1-distill-llama-8b",   "provider": "nvidia"},
    # Google Gemma
    {"name": "Gemma 4 31B",          "id": "google/gemma-4-31b-it",                       "provider": "nvidia"},
    {"name": "Gemma 3 27B",          "id": "google/gemma-3-27b-it",                       "provider": "nvidia"},
    {"name": "Gemma 2 27B",          "id": "google/gemma-2-27b-it",                       "provider": "nvidia"},
    {"name": "Gemma 2 9B",           "id": "google/gemma-2-9b-it",                        "provider": "nvidia"},
    # NVIDIA Nemotron
    {"name": "Nemotron Ultra 253B",  "id": "nvidia/llama-3.1-nemotron-ultra-253b-v1",     "provider": "nvidia"},
    {"name": "Nemotron Super 49B",   "id": "nvidia/llama-3.3-nemotron-super-49b-v1",      "provider": "nvidia"},
    {"name": "Nemotron Nano 8B",     "id": "nvidia/llama-3.1-nemotron-nano-8b-v1",        "provider": "nvidia"},
    {"name": "Nemotron Nano 4B",     "id": "nvidia/llama-3.1-nemotron-nano-4b-v1_1",      "provider": "nvidia"},
    # Mistral
    {"name": "Mixtral 8x22B",        "id": "mistralai/mixtral-8x22b-instruct",            "provider": "nvidia"},
    {"name": "Mixtral 8x7B",         "id": "mistralai/mixtral-8x7b-instruct",             "provider": "nvidia"},
    {"name": "Mistral Small 24B",    "id": "mistralai/mistral-small-24b-instruct",         "provider": "nvidia"},
    {"name": "Mistral 7B",           "id": "mistralai/mistral-7b-instruct-v0.3",           "provider": "nvidia"},
    {"name": "Mistral Nemotron",     "id": "mistralai/mistral-nemotron",                   "provider": "nvidia"},
    {"name": "Codestral 22B",        "id": "mistralai/codestral-22b-instruct-v0.1",        "provider": "nvidia"},
    # Qwen
    {"name": "Qwen3 Coder 480B",     "id": "qwen/qwen3-coder-480b-a35b-instruct",         "provider": "nvidia"},
    {"name": "QwQ 32B",              "id": "qwen/qwq-32b",                                 "provider": "nvidia"},
    {"name": "Qwen2.5 Coder 32B",    "id": "qwen/qwen2.5-coder-32b-instruct",             "provider": "nvidia"},
    {"name": "Qwen2.5 7B",           "id": "qwen/qwen2.5-7b-instruct",                    "provider": "nvidia"},
    # Microsoft Phi
    {"name": "Phi-4 Mini Instruct",  "id": "microsoft/phi-4-mini-instruct",                "provider": "nvidia"},
    {"name": "Phi-4 Mini Reasoning", "id": "microsoft/phi-4-mini-flash-reasoning",         "provider": "nvidia"},
    {"name": "Phi-3.5 Mini",         "id": "microsoft/phi-3.5-mini",                       "provider": "nvidia"},
    # Moonshot Kimi
    {"name": "Kimi K2",              "id": "moonshotai/kimi-k2-instruct",                  "provider": "nvidia"},
    {"name": "Kimi K2 Thinking",     "id": "moonshotai/kimi-k2-thinking",                  "provider": "nvidia"},
    # Khác
    {"name": "MiniMax M2.5",         "id": "minimaxai/minimax-m2.5",                       "provider": "nvidia"},
    {"name": "IBM Granite 3.3",      "id": "ibm/granite-3_3-8b-instruct",                  "provider": "nvidia"},
]

user_data  = {}
edit_state = {}

# ===== HELPERS =====

def get_model_name(model_id):
    for m in models:
        if m["id"] == model_id:
            return m["name"]
    return model_id.split("/")[-1]

def parse_error(data, prefix):
    err = data.get("error", "Khong ro")
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return prefix + ": " + msg

# ===== GITHUB FUNCTIONS =====

def github_headers():
    return {"Authorization": "token " + GIHUB_TOKEN, "Accept": "application/vnd.github.v3+json"}

def list_files(path=""):
    r = requests.get(
        "https://api.github.com/repos/" + GIHUB_REPO + "/contents/" + path,
        headers=github_headers()
    )
    return r.json() if r.status_code == 200 else None

def get_file(path):
    r = requests.get(
        "https://api.github.com/repos/" + GIHUB_REPO + "/contents/" + path,
        headers=github_headers()
    )
    if r.status_code == 200:
        data = r.json()
        return base64.b64decode(data["content"]).decode("utf-8"), data["sha"]
    return None, None

def update_file(path, content, sha, message="Update via bot"):
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    r = requests.put(
        "https://api.github.com/repos/" + GIHUB_REPO + "/contents/" + path,
        headers=github_headers(),
        json={"message": message, "content": encoded, "sha": sha}
    )
    return r.status_code == 200

def create_file_github(path, content, message="Create via bot"):
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    r = requests.put(
        "https://api.github.com/repos/" + GIHUB_REPO + "/contents/" + path,
        headers=github_headers(),
        json={"message": message, "content": encoded}
    )
    return r.status_code == 201

# ===== USER DATA PERSIST (luu/load model da chon) =====

def save_user_data():
    content, sha = get_file(USER_DATA_FILE)
    data_to_save = {}
    for uid, d in user_data.items():
        data_to_save[str(uid)] = {
            "model_id":    d.get("model_id", ""),
            "model_name":  d.get("model_name", ""),
            "provider":    d.get("provider", "nvidia"),
            "system_prompt": d.get("system_prompt", "")
        }
    content_str = json.dumps(data_to_save, ensure_ascii=False, indent=2)
    if sha:
        update_file(USER_DATA_FILE, content_str, sha, "Save user data")
    else:
        create_file_github(USER_DATA_FILE, content_str, "Create user data")

def load_user_data():
    content, _ = get_file(USER_DATA_FILE)
    if not content:
        return
    try:
        data = json.loads(content)
        for uid, d in data.items():
            user_data[int(uid)] = {
                "model_id":    d.get("model_id", ""),
                "model_name":  d.get("model_name", ""),
                "provider":    d.get("provider", "nvidia"),
                "system_prompt": d.get("system_prompt", ""),
                "messages":    []
            }
    except:
        pass

# ===== MEMORY FUNCTIONS (nho thong tin dai han) =====

def load_memory(user_id):
    content, sha = get_file(MEMORY_FILE)
    if content is None:
        return {}, None
    try:
        data = json.loads(content)
        return data.get(str(user_id), {}), sha
    except:
        return {}, None

def save_memory(user_id, memory_dict):
    content, sha = get_file(MEMORY_FILE)
    try:
        all_memory = json.loads(content) if content else {}
    except:
        all_memory = {}
    all_memory[str(user_id)] = memory_dict
    content_str = json.dumps(all_memory, ensure_ascii=False, indent=2)
    if sha:
        update_file(MEMORY_FILE, content_str, sha, "Update memory")
    else:
        create_file_github(MEMORY_FILE, content_str, "Create memory")

def memory_to_system(memory_dict):
    if not memory_dict:
        return ""
    lines = ["Thong tin nguoi dung can nho:"]
    for k, v in memory_dict.items():
        lines.append("- " + k + ": " + str(v))
    return "\n".join(lines)

# ===== AI CALL =====

def goi_nvidia(model_id, messages, timeout=45):
    try:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + NVIDIA_API_KEY, "Content-Type": "application/json"},
            json={"model": model_id, "messages": messages, "max_tokens": 1024},
            timeout=timeout
        )
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        return parse_error(data, "Loi NIM"), "error"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return "Loi ket noi: " + str(e), "error"

# ===== SKILL: WEB SEARCH (DuckDuckGo HTML) =====

def web_search(query, max_results=5):
    try:
        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.in_result = False
                self.current = ""
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and "result__a" in attrs.get("class", ""):
                    self.in_result = True
                    self.current = ""
            def handle_endtag(self, tag):
                if tag == "a" and self.in_result:
                    self.in_result = False
                    if self.current.strip():
                        self.results.append(self.current.strip())
            def handle_data(self, data):
                if self.in_result:
                    self.current += data

        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; TelegramBot/1.0)"},
            timeout=10
        )
        parser = DDGParser()
        parser.feed(r.text)
        if parser.results:
            return "\n".join("- " + t for t in parser.results[:max_results])
        return "Khong tim thay ket qua cho: " + query
    except Exception as e:
        return "Loi web search: " + str(e)

# ===== SKILL: IMAGE GENERATION (Gemini 2.0 Flash) =====

def tao_anh(prompt):
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}
            },
            timeout=60
        )
        data = r.json()
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])
        return "Loi tao anh: Khong co anh trong response"
    except requests.exceptions.Timeout:
        return "Timeout: Thu lai sau"
    except Exception as e:
        return "Loi tao anh: " + str(e)

# ===== SKILL: PHAN TICH ANH (Llama 4 Vision) =====

def phan_tich_anh(image_bytes, question="Mo ta chi tiet anh nay bang tieng Viet"):
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": question}
            ]
        }]
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + NVIDIA_API_KEY, "Content-Type": "application/json"},
            json={"model": "meta/llama-4-maverick-17b-128e-instruct", "messages": messages, "max_tokens": 1024},
            timeout=45
        )
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return parse_error(data, "Loi phan tich anh")
    except Exception as e:
        return "Loi phan tich anh: " + str(e)

# ===== BOT COMMANDS =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot AI (NVIDIA NIM)\n\n"
        "💬 CHAT AI:\n"
        "/model - Chon model\n"
        "/mymodel - Model dang dung\n"
        "/system [noi dung] - System prompt\n"
        "/reset - Xoa lich su chat\n\n"
        "🧠 MEMORY (nho dai han):\n"
        "/remember key=value - Luu thong tin\n"
        "/forget key - Xoa thong tin\n\n"
        "🔍 SKILLS:\n"
        "/search [tu khoa] - Tim kiem web\n"
        "/imagine [mo ta] - Tao anh AI\n"
        "📷 Gui anh - Phan tich anh\n\n"
        "📁 GITHUB:\n"
        "/files - Danh sach file\n"
        "/read ten_file - Doc file\n"
        "/edit ten_file - Sua file\n"
        "/push ten_file - Tao file moi\n"
        "/cancel - Huy thao tac"
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
        "Chon model NVIDIA NIM:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def xu_ly_chon_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    model = models[int(query.data.split("_")[1])]
    existing = user_data.get(user_id, {})
    user_data[user_id] = {
        "model_id":    model["id"],
        "model_name":  model["name"],
        "provider":    "nvidia",
        "messages":    existing.get("messages", []),  # giữ nguyên lich su
        "system_prompt": existing.get("system_prompt", "")
    }
    save_user_data()
    await query.edit_message_text(
        "✅ Da chon: " + model["name"] + "\n\n"
        "Nhan tin de bat dau chat!\n"
        "(Lich su chat duoc giu nguyen)"
    )

async def my_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data and user_data[user_id].get("model_id"):
        d = user_data[user_id]
        sys_info = "\nSystem: " + d["system_prompt"][:60] + "..." if d.get("system_prompt") else ""
        msg_count = len([m for m in d["messages"] if m["role"] == "user"])
        await update.message.reply_text(
            "🤖 Model: " + d["model_name"] +
            "\n💬 Tin nhan: " + str(msg_count) +
            sys_info
        )
    else:
        await update.message.reply_text("Chua chon model. Dung /model")

async def set_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        current = user_data.get(user_id, {}).get("system_prompt", "(chua dat)")
        await update.message.reply_text(
            "System prompt hien tai:\n" + current + "\n\n"
            "Dung: /system [noi dung]\n"
            "De xoa: /system clear"
        )
        return
    if context.args[0] == "clear":
        if user_id in user_data:
            user_data[user_id]["system_prompt"] = ""
            user_data[user_id]["messages"] = []
        save_user_data()
        await update.message.reply_text("✅ Da xoa system prompt!")
        return
    system_prompt = " ".join(context.args)
    if user_id not in user_data:
        user_data[user_id] = {"messages": [], "system_prompt": "", "model_id": "", "model_name": "", "provider": "nvidia"}
    user_data[user_id]["system_prompt"] = system_prompt
    user_data[user_id]["messages"] = []
    save_user_data()
    await update.message.reply_text("✅ Da dat system prompt:\n" + system_prompt + "\n\n(Lich su chat da reset)")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_data:
        user_data[user_id]["messages"] = []
    await update.message.reply_text("✅ Da xoa lich su chat!")

# ===== MEMORY COMMANDS =====

async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        memory, _ = load_memory(user_id)
        if not memory:
            await update.message.reply_text("Memory trong.\nDung: /remember ten=Nam nghe=lapTrinh")
            return
        text = "🧠 Memory cua ban:\n\n"
        for k, v in memory.items():
            text += "- " + k + ": " + str(v) + "\n"
        await update.message.reply_text(text)
        return
    memory, _ = load_memory(user_id)
    for arg in context.args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            memory[k.strip()] = v.strip()
    save_memory(user_id, memory)
    await update.message.reply_text("✅ Da luu memory!")

async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Dung: /forget key\nHoac: /forget all")
        return
    if context.args[0] == "all":
        save_memory(user_id, {})
        await update.message.reply_text("✅ Da xoa toan bo memory!")
        return
    memory, _ = load_memory(user_id)
    key = context.args[0]
    if key in memory:
        del memory[key]
        save_memory(user_id, memory)
        await update.message.reply_text("✅ Da xoa: " + key)
    else:
        await update.message.reply_text("Khong tim thay: " + key)

# ===== SKILL COMMANDS =====

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /search [tu khoa]")
        return
    query = " ".join(context.args)
    await update.message.reply_chat_action("typing")
    result = web_search(query)
    await update.message.reply_text("🔍 " + query + "\n\n" + result)

async def imagine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /imagine [mo ta]\nVi du: /imagine a dragon over mountains, epic, 4k")
        return
    prompt = " ".join(context.args)
    await update.message.reply_text("🎨 Dang tao anh...")
    await update.message.reply_chat_action("upload_photo")
    result = tao_anh(prompt)
    if isinstance(result, bytes):
        await update.message.reply_photo(photo=result, caption="🎨 " + prompt)
    else:
        await update.message.reply_text(result)

# ===== GITHUB COMMANDS =====

async def files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = list_files()
    if not result:
        await update.message.reply_text("Loi: Khong the lay danh sach file")
        return
    text = "📁 File trong repo:\n\n"
    for f in result:
        icon = "📂" if f["type"] == "dir" else "📄"
        text += icon + " " + f["name"] + "\n"
    await update.message.reply_text(text)

async def read_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /read ten_file")
        return
    content, _ = get_file(context.args[0])
    if content is None:
        await update.message.reply_text("Loi: Khong tim thay file " + context.args[0])
        return
    if len(content) > 4000:
        content = content[:4000] + "\n...(con tiep)"
    await update.message.reply_text("📄 " + context.args[0] + ":\n\n" + content)

async def edit_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Dung: /edit ten_file")
        return
    path = context.args[0]
    content, sha = get_file(path)
    if content is None:
        await update.message.reply_text("Loi: Khong tim thay file " + path)
        return
    edit_state[user_id] = {"mode": "edit", "path": path, "sha": sha}
    preview = content[:1500] + "\n...(con tiep)" if len(content) > 1500 else content
    await update.message.reply_text(
        "✏️ Dang sua: " + path + "\n\nNoi dung hien tai:\n" + preview +
        "\n\nGui noi dung MOI de thay the\nHoac /cancel de huy"
    )

async def push_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Dung: /push ten_file")
        return
    edit_state[user_id] = {"mode": "create", "path": context.args[0]}
    await update.message.reply_text("➕ Tao file: " + context.args[0] + "\n\nGui noi dung\nHoac /cancel de huy")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in edit_state:
        del edit_state[user_id]
    await update.message.reply_text("Da huy!")

# ===== XU LY ANH =====

async def xu_ly_anh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_chat_action("typing")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    r = requests.get(file.file_path)
    question = update.message.caption or "Mo ta chi tiet anh nay bang tieng Viet"
    result = phan_tich_anh(r.content, question)
    await update.message.reply_text("🔍 Phan tich anh:\n\n" + result)

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
            await update.message.reply_text("✅ Da cap nhat " + state["path"] + "!" if success else "❌ Loi cap nhat!")
        elif state["mode"] == "create":
            success = create_file_github(state["path"], tin_nhan)
            del edit_state[user_id]
            await update.message.reply_text("✅ Da tao " + state["path"] + "!" if success else "❌ Loi tao file!")
        return

    # Lenh tat (khong can /)
    tin_lower = tin_nhan.strip().lower()

    if tin_lower.startswith("imagine "):
        prompt = tin_nhan[8:].strip()
        await update.message.reply_text("🎨 Dang tao anh...")
        await update.message.reply_chat_action("upload_photo")
        result = tao_anh(prompt)
        if isinstance(result, bytes):
            await update.message.reply_photo(photo=result, caption="🎨 " + prompt)
        else:
            await update.message.reply_text(result)
        return

    if tin_lower.startswith("search "):
        query = tin_nhan[7:].strip()
        await update.message.reply_chat_action("typing")
        result = web_search(query)
        await update.message.reply_text("🔍 " + query + "\n\n" + result)
        return

    if tin_lower.startswith("remember "):
        args = tin_nhan[9:].strip().split()
        memory, _ = load_memory(user_id)
        for arg in args:
            if "=" in arg:
                k, v = arg.split("=", 1)
                memory[k.strip()] = v.strip()
        save_memory(user_id, memory)
        await update.message.reply_text("✅ Da luu memory!")
        return

    if tin_lower == "remember":
        memory, _ = load_memory(user_id)
        if not memory:
            await update.message.reply_text("Memory trong. Vi du: remember ten=Nam")
            return
        text = "🧠 Memory:\n\n"
        for k, v in memory.items():
            text += "- " + k + ": " + str(v) + "\n"
        await update.message.reply_text(text)
        return

    if tin_lower.startswith("forget "):
        key = tin_nhan[7:].strip()
        if key == "all":
            save_memory(user_id, {})
            await update.message.reply_text("✅ Da xoa toan bo memory!")
        else:
            memory, _ = load_memory(user_id)
            if key in memory:
                del memory[key]
                save_memory(user_id, memory)
                await update.message.reply_text("✅ Da xoa: " + key)
            else:
                await update.message.reply_text("Khong tim thay: " + key)
        return

    # Chat AI
    if user_id not in user_data or not user_data[user_id].get("model_id"):
        await update.message.reply_text("Chua chon model! Dung /model truoc.")
        return

    d = user_data[user_id]
    d["messages"].append({"role": "user", "content": tin_nhan})
    await update.message.reply_chat_action("typing")

    # Ghep system prompt + memory
    messages_to_send = []
    memory, _ = load_memory(user_id)
    memory_text = memory_to_system(memory)
    system_parts = []
    if d.get("system_prompt"):
        system_parts.append(d["system_prompt"])
    if memory_text:
        system_parts.append(memory_text)
    if system_parts:
        messages_to_send.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages_to_send.extend(d["messages"])

    # Goi AI
    tra_loi, err_type = goi_nvidia(d["model_id"], messages_to_send)

    if err_type == "timeout":
        d["messages"].pop()
        await update.message.reply_text("⏱ Timeout! Thu lai hoac doi /model khac.")
        return
    if err_type == "error":
        d["messages"].pop()
        await update.message.reply_text("❌ Loi: " + tra_loi)
        return

    d["messages"].append({"role": "assistant", "content": tra_loi})
    if len(tra_loi) > 4096:
        tra_loi = tra_loi[:4090] + "..."
    await update.message.reply_text(d["model_name"] + ":\n\n" + tra_loi)

# ===== CHAY BOT =====
print("Dang load user data tu GitHub...")
load_user_data()
print("Bot dang chay...")

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", chon_model))
app.add_handler(CommandHandler("mymodel", my_model))
app.add_handler(CommandHandler("system", set_system))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CommandHandler("remember", remember))
app.add_handler(CommandHandler("forget", forget))
app.add_handler(CommandHandler("search", search))
app.add_handler(CommandHandler("imagine", imagine))
app.add_handler(CommandHandler("files", files))
app.add_handler(CommandHandler("read", read_file))
app.add_handler(CommandHandler("edit", edit_file))
app.add_handler(CommandHandler("push", push_file))
app.add_handler(CommandHandler("cancel", cancel))
app.add_handler(CallbackQueryHandler(xu_ly_chon_model, pattern="^model_"))
app.add_handler(MessageHandler(filters.PHOTO, xu_ly_anh))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, xu_ly_tin_nhan))

app.run_polling()
