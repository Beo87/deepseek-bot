import os
import json
import time
import logging
import requests
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- BIẾN MÔI TRƯỜNG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "bot-memory")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")

if not TELEGRAM_TOKEN:
    logger.error("Thiếu TELEGRAM_TOKEN. Vui lòng thiết lập biến môi trường.")
if not NVIDIA_API_KEY:
    logger.warning("Thiếu NVIDIA_API_KEY. Các tính năng AI sẽ không hoạt động.")

# --- CẤU HÌNH API ---
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

# --- BỘ NHỚ (GITHUB BACKUP) ---
class MemorySystem:
    def __init__(self):
        self.memory = {}
        self.load_from_github()

    def load_from_github(self):
        if not GITHUB_TOKEN or not GITHUB_OWNER:
            logger.info("Không có thông tin GitHub, sử dụng bộ nhớ tạm.")
            return
        
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/memory.json"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                content = response.json()
                # Giải mã nội dung base64 nếu cần (ở đây giả sử trả về JSON trực tiếp hoặc cần decode)
                # Lưu ý: GitHub API trả về content dưới dạng base64
                import base64
                decoded = base64.b64decode(content['content']).decode('utf-8')
                self.memory = json.loads(decoded)
                logger.info("Đã tải bộ nhớ từ GitHub thành công.")
            else:
                logger.info("Chưa có file memory.json trên GitHub, tạo mới.")
        except Exception as e:
            logger.error(f"Lỗi tải bộ nhớ: {e}")

    def save_to_github(self):
        if not GITHUB_TOKEN or not GITHUB_OWNER:
            return
            
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/memory.json"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        content_json = json.dumps(self.memory)
        import base64
        content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')

        # Kiểm tra xem file đã tồn tại để update hay create
        sha = ""
        check_resp = requests.get(url, headers=headers)
        if check_resp.status_code == 200:
            sha = check_resp.json().get('sha', '')

        data = {
            "message": "Update memory",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            data["sha"] = sha

        try:
            resp = requests.put(url, json=data, headers=headers)
            if resp.status_code in [200, 201]:
                logger.info("Đã lưu bộ nhớ lên GitHub.")
            else:
                logger.error(f"Lỗi lưu GitHub: {resp.text}")
        except Exception as e:
            logger.error(f"Lỗi kết nối GitHub khi lưu: {e}")

    def add(self, user_id, text):
        if str(user_id) not in self.memory:
            self.memory[str(user_id)] = []
        self.memory[str(user_id)].append({
            "time": datetime.now().isoformat(),
            "text": text
        })
        # Giới hạn 50 tin gần nhất
        self.memory[str(user_id)] = self.memory[str(user_id)][-50:]
        self.save_to_github()

    def get(self, user_id, limit=5):
        history = self.memory.get(str(user_id), [])
        return history[-limit:]

memory_db = MemorySystem()

# --- HỆ THỐNG MULTI-AGENT ---

SYSTEM_PROMPT_ORCHESTRATOR = """
You are the ORCHESTRATOR of a multi-agent AI system.
Your job is to analyze user requests and delegate tasks to the most suitable specialist agent.

AGENT ROSTER:
1. [ANALYST]: Break down complex problems. Trigger: Ambiguous/Multi-step requests.
2. [RESEARCHER]: Gather facts/data. Trigger: Needs factual context.
3. [CODER]: Write/Debug code. Trigger: Programming tasks.
4. [WRITER]: Draft/Edit content. Trigger: Writing tasks.
5. [CRITIC]: Review outputs for errors. Trigger: Always runs LAST before final reply.

WORKFLOW:
1. ROUTE: Choose 1-3 agents.
2. EXECUTE: Simulate the agents' work internally.
3. CRITIC: Review the result.
4. FINAL ANSWER: Provide a clean, merged response without agent labels.

RULES:
- Think step-by-step.
- If uncertain, use ANALYST first.
- Never hallucinate.
- Be concise and human-friendly.
"""

def call_nvidia_api(messages, model="meta/llama-3.1-70b-instruct"):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(NVIDIA_API_URL, json=payload, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Kiểm tra cấu trúc phản hồi an toàn
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"], None
        else:
            logger.warning(f"API trả về định dạng lạ: {data}")
            return "Xin lỗi, tôi không nhận được phản hồi hợp lệ từ AI.", None
            
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        logger.error(f"Lỗi API: {e}")
        return None, str(e)

async def handle_orchestrator_logic(user_input, user_id):
    # Lấy lịch sử ngắn hạn
    history = memory_db.get(user_id, limit=3)
    context = "\n".join([f"- {h['text']}" for h in history])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ORCHESTRATOR},
        {"role": "user", "content": f"User History:\n{context}\n\nCurrent Request: {user_input}"}
    ]
    
    response, error = call_nvidia_api(messages)
    
    if error == "timeout":
        return "Bot đang quá tải, vui lòng thử lại sau ít phút."
    elif error:
        return f"Lỗi hệ thống: {error}"
    
    return response

# --- XỬ LÝ TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Chào mừng đến với Multi-Agent Bot!\n\n"
        "Các lệnh khả dụng:\n"
        "/orchestrator - Bật chế độ suy nghĩ đa tác tử\n"
        "/myagents - Xem danh sách agent\n"
        "/memory - Xem lịch sử chat\n"
        "/clear - Xóa lịch sử chat\n"
        "\nGửi tin nhắn bất kỳ để bắt đầu!"
    )

async def my_agents(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 **DANH SÁCH AGENT**\n\n"
        "1. **ANALYST**: Phân tích vấn đề phức tạp.\n"
        "2. **RESEARCHER**: Tìm kiếm thông tin chính xác.\n"
        "3. **CODER**: Viết và sửa code.\n"
        "4. **WRITER**: Soạn thảo văn bản.\n"
        "5. **CRITIC**: Kiểm tra chất lượng đầu ra.\n\n"
        "Hệ thống tự động điều phối các agent này khi bạn yêu cầu."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = memory_db.get(user_id, limit=10)
    
    if not history:
        await update.message.reply_text("📭 Chưa có lịch sử trò chuyện nào.")
        return
    
    text = "📜 **LỊCH SỬ GẦN ĐÂY**\n\n"
    for h in history:
        time_str = h['time'].split('T')[1][:5]
        text += f"[{time_str}] {h['text'][:50]}...\n"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if str(user_id) in memory_db.memory:
        del memory_db.memory[str(user_id)]
        memory_db.save_to_github()
        await update.message.reply_text("🗑️ Đã xóa lịch sử trò chuyện của bạn.")
    else:
        await update.message.reply_text("Không có gì để xóa.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    user_id = update.effective_user.id
    
    # Lưu vào memory
    memory_db.add(user_id, user_input)
    
    # Gửi trạng thái đang gõ
    await update.message.chat.send_action(action='typing')
    
    # Xử lý logic Orchestrator
    response_text = await handle_orchestrator_logic(user_input, user_id)
    
    # Trả lời
    await update.message.reply_text(response_text)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("Đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại.")

# --- MAIN ---
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("orchestrator", start)) # Alias
    app.add_handler(CommandHandler("myagents", my_agents))
    app.add_handler(CommandHandler("memory", show_memory))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.add_error_handler(error_handler)
    
    logger.info("Bot đang chạy...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
