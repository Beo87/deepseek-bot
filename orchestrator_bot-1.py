"""
╔══════════════════════════════════════════════════════════════╗
║           MULTI-AGENT ORCHESTRATOR BOT                      ║
║           Version: 2.0.0 | Author: Generated                ║
║           Stack: python-telegram-bot v20+ | NVIDIA NIM      ║
╚══════════════════════════════════════════════════════════════╝

ENV VARIABLES REQUIRED:
  TELEGRAM_TOKEN   — Bot token từ @BotFather
  NVIDIA_API_KEY   — API key từ build.nvidia.com
  GITHUB_TOKEN     — (Optional) GitHub personal access token
  GITHUB_OWNER     — (Optional) GitHub username
  GITHUB_REPO      — (Optional) Repo lưu memory, mặc định "bot-memory"
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.error import Conflict, NetworkError, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("OrchestratorBot")


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    telegram_token: str
    nvidia_api_key: str
    github_token: Optional[str] = None
    github_owner: Optional[str] = None
    github_repo: str = "bot-memory"

    # AI settings
    nvidia_api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    model: str = "minimaxai/minimax-m2.7"
    temperature: float = 0.7
    max_tokens: int = 1500
    request_timeout: int = 45

    # Memory settings
    memory_limit_per_user: int = 50
    history_context_limit: int = 5

    # Rate limiting
    rate_limit_messages: int = 10   # max messages
    rate_limit_window: int = 60     # per N seconds

    @classmethod
    def from_env(cls) -> "Config":
        token = os.getenv("TELEGRAM_TOKEN", "").strip()
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()

        if not token:
            logger.critical("❌ Thiếu TELEGRAM_TOKEN")
            sys.exit(1)
        if not api_key:
            logger.warning("⚠️  Thiếu NVIDIA_API_KEY — tính năng AI sẽ không hoạt động")

        return cls(
            telegram_token=token,
            nvidia_api_key=api_key,
            github_token=os.getenv("GITHUB_TOKEN", "").strip() or None,
            github_owner=os.getenv("GITHUB_OWNER", "").strip() or None,
            github_repo=os.getenv("GITHUB_REPO", "bot-memory").strip(),
        )


# ══════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """Token bucket per user."""

    def __init__(self, max_calls: int, window_seconds: int):
        self._max = max_calls
        self._window = window_seconds
        self._records: dict[int, list[float]] = {}

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        history = self._records.setdefault(user_id, [])
        # Drop expired timestamps
        self._records[user_id] = [t for t in history if now - t < self._window]
        if len(self._records[user_id]) >= self._max:
            return False
        self._records[user_id].append(now)
        return True

    def seconds_until_reset(self, user_id: int) -> int:
        history = self._records.get(user_id, [])
        if not history:
            return 0
        oldest = min(history)
        return max(0, int(self._window - (time.monotonic() - oldest)))


# ══════════════════════════════════════════════════════════════
# MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemorySystem:
    """
    In-memory store with optional GitHub persistence.
    Async-safe: GitHub I/O chạy trên thread pool.
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._data: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

    # ── GitHub helpers ──────────────────────────────────────────

    def _gh_headers(self) -> dict:
        return {
            "Authorization": f"token {self._cfg.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _gh_url(self) -> str:
        return (
            f"https://api.github.com/repos/"
            f"{self._cfg.github_owner}/{self._cfg.github_repo}"
            f"/contents/memory.json"
        )

    async def load(self) -> None:
        if not self._cfg.github_token or not self._cfg.github_owner:
            logger.info("GitHub không cấu hình — dùng bộ nhớ tạm (mất khi restart)")
            return

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(self._gh_url(), headers=self._gh_headers())
                if resp.status_code == 200:
                    payload = resp.json()
                    raw = base64.b64decode(payload["content"]).decode("utf-8")
                    async with self._lock:
                        self._data = json.loads(raw)
                    logger.info(f"✅ Đã tải memory từ GitHub ({len(self._data)} users)")
                elif resp.status_code == 404:
                    logger.info("memory.json chưa tồn tại trên GitHub — sẽ tạo khi lưu lần đầu")
                else:
                    logger.warning(f"GitHub load trả về status {resp.status_code}")
        except Exception as e:
            logger.error(f"Lỗi tải memory từ GitHub: {e}")

    async def save(self) -> None:
        if not self._cfg.github_token or not self._cfg.github_owner:
            return

        async with self._lock:
            snapshot = json.dumps(self._data, ensure_ascii=False, indent=2)

        encoded = base64.b64encode(snapshot.encode()).decode()
        body: dict = {
            "message": f"chore: update memory [{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
            "content": encoded,
            "branch": "main",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Lấy SHA để update (nếu file đã tồn tại)
                check = await client.get(self._gh_url(), headers=self._gh_headers())
                if check.status_code == 200:
                    body["sha"] = check.json().get("sha", "")

                resp = await client.put(self._gh_url(), json=body, headers=self._gh_headers())
                if resp.status_code in (200, 201):
                    logger.debug("✅ Memory đã lưu lên GitHub")
                else:
                    logger.error(f"Lỗi lưu GitHub {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Lỗi kết nối GitHub khi lưu: {e}")

    # ── Public API ──────────────────────────────────────────────

    async def add(self, user_id: int, text: str) -> None:
        uid = str(user_id)
        async with self._lock:
            if uid not in self._data:
                self._data[uid] = []
            self._data[uid].append(MemoryEntry(text=text).__dict__)
            # Giữ N tin gần nhất
            self._data[uid] = self._data[uid][-self._cfg.memory_limit_per_user:]

        # Lưu GitHub nền (không await để không block bot)
        asyncio.create_task(self.save())

    async def get(self, user_id: int, limit: int | None = None) -> list[dict]:
        limit = limit or self._cfg.history_context_limit
        async with self._lock:
            return list(self._data.get(str(user_id), [])[-limit:])

    async def clear(self, user_id: int) -> bool:
        uid = str(user_id)
        async with self._lock:
            if uid not in self._data:
                return False
            del self._data[uid]
        asyncio.create_task(self.save())
        return True

    async def stats(self, user_id: int) -> dict:
        async with self._lock:
            entries = self._data.get(str(user_id), [])
        return {
            "count": len(entries),
            "oldest": entries[0]["timestamp"][:16] if entries else None,
            "newest": entries[-1]["timestamp"][:16] if entries else None,
        }


# ══════════════════════════════════════════════════════════════
# AI ENGINE
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are the ORCHESTRATOR of a multi-agent AI system powered by specialist agents.
Respond in the SAME LANGUAGE the user writes in (Vietnamese if they write Vietnamese).

━━━ AGENT ROSTER ━━━
[ANALYST]    → Decompose ambiguous or multi-step problems into clear subtasks.
[RESEARCHER] → Gather, verify, and synthesize factual information.
[CODER]      → Write, review, debug, and explain code in any language.
[WRITER]     → Draft, edit, translate, or improve written content.
[CRITIC]     → Review all outputs for errors, gaps, or improvements. Runs LAST.

━━━ WORKFLOW ━━━
1. ROUTE   – Identify which agents are needed (1–3 max).
2. EXECUTE – Each agent completes its subtask internally.
3. CRITIC  – Validate and improve the combined result.
4. REPLY   – Deliver one clean, concise answer. No agent labels in final output.

━━━ RULES ━━━
• Think step-by-step before responding.
• ANALYST runs first when the request is ambiguous.
• CRITIC is never skipped.
• Never hallucinate — say "Tôi không biết" when uncertain.
• Be helpful, concise, and human-friendly.
• Use markdown where it improves readability (code blocks, bullet lists).
"""


class AIEngine:
    """Async wrapper cho NVIDIA NIM API với retry logic."""

    def __init__(self, config: Config):
        self._cfg = config
        self._headers = {
            "Authorization": f"Bearer {config.nvidia_api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        user_input: str,
        history: list[dict],
        *,
        retries: int = 2,
    ) -> str:
        """
        Gọi API với context history.
        Retry tối đa `retries` lần khi gặp lỗi mạng.
        """
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY. Bot không thể xử lý yêu cầu này."

        # Xây dựng context từ lịch sử
        context_lines = [f"• {h['text'][:80]}" for h in history]
        context_block = "\n".join(context_lines) if context_lines else "(Không có lịch sử)"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[Lịch sử gần đây]\n{context_block}\n\n"
                    f"[Yêu cầu hiện tại]\n{user_input}"
                ),
            },
        ]

        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }

        last_error: str = "Lỗi không xác định"

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._cfg.request_timeout) as client:
                    resp = await client.post(
                        self._cfg.nvidia_api_url,
                        json=payload,
                        headers=self._headers,
                    )
                    if resp.status_code == 429:
                        return "⏳ API đang quá tải. Vui lòng thử lại sau vài giây."
                    if resp.status_code == 401:
                        return "❌ NVIDIA API Key không hợp lệ hoặc đã hết hạn."
                    if resp.status_code >= 500:
                        last_error = f"Server NVIDIA lỗi (HTTP {resp.status_code})"
                        await asyncio.sleep(2 ** attempt)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    choices = data.get("choices", [])
                    if not choices:
                        return "⚠️ API không trả về kết quả hợp lệ."

                    return choices[0]["message"]["content"].strip()

            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning(f"API timeout (attempt {attempt + 1}/{retries + 1})")
                await asyncio.sleep(1)
            except httpx.HTTPError as e:
                last_error = str(e)
                logger.error(f"Lỗi mạng khi gọi API: {e}")
                await asyncio.sleep(1)

        if last_error == "timeout":
            return "⏱️ Bot đang quá tải hoặc mạng chậm. Vui lòng thử lại."
        return f"❌ Lỗi hệ thống: {last_error}"


# ══════════════════════════════════════════════════════════════
# BOT HANDLERS
# ══════════════════════════════════════════════════════════════

class OrchestratorBot:
    """Encapsulate toàn bộ logic bot."""

    def __init__(self, config: Config):
        self._cfg = config
        self._memory = MemorySystem(config)
        self._ai = AIEngine(config)
        self._rate = RateLimiter(config.rate_limit_messages, config.rate_limit_window)

    # ── /start ──────────────────────────────────────────────────

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        name = update.effective_user.first_name or "bạn"
        await update.message.reply_text(
            f"👋 Xin chào *{name}*\\! Tôi là **Multi\\-Agent Orchestrator Bot**\\.\n\n"
            "🧠 Tôi sử dụng 5 agent chuyên biệt để xử lý mọi yêu cầu:\n"
            "• `ANALYST` — Phân tích vấn đề\n"
            "• `RESEARCHER` — Tìm kiếm thông tin\n"
            "• `CODER` — Viết & sửa code\n"
            "• `WRITER` — Soạn thảo văn bản\n"
            "• `CRITIC` — Kiểm tra chất lượng\n\n"
            "📋 *Lệnh khả dụng:*\n"
            "/agents — Xem chi tiết các agent\n"
            "/memory — Xem lịch sử trò chuyện\n"
            "/clear — Xóa lịch sử\n"
            "/stats — Thống kê sử dụng\n"
            "/help — Hướng dẫn\n\n"
            "💬 Gửi tin nhắn bất kỳ để bắt đầu\\!",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── /agents ────────────────────────────────────────────────

    async def cmd_agents(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "🤖 *DANH SÁCH AGENT*\n\n"
            "*1\\. ANALYST*\n"
            "Phân tích vấn đề phức tạp, chia nhỏ thành các bước rõ ràng\\.\n\n"
            "*2\\. RESEARCHER*\n"
            "Tổng hợp thông tin chính xác, tránh hallucination\\.\n\n"
            "*3\\. CODER*\n"
            "Viết, review, debug code mọi ngôn ngữ lập trình\\.\n\n"
            "*4\\. WRITER*\n"
            "Soạn thảo, chỉnh sửa, dịch thuật nội dung văn bản\\.\n\n"
            "*5\\. CRITIC*\n"
            "Luôn chạy cuối cùng để kiểm tra và cải thiện đầu ra\\.\n\n"
            "💡 Hệ thống tự động điều phối agent phù hợp với yêu cầu của bạn\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── /memory ────────────────────────────────────────────────

    async def cmd_memory(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        history = await self._memory.get(uid, limit=10)

        if not history:
            await update.message.reply_text("📭 Chưa có lịch sử trò chuyện nào.")
            return

        lines = ["📜 *Lịch sử gần đây:*\n"]
        for i, h in enumerate(history, 1):
            ts = h["timestamp"][11:16]
            preview = h["text"][:60].replace("*", "\\*").replace("_", "\\_")
            ellipsis = "…" if len(h["text"]) > 60 else ""
            lines.append(f"`{i:02d}` \\[{ts}\\] {preview}{ellipsis}")

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── /clear ─────────────────────────────────────────────────

    async def cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        cleared = await self._memory.clear(uid)
        if cleared:
            await update.message.reply_text("🗑️ Đã xóa toàn bộ lịch sử trò chuyện.")
        else:
            await update.message.reply_text("ℹ️ Không có gì để xóa.")

    # ── /stats ─────────────────────────────────────────────────

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        s = await self._memory.stats(uid)
        github_status = "✅ Kết nối" if self._cfg.github_token else "❌ Không cấu hình"
        ai_status = "✅ Sẵn sàng" if self._cfg.nvidia_api_key else "❌ Không cấu hình"

        await update.message.reply_text(
            f"📊 *Thống kê của bạn*\n\n"
            f"💬 Tin nhắn đã lưu: `{s['count']}`\n"
            f"🕐 Cũ nhất: `{s['oldest'] or 'N/A'}`\n"
            f"🕐 Mới nhất: `{s['newest'] or 'N/A'}`\n\n"
            f"*Hệ thống:*\n"
            f"🤖 AI Engine: {ai_status}\n"
            f"💾 GitHub Backup: {github_status}\n"
            f"📦 Model: `{self._cfg.model}`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── /help ──────────────────────────────────────────────────

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "📖 *HƯỚNG DẪN SỬ DỤNG*\n\n"
            "*Gửi tin nhắn thường:*\n"
            "Bot sẽ tự phân tích và chọn agent phù hợp\\.\n\n"
            "*Ví dụ yêu cầu:*\n"
            "• `Viết hàm Python đọc file CSV`\n"
            "• `Giải thích thuật toán Quick Sort`\n"
            "• `Tóm tắt lý thuyết machine learning`\n"
            "• `Dịch đoạn văn này sang tiếng Anh`\n\n"
            "*Giới hạn:*\n"
            f"• Tối đa {self._cfg.rate_limit_messages} tin/{self._cfg.rate_limit_window}s\n"
            f"• Lưu {self._cfg.memory_limit_per_user} tin gần nhất\n\n"
            "💡 _Mẹo: Càng mô tả rõ, bot xử lý càng tốt\\._",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── Message handler ────────────────────────────────────────

    async def handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        user_input = update.message.text.strip()

        if not user_input:
            return

        # Rate limiting
        if not self._rate.is_allowed(user.id):
            wait = self._rate.seconds_until_reset(user.id)
            await update.message.reply_text(
                f"⏳ Bạn gửi quá nhanh. Vui lòng chờ *{wait}* giây.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        # Lưu tin nhắn
        await self._memory.add(user.id, user_input)

        # Hiện typing indicator
        await update.message.chat.send_action("typing")

        logger.info(f"User {user.id} (@{user.username}): {user_input[:80]}")

        # Lấy lịch sử context
        history = await self._memory.get(user.id, limit=self._cfg.history_context_limit - 1)
        # Loại bỏ tin hiện tại khỏi context (đã thêm vào)
        history = history[:-1] if history else []

        # Gọi AI
        response = await self._ai.complete(user_input, history)

        # Gửi trả lời (chia nhỏ nếu > 4096 ký tự)
        await self._send_long_message(update, response)

    async def _send_long_message(self, update: Update, text: str) -> None:
        """Tự động chia nhỏ tin nhắn dài > 4096 ký tự."""
        max_len = 4000
        if len(text) <= max_len:
            try:
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            except TelegramError:
                # Fallback: gửi plain text nếu markdown lỗi
                await update.message.reply_text(text)
            return

        # Chia theo dòng để tránh cắt giữa chừng
        chunks: list[str] = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > max_len:
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            suffix = f"\n\n_({i+1}/{len(chunks)})_" if len(chunks) > 1 else ""
            try:
                await update.message.reply_text(chunk + suffix, parse_mode=ParseMode.MARKDOWN)
            except TelegramError:
                await update.message.reply_text(chunk + suffix)

    # ── Error handler ──────────────────────────────────────────

    async def handle_error(self, update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        error = ctx.error

        if isinstance(error, Conflict):
            logger.critical(
                "❌ CONFLICT: Có instance bot khác đang chạy! "
                "Tắt instance đó hoặc xóa webhook tại: "
                f"https://api.telegram.org/bot{self._cfg.telegram_token}/deleteWebhook"
            )
            return

        if isinstance(error, NetworkError):
            logger.warning(f"Network error (tạm thời): {error}")
            return

        logger.error(f"Unhandled error: {error}", exc_info=error)

        if isinstance(update, Update) and update.message:
            try:
                await update.message.reply_text(
                    "❌ Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau."
                )
            except Exception:
                pass

    # ── Build & Run ────────────────────────────────────────────

    def build_app(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self._cfg.telegram_token)
            .concurrent_updates(True)
            .post_init(self._post_init)
            .build()
        )

        # Commands
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("agents", self.cmd_agents))
        app.add_handler(CommandHandler("myagents", self.cmd_agents))  # alias
        app.add_handler(CommandHandler("memory", self.cmd_memory))
        app.add_handler(CommandHandler("clear", self.cmd_clear))
        app.add_handler(CommandHandler("stats", self.cmd_stats))

        # Messages
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Errors
        app.add_error_handler(self.handle_error)

        return app

    async def _post_init(self, app: Application) -> None:
        """
        Hook chạy SAU khi Application khởi tạo, TRƯỚC khi polling.
        Cách duy nhất an toàn để chạy async setup trong PTB v20+.
        """
        logger.info("🔧 Đang khởi tạo bot...")
        await self._memory.load()
        await app.bot.set_my_commands([
            BotCommand("start",  "Màn hình chào mừng"),
            BotCommand("agents", "Danh sách agent"),
            BotCommand("memory", "Xem lịch sử chat"),
            BotCommand("clear",  "Xóa lịch sử"),
            BotCommand("stats",  "Thống kê sử dụng"),
            BotCommand("help",   "Hướng dẫn sử dụng"),
        ])
        logger.info(f"✅ Bot sẵn sàng | Model: {self._cfg.model}")

    def run(self) -> None:
        """
        PTB v20+ tự quản lý event loop nội bộ.
        KHÔNG dùng asyncio.run() hay await run_polling().
        """
        app = self.build_app()
        # run_polling() là synchronous — tự tạo & quản lý event loop
        app.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
        )


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main() -> None:
    config = Config.from_env()
    bot = OrchestratorBot(config)
    try:
        bot.run()   # ← synchronous, KHÔNG asyncio.run()
    except KeyboardInterrupt:
        logger.info("⛔ Bot đã dừng bởi người dùng (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 Bot crash: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
