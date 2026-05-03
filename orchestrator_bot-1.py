"""
╔══════════════════════════════════════════════════════════════╗
║           MULTI-AGENT ORCHESTRATOR BOT                      ║
║           Version: 3.0.0                                    ║
║           Stack: python-telegram-bot v20+ | NVIDIA NIM      ║
╚══════════════════════════════════════════════════════════════╝

TÍNH NĂNG:
  • Chat AI đa agent (Analyst, Researcher, Coder, Writer, Critic)
  • Đọc & phân tích ảnh (vision AI)
  • Đọc file code / text và xử lý
  • Xuất file theo yêu cầu (.py, .js, .txt, .md, .html, .json, ...)
  • Lịch sử hội thoại + GitHub backup
  • Rate limiting

ENV VARIABLES:
  TELEGRAM_TOKEN   — Bot token từ @BotFather         (bắt buộc)
  NVIDIA_API_KEY   — API key từ build.nvidia.com      (bắt buộc)
  GITHUB_TOKEN     — GitHub personal access token     (tùy chọn)
  GITHUB_OWNER     — GitHub username                  (tùy chọn)
  GITHUB_REPO      — Repo lưu memory [bot-memory]     (tùy chọn)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from telegram import BotCommand, Document, Message, PhotoSize, Update
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
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# Các extension file code/text được hỗ trợ đọc
SUPPORTED_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".html", ".css", ".scss", ".less", ".xml", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".md", ".txt", ".rst", ".csv", ".log", ".dockerfile",
    ".gitignore", ".htaccess",
}

# Từ khoá xuất file — bot sẽ gửi file thay vì text
OUTPUT_KEYWORDS = {
    "xuất file", "tạo file", "save file", "export file", "lưu file",
    "ghi ra file", "tạo ra file", "viết ra file", "xuất ra file",
    "save as", "tạo script", "tạo code file", "download file",
    "generate file", "create file", "output file",
}

# Extension map từ ngôn ngữ → đuôi file
LANG_TO_EXT: dict[str, str] = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "java": ".java",
    "cpp": ".cpp", "c++": ".cpp",
    "c": ".c",
    "csharp": ".cs", "c#": ".cs",
    "go": ".go",
    "rust": ".rs",
    "ruby": ".rb",
    "php": ".php",
    "swift": ".swift",
    "kotlin": ".kt",
    "html": ".html",
    "css": ".css",
    "sql": ".sql",
    "bash": ".sh", "shell": ".sh", "sh": ".sh",
    "json": ".json",
    "yaml": ".yaml", "yml": ".yaml",
    "markdown": ".md", "md": ".md",
    "text": ".txt", "txt": ".txt",
    "xml": ".xml",
    "toml": ".toml",
    "powershell": ".ps1",
}

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


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

    # AI — text model
    nvidia_api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    model: str = "meta/llama-4-maverick-17b-128e-instruct"
    # AI — vision model (hỗ trợ ảnh)
    vision_model: str = "meta/llama-4-maverick-17b-128e-instruct"

    temperature: float = 0.7
    max_tokens: int = 2000
    request_timeout: int = 60

    # Memory
    memory_limit_per_user: int = 50
    history_context_limit: int = 5

    # Rate limiting
    rate_limit_messages: int = 10
    rate_limit_window: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        token = os.getenv("TELEGRAM_TOKEN", "").strip()
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()

        if not token:
            logger.critical("❌ Thiếu TELEGRAM_TOKEN")
            sys.exit(1)
        if not api_key:
            logger.warning("⚠️  Thiếu NVIDIA_API_KEY — AI sẽ không hoạt động")

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
    def __init__(self, max_calls: int, window_seconds: int):
        self._max = max_calls
        self._window = window_seconds
        self._records: dict[int, list[float]] = {}

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        self._records[user_id] = [
            t for t in self._records.get(user_id, [])
            if now - t < self._window
        ]
        if len(self._records[user_id]) >= self._max:
            return False
        self._records[user_id].append(now)
        return True

    def seconds_until_reset(self, user_id: int) -> int:
        history = self._records.get(user_id, [])
        if not history:
            return 0
        return max(0, int(self._window - (time.monotonic() - min(history))))


# ══════════════════════════════════════════════════════════════
# MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemorySystem:
    def __init__(self, config: Config):
        self._cfg = config
        self._data: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

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
            logger.info("GitHub không cấu hình — dùng bộ nhớ tạm")
            return
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(self._gh_url(), headers=self._gh_headers())
                if resp.status_code == 200:
                    raw = base64.b64decode(resp.json()["content"]).decode()
                    async with self._lock:
                        self._data = json.loads(raw)
                    logger.info(f"✅ Memory loaded ({len(self._data)} users)")
                elif resp.status_code == 404:
                    logger.info("memory.json chưa tồn tại — sẽ tạo khi lưu lần đầu")
        except Exception as e:
            logger.error(f"Lỗi load memory: {e}")

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
                check = await client.get(self._gh_url(), headers=self._gh_headers())
                if check.status_code == 200:
                    body["sha"] = check.json().get("sha", "")
                resp = await client.put(self._gh_url(), json=body, headers=self._gh_headers())
                if resp.status_code not in (200, 201):
                    logger.error(f"GitHub save lỗi {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Lỗi save memory: {e}")

    async def add(self, user_id: int, text: str) -> None:
        uid = str(user_id)
        async with self._lock:
            if uid not in self._data:
                self._data[uid] = []
            self._data[uid].append(MemoryEntry(text=text).__dict__)
            self._data[uid] = self._data[uid][-self._cfg.memory_limit_per_user:]
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
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════

class FileProcessor:
    """Download & xử lý file/ảnh từ Telegram."""

    @staticmethod
    async def download_bytes(file_id: str, bot) -> bytes:
        """Download file từ Telegram, trả về bytes."""
        tg_file = await bot.get_file(file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        return buf.getvalue()

    @staticmethod
    def to_base64(data: bytes) -> str:
        return base64.standard_b64encode(data).decode()

    @classmethod
    async def process_photo(cls, photo: PhotoSize, bot) -> tuple[str, str]:
        """
        Download ảnh lớn nhất, trả về (base64, media_type).
        """
        data = await cls.download_bytes(photo.file_id, bot)
        return cls.to_base64(data), "image/jpeg"

    @classmethod
    async def process_document(cls, doc: Document, bot) -> tuple[str, str, str]:
        """
        Download document, trả về (content_text, filename, ext).
        Nếu file quá lớn hoặc không phải text → raise ValueError.
        """
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File quá lớn ({doc.file_size // 1024 // 1024}MB). "
                f"Giới hạn {MAX_FILE_SIZE_MB}MB."
            )

        filename = doc.file_name or "unknown"
        ext = Path(filename).suffix.lower()

        # Kiểm tra extension có hỗ trợ không
        if ext not in SUPPORTED_CODE_EXTS and not (
            doc.mime_type and doc.mime_type.startswith("text/")
        ):
            raise ValueError(
                f"Định dạng `{ext or 'không rõ'}` không được hỗ trợ.\n"
                f"Hỗ trợ: {', '.join(sorted(SUPPORTED_CODE_EXTS)[:20])}..."
            )

        data = await cls.download_bytes(doc.file_id, bot)

        # Decode text
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return data.decode(encoding), filename, ext
            except UnicodeDecodeError:
                continue

        raise ValueError("Không thể đọc file — có thể là file binary.")


# ══════════════════════════════════════════════════════════════
# FILE OUTPUT DETECTOR
# ══════════════════════════════════════════════════════════════

class FileOutputDetector:
    """Phát hiện yêu cầu xuất file và trích xuất code block từ AI response."""

    @staticmethod
    def wants_file(user_input: str) -> bool:
        """Kiểm tra user có muốn xuất file không."""
        lower = user_input.lower()
        return any(kw in lower for kw in OUTPUT_KEYWORDS)

    @staticmethod
    def extract_code_blocks(text: str) -> list[tuple[str, str]]:
        """
        Trích xuất tất cả code blocks từ markdown.
        Trả về list of (language, code).
        """
        pattern = r"```(\w*)\n?([\s\S]*?)```"
        matches = re.findall(pattern, text)
        return [(lang.lower().strip(), code.strip()) for lang, code in matches if code.strip()]

    @classmethod
    def pick_extension(cls, language: str, fallback_ext: str = ".txt") -> str:
        """Chọn extension phù hợp với ngôn ngữ."""
        return LANG_TO_EXT.get(language.lower(), fallback_ext)

    @classmethod
    def build_output_files(
        cls, ai_response: str, user_input: str
    ) -> list[tuple[bytes, str]]:
        """
        Tạo danh sách (file_bytes, filename) để gửi.
        Trả về list rỗng nếu không có gì để xuất.
        """
        blocks = cls.extract_code_blocks(ai_response)
        if not blocks:
            return []

        results = []
        timestamp = datetime.now().strftime("%H%M%S")

        for i, (lang, code) in enumerate(blocks):
            ext = cls.pick_extension(lang)
            name = f"output_{timestamp}_{i+1}{ext}" if len(blocks) > 1 else f"output_{timestamp}{ext}"
            results.append((code.encode("utf-8"), name))

        return results


# ══════════════════════════════════════════════════════════════
# AI ENGINE
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are the ORCHESTRATOR of a multi-agent AI system. \
Respond in the SAME LANGUAGE the user writes in (Vietnamese if they write Vietnamese).

━━━ AGENT ROSTER ━━━
[ANALYST]    → Decompose ambiguous or multi-step problems into clear subtasks.
[RESEARCHER] → Gather, verify, and synthesize factual information.
[CODER]      → Write, review, debug, and explain code. Always wrap code in ```lang fences.
[WRITER]     → Draft, edit, translate, or improve written content.
[CRITIC]     → Review all outputs for errors, gaps, or improvements. Runs LAST.

━━━ FILE OUTPUT RULE ━━━
When the user asks to create/export/save a file, ALWAYS wrap the complete file \
content in a proper ```language code fence so it can be extracted and sent as a file.

━━━ WORKFLOW ━━━
1. ROUTE   – Identify which agents are needed (1–3 max).
2. EXECUTE – Each agent completes its subtask.
3. CRITIC  – Validate and improve.
4. REPLY   – One clean, concise answer. No agent labels in final output.

━━━ RULES ━━━
• Think step-by-step before responding.
• CRITIC is never skipped.
• Never hallucinate — say "Tôi không biết" when uncertain.
• Use markdown formatting, especially code blocks with language tags.
"""

VISION_SYSTEM_PROMPT = """\
You are an expert AI vision analyst. Analyze the provided image thoroughly.
Respond in the SAME LANGUAGE the user writes in (Vietnamese if they write Vietnamese).

If the image contains:
- CODE → Extract, explain, find bugs, suggest improvements. Wrap code in ```lang fences.
- DIAGRAM/CHART → Describe structure, data, relationships in detail.
- SCREENSHOT/UI → Describe layout, identify issues, suggest improvements.
- DOCUMENT/TEXT → Extract and transcribe the text content.
- OTHER → Describe what you see in useful detail.

Be specific, structured, and actionable.
"""


class AIEngine:
    """Async AI caller với retry logic. Hỗ trợ text và vision."""

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
        """Text completion với context history."""
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY."

        context_lines = [f"• {h['text'][:80]}" for h in history]
        context_block = "\n".join(context_lines) if context_lines else "(Không có)"

        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[Lịch sử]\n{context_block}\n\n"
                        f"[Yêu cầu]\n{user_input}"
                    ),
                },
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }

        return await self._call(payload, retries=retries)

    async def complete_vision(
        self,
        user_input: str,
        image_b64: str,
        media_type: str = "image/jpeg",
        *,
        retries: int = 2,
    ) -> str:
        """Vision completion — gửi ảnh kèm prompt."""
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY."

        payload = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_input or "Hãy phân tích ảnh này chi tiết.",
                        },
                    ],
                },
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }

        return await self._call(payload, retries=retries)

    async def complete_with_file(
        self,
        user_input: str,
        file_content: str,
        filename: str,
        history: list[dict],
        *,
        retries: int = 2,
    ) -> str:
        """Text completion kèm nội dung file."""
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY."

        context_lines = [f"• {h['text'][:80]}" for h in history]
        context_block = "\n".join(context_lines) if context_lines else "(Không có)"

        # Giới hạn nội dung file để không vượt context window
        max_file_chars = 8000
        truncated = ""
        if len(file_content) > max_file_chars:
            truncated = f"\n\n⚠️ *[File bị cắt bớt — hiển thị {max_file_chars} ký tự đầu]*"
        file_preview = file_content[:max_file_chars]

        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[Lịch sử]\n{context_block}\n\n"
                        f"[File đính kèm: {filename}]\n"
                        f"```\n{file_preview}\n```{truncated}\n\n"
                        f"[Yêu cầu]\n{user_input or 'Hãy phân tích file này.'}"
                    ),
                },
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }

        return await self._call(payload, retries=retries)

    async def _call(self, payload: dict, *, retries: int) -> str:
        """Gọi NVIDIA API với retry logic."""
        last_error = "Lỗi không xác định"
        timeout = httpx.Timeout(self._cfg.request_timeout)

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        self._cfg.nvidia_api_url,
                        json=payload,
                        headers=self._headers,
                    )
                    if resp.status_code == 429:
                        return "⏳ API đang quá tải. Vui lòng thử lại sau vài giây."
                    if resp.status_code == 401:
                        return "❌ NVIDIA API Key không hợp lệ."
                    if resp.status_code >= 500:
                        last_error = f"Server lỗi HTTP {resp.status_code}"
                        await asyncio.sleep(2 ** attempt)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return "⚠️ API không trả về kết quả."
                    return choices[0]["message"]["content"].strip()

            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning(f"API timeout (attempt {attempt + 1})")
                await asyncio.sleep(1)
            except httpx.HTTPError as e:
                last_error = str(e)
                logger.error(f"HTTP error: {e}")
                await asyncio.sleep(1)

        return (
            "⏱️ Bot đang bận hoặc mạng chậm. Vui lòng thử lại."
            if last_error == "timeout"
            else f"❌ Lỗi hệ thống: {last_error}"
        )


# ══════════════════════════════════════════════════════════════
# BOT
# ══════════════════════════════════════════════════════════════

class OrchestratorBot:
    def __init__(self, config: Config):
        self._cfg = config
        self._memory = MemorySystem(config)
        self._ai = AIEngine(config)
        self._rate = RateLimiter(config.rate_limit_messages, config.rate_limit_window)
        self._file_proc = FileProcessor()
        self._file_detector = FileOutputDetector()

    # ── Helpers ────────────────────────────────────────────────

    def _check_rate(self, user_id: int) -> int:
        """Trả về 0 nếu cho phép, hoặc giây cần chờ."""
        if self._rate.is_allowed(user_id):
            return 0
        return self._rate.seconds_until_reset(user_id)

    async def _send_long(self, message: Message, text: str) -> None:
        """Gửi text, tự chia nhỏ nếu > 4000 ký tự."""
        max_len = 4000
        if len(text) <= max_len:
            try:
                await message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            except TelegramError:
                await message.reply_text(text)
            return

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
                await message.reply_text(chunk + suffix, parse_mode=ParseMode.MARKDOWN)
            except TelegramError:
                await message.reply_text(chunk + suffix)

    async def _maybe_send_files(
        self, message: Message, ai_response: str, user_input: str
    ) -> bool:
        """
        Nếu user yêu cầu xuất file, trích code blocks và gửi file.
        Trả về True nếu đã gửi file.
        """
        if not FileOutputDetector.wants_file(user_input):
            return False

        file_pairs = FileOutputDetector.build_output_files(ai_response, user_input)
        if not file_pairs:
            return False

        for file_bytes, filename in file_pairs:
            buf = io.BytesIO(file_bytes)
            buf.name = filename
            await message.reply_document(
                document=buf,
                filename=filename,
                caption=f"📄 `{filename}`",
                parse_mode=ParseMode.MARKDOWN,
            )

        return True

    # ── Commands ───────────────────────────────────────────────

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        name = update.effective_user.first_name or "bạn"
        await update.message.reply_text(
            f"👋 Xin chào *{name}*\\! Tôi là **Multi\\-Agent Orchestrator Bot**\\.\n\n"
            "🧠 *5 Agent chuyên biệt:*\n"
            "• `ANALYST` — Phân tích vấn đề\n"
            "• `RESEARCHER` — Tìm kiếm thông tin\n"
            "• `CODER` — Viết & sửa code\n"
            "• `WRITER` — Soạn thảo văn bản\n"
            "• `CRITIC` — Kiểm tra chất lượng\n\n"
            "📁 *Tôi có thể xử lý:*\n"
            "• 🖼️ Ảnh — phân tích, trích text, nhận diện\n"
            "• 📄 File code/text — review, debug, giải thích\n"
            "• 💾 Xuất file theo yêu cầu \\(py, js, md, json…\\)\n\n"
            "📋 *Lệnh:* /help để xem hướng dẫn đầy đủ",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "📖 *HƯỚNG DẪN SỬ DỤNG*\n\n"
            "*💬 Chat thường:*\n"
            "Gửi tin nhắn bất kỳ, bot tự chọn agent phù hợp\\.\n\n"
            "*🖼️ Gửi ảnh:*\n"
            "Gửi ảnh \\(kèm caption nếu muốn\\) — bot phân tích nội dung, "
            "đọc code trong ảnh, mô tả biểu đồ\\.\n\n"
            "*📄 Gửi file code/text:*\n"
            "Gửi file \\(.py .js .ts .html .json .md .txt…\\) kèm yêu cầu\\.\n"
            "Ví dụ: gửi `main.py` + caption `\"Review code này\"`\n\n"
            "*💾 Xuất file:*\n"
            "Thêm từ khoá vào yêu cầu:\n"
            "`\"xuất file\"` `\"tạo file\"` `\"save file\"` `\"export\"`\n"
            "Ví dụ: `\"Viết bot Python và xuất file\"`\n\n"
            "*Lệnh khác:*\n"
            "/agents — Danh sách agent\n"
            "/memory — Lịch sử chat\n"
            "/clear — Xóa lịch sử\n"
            "/stats — Thống kê\n\n"
            f"⚠️ Giới hạn: {self._cfg.rate_limit_messages} tin/{self._cfg.rate_limit_window}s "
            f"• File tối đa {MAX_FILE_SIZE_MB}MB",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def cmd_agents(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "🤖 *DANH SÁCH AGENT*\n\n"
            "*1\\. ANALYST* — Phân tích vấn đề phức tạp\n"
            "*2\\. RESEARCHER* — Tổng hợp thông tin chính xác\n"
            "*3\\. CODER* — Viết, review, debug code\n"
            "*4\\. WRITER* — Soạn thảo, dịch thuật nội dung\n"
            "*5\\. CRITIC* — Luôn chạy cuối, kiểm tra đầu ra\n\n"
            "🔀 Hệ thống tự điều phối theo từng yêu cầu\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def cmd_memory(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        history = await self._memory.get(uid, limit=10)
        if not history:
            await update.message.reply_text("📭 Chưa có lịch sử.")
            return
        lines = ["📜 *Lịch sử gần đây:*\n"]
        for i, h in enumerate(history, 1):
            ts = h["timestamp"][11:16]
            preview = h["text"][:60].replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")
            ellipsis = "…" if len(h["text"]) > 60 else ""
            lines.append(f"`{i:02d}` \\[{ts}\\] {preview}{ellipsis}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)

    async def cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        cleared = await self._memory.clear(update.effective_user.id)
        await update.message.reply_text(
            "🗑️ Đã xóa lịch sử." if cleared else "ℹ️ Không có gì để xóa."
        )

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        s = await self._memory.stats(uid)
        await update.message.reply_text(
            f"📊 *Thống kê*\n\n"
            f"💬 Tin đã lưu: `{s['count']}`\n"
            f"🕐 Cũ nhất: `{s['oldest'] or 'N/A'}`\n"
            f"🕐 Mới nhất: `{s['newest'] or 'N/A'}`\n\n"
            f"🤖 AI: {'✅' if self._cfg.nvidia_api_key else '❌'}\n"
            f"💾 GitHub: {'✅' if self._cfg.github_token else '❌ Không cấu hình'}\n"
            f"📦 Model: `{self._cfg.model}`\n"
            f"👁️ Vision: `{self._cfg.vision_model}`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── Message handler (text) ─────────────────────────────────

    async def handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        user_input = update.message.text.strip()
        if not user_input:
            return

        wait = self._check_rate(user.id)
        if wait:
            await update.message.reply_text(f"⏳ Chờ *{wait}* giây.", parse_mode=ParseMode.MARKDOWN)
            return

        await self._memory.add(user.id, user_input)
        await update.message.chat.send_action("typing")
        logger.info(f"[TEXT] User {user.id}: {user_input[:80]}")

        history = await self._memory.get(user.id, limit=self._cfg.history_context_limit - 1)
        history = history[:-1] if history else []

        response = await self._ai.complete(user_input, history)

        # Gửi file nếu user yêu cầu
        sent_file = await self._maybe_send_files(update.message, response, user_input)

        # Luôn gửi text response kèm (trừ khi response rất ngắn và đã có file)
        if not sent_file or len(response) > 200:
            await self._send_long(update.message, response)

    # ── Photo handler ──────────────────────────────────────────

    async def handle_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message

        wait = self._check_rate(user.id)
        if wait:
            await message.reply_text(f"⏳ Chờ *{wait}* giây.", parse_mode=ParseMode.MARKDOWN)
            return

        await message.chat.send_action("typing")
        logger.info(f"[PHOTO] User {user.id}")

        # Caption là yêu cầu của user
        caption = (message.caption or "").strip()
        prompt = caption if caption else "Hãy phân tích ảnh này chi tiết."

        status_msg = await message.reply_text("🔍 Đang phân tích ảnh...")

        try:
            # Lấy ảnh có độ phân giải cao nhất
            best_photo = max(message.photo, key=lambda p: p.file_size or 0)
            img_b64, media_type = await FileProcessor.process_photo(best_photo, ctx.bot)

            response = await self._ai.complete_vision(prompt, img_b64, media_type)

            await status_msg.delete()

            # Lưu vào memory
            await self._memory.add(user.id, f"[Ảnh] {caption or 'Phân tích ảnh'}")

            # Xuất file nếu cần
            sent_file = await self._maybe_send_files(message, response, prompt)
            await self._send_long(message, response)

        except Exception as e:
            logger.error(f"Lỗi xử lý ảnh: {e}")
            await status_msg.edit_text(f"❌ Không thể xử lý ảnh: {e}")

    # ── Document handler ───────────────────────────────────────

    async def handle_document(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message
        doc = message.document

        wait = self._check_rate(user.id)
        if wait:
            await message.reply_text(f"⏳ Chờ *{wait}* giây.", parse_mode=ParseMode.MARKDOWN)
            return

        await message.chat.send_action("typing")
        logger.info(f"[DOC] User {user.id}: {doc.file_name}")

        caption = (message.caption or "").strip()
        status_msg = await message.reply_text(f"📂 Đang đọc `{doc.file_name}`...", parse_mode=ParseMode.MARKDOWN)

        try:
            file_content, filename, ext = await FileProcessor.process_document(doc, ctx.bot)

            prompt = caption if caption else f"Hãy phân tích file `{filename}`."

            # Lấy history
            history = await self._memory.get(user.id, limit=self._cfg.history_context_limit - 1)

            await status_msg.edit_text(f"🤖 Đang xử lý `{filename}`...", parse_mode=ParseMode.MARKDOWN)

            response = await self._ai.complete_with_file(
                prompt, file_content, filename, history
            )

            await status_msg.delete()

            # Lưu vào memory
            await self._memory.add(
                user.id, f"[File: {filename}] {caption or 'Phân tích file'}"
            )

            # Xuất file nếu cần
            await self._maybe_send_files(message, response, prompt)
            await self._send_long(message, response)

        except ValueError as e:
            await status_msg.edit_text(f"⚠️ {e}", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi xử lý document: {e}")
            await status_msg.edit_text(f"❌ Lỗi: {e}")

    # ── Error handler ──────────────────────────────────────────

    async def handle_error(self, update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        error = ctx.error
        if isinstance(error, Conflict):
            logger.critical(
                "❌ CONFLICT: Bot instance khác đang chạy! "
                f"Xóa webhook: https://api.telegram.org/bot{self._cfg.telegram_token}/deleteWebhook"
            )
            return
        if isinstance(error, NetworkError):
            logger.warning(f"Network error (tạm thời): {error}")
            return
        logger.error(f"Unhandled: {error}", exc_info=error)
        if isinstance(update, Update) and update.message:
            try:
                await update.message.reply_text("❌ Lỗi không mong muốn. Vui lòng thử lại.")
            except Exception:
                pass

    # ── Build ──────────────────────────────────────────────────

    def build_app(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self._cfg.telegram_token)
            .concurrent_updates(True)
            .post_init(self._post_init)
            .build()
        )

        # Commands
        app.add_handler(CommandHandler("start",   self.cmd_start))
        app.add_handler(CommandHandler("help",    self.cmd_help))
        app.add_handler(CommandHandler("agents",  self.cmd_agents))
        app.add_handler(CommandHandler("memory",  self.cmd_memory))
        app.add_handler(CommandHandler("clear",   self.cmd_clear))
        app.add_handler(CommandHandler("stats",   self.cmd_stats))

        # Media handlers
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

        # Text handler (sau cùng)
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        app.add_error_handler(self.handle_error)
        return app

    async def _post_init(self, app: Application) -> None:
        logger.info("🔧 Đang khởi tạo bot...")
        await self._memory.load()
        await app.bot.set_my_commands([
            BotCommand("start",   "Màn hình chào mừng"),
            BotCommand("agents",  "Danh sách agent"),
            BotCommand("memory",  "Xem lịch sử chat"),
            BotCommand("clear",   "Xóa lịch sử"),
            BotCommand("stats",   "Thống kê sử dụng"),
            BotCommand("help",    "Hướng dẫn sử dụng"),
        ])
        logger.info(f"✅ Bot sẵn sàng | Model: {self._cfg.model} | Vision: {self._cfg.vision_model}")

    def run(self) -> None:
        """PTB v20+ tự quản lý event loop — KHÔNG dùng asyncio.run()."""
        app = self.build_app()
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
        bot.run()
    except KeyboardInterrupt:
        logger.info("⛔ Bot dừng (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 Bot crash: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
