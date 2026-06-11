"""
╔══════════════════════════════════════════════════════════════╗
║           MULTI-AGENT ORCHESTRATOR BOT                      ║
║           Version: 4.0.0                                    ║
║           Stack: python-telegram-bot v20+ | NVIDIA NIM      ║
╚══════════════════════════════════════════════════════════════╝

TÍNH NĂNG v4.0:
  • Chat AI đa agent (Analyst, Researcher, Coder, Writer, Critic)
  • OCR thông minh — đọc chữ viết tay tiếng Việt + code trong ảnh
  • Đọc file code / text và xử lý
  • Web Search — tìm tài liệu, StackOverflow, GitHub, docs online
  • Deep Analysis — kết hợp OCR + web search → phân tích chính xác nhất
  • Xuất file theo yêu cầu (.py, .js, .txt, .md, .html, .json, ...)
  • Lịch sử hội thoại + GitHub backup
  • Rate limiting

ENV VARIABLES:
  TELEGRAM_TOKEN    — Bot token từ @BotFather              (bắt buộc)
  NVIDIA_API_KEY    — API key từ build.nvidia.com           (bắt buộc)
  TAVILY_API_KEY    — Tavily search API (tavily.com)        (tùy chọn, search tốt hơn)
  GITHUB_TOKEN      — GitHub personal access token          (tùy chọn)
  GITHUB_OWNER      — GitHub username                       (tùy chọn)
  GITHUB_REPO       — Repo lưu memory [bot-memory]          (tùy chọn)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
import urllib.parse
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

SUPPORTED_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".html", ".css", ".scss", ".less", ".xml", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".md", ".txt", ".rst", ".csv", ".log", ".dockerfile",
    ".gitignore", ".htaccess",
}

OUTPUT_KEYWORDS = {
    "xuất file", "tạo file", "save file", "export file", "lưu file",
    "ghi ra file", "tạo ra file", "viết ra file", "xuất ra file",
    "save as", "tạo script", "tạo code file", "download file",
    "generate file", "create file", "output file",
}

LANG_TO_EXT: dict[str, str] = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "java": ".java", "cpp": ".cpp", "c++": ".cpp",
    "c": ".c", "csharp": ".cs", "c#": ".cs",
    "go": ".go", "rust": ".rs", "ruby": ".rb",
    "php": ".php", "swift": ".swift", "kotlin": ".kt",
    "html": ".html", "css": ".css", "sql": ".sql",
    "bash": ".sh", "shell": ".sh", "sh": ".sh",
    "json": ".json", "yaml": ".yaml", "yml": ".yaml",
    "markdown": ".md", "md": ".md",
    "text": ".txt", "txt": ".txt",
    "xml": ".xml", "toml": ".toml", "powershell": ".ps1",
}

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Số kết quả web search tối đa
MAX_SEARCH_RESULTS = 5
# Số ký tự tối đa mỗi kết quả search
MAX_SNIPPET_CHARS = 500


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    telegram_token: str
    nvidia_api_key: str
    tavily_api_key: Optional[str] = None
    github_token: Optional[str] = None
    github_owner: Optional[str] = None
    github_repo: str = "bot-memory"

    nvidia_api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    vision_model: str = "meta/llama-3.2-90b-vision-instruct"

    temperature: float = 0.6
    max_tokens: int = 2500
    request_timeout: int = 90

    memory_limit_per_user: int = 50
    history_context_limit: int = 5

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
            logger.warning("⚠️  Thiếu NVIDIA_API_KEY")
        return cls(
            telegram_token=token,
            nvidia_api_key=api_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
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
# WEB SEARCH ENGINE
# ══════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearchEngine:
    """
    Tìm kiếm web để lấy tài liệu, StackOverflow, GitHub, docs.
    Ưu tiên: Tavily API (tốt nhất) → DuckDuckGo (miễn phí, fallback).
    """

    def __init__(self, config: Config):
        self._cfg = config

    async def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[SearchResult]:
        """Tìm kiếm, tự chọn engine tốt nhất có sẵn."""
        if self._cfg.tavily_api_key:
            results = await self._tavily_search(query, max_results)
            if results:
                return results
        # Fallback DuckDuckGo
        return await self._ddg_search(query, max_results)

    async def _tavily_search(self, query: str, max_results: int) -> list[SearchResult]:
        """Tavily API — kết quả chất lượng cao, hỗ trợ technical queries."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._cfg.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_domains": [
                            "stackoverflow.com", "github.com", "docs.python.org",
                            "developer.mozilla.org", "docs.microsoft.com",
                            "medium.com", "dev.to", "realpython.com",
                        ],
                    },
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for r in data.get("results", [])[:max_results]:
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("content", "")[:MAX_SNIPPET_CHARS],
                            source="tavily",
                        ))
                    logger.info(f"🔍 Tavily: {len(results)} kết quả cho '{query[:40]}'")
                    return results
        except Exception as e:
            logger.warning(f"Tavily search lỗi: {e}")
        return []

    async def _ddg_search(self, query: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo HTML search — miễn phí, không cần API key."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    return []

                html = resp.text
                results = []

                # Parse kết quả từ HTML
                # Tìm các block kết quả
                blocks = re.findall(
                    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )

                for href, title_raw, snippet_raw in blocks[:max_results]:
                    # Lọc redirect URL của DDG
                    real_url = href
                    uddg_match = re.search(r'uddg=([^&]+)', href)
                    if uddg_match:
                        real_url = urllib.parse.unquote(uddg_match.group(1))

                    title = re.sub(r'<[^>]+>', '', title_raw).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet_raw).strip()[:MAX_SNIPPET_CHARS]

                    if title and real_url.startswith("http"):
                        results.append(SearchResult(
                            title=title,
                            url=real_url,
                            snippet=snippet,
                            source="duckduckgo",
                        ))

                logger.info(f"🔍 DDG: {len(results)} kết quả cho '{query[:40]}'")
                return results

        except Exception as e:
            logger.warning(f"DDG search lỗi: {e}")
        return []

    @staticmethod
    def format_for_ai(results: list[SearchResult]) -> str:
        """Format kết quả search thành context cho AI."""
        if not results:
            return "(Không tìm được kết quả web)"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.title}\n"
                f"    URL: {r.url}\n"
                f"    {r.snippet}"
            )
        return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════

class FileProcessor:
    @staticmethod
    async def download_bytes(file_id: str, bot) -> bytes:
        tg_file = await bot.get_file(file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        return buf.getvalue()

    @staticmethod
    def to_base64(data: bytes) -> str:
        return base64.standard_b64encode(data).decode()

    @classmethod
    async def process_photo(cls, photo: PhotoSize, bot) -> tuple[str, str]:
        data = await cls.download_bytes(photo.file_id, bot)
        return cls.to_base64(data), "image/jpeg"

    @classmethod
    async def process_document(cls, doc: Document, bot) -> tuple[str, str, str]:
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File quá lớn ({doc.file_size // 1024 // 1024}MB). Giới hạn {MAX_FILE_SIZE_MB}MB.")

        filename = doc.file_name or "unknown"
        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_CODE_EXTS and not (
            doc.mime_type and doc.mime_type.startswith("text/")
        ):
            raise ValueError(
                f"Định dạng `{ext or 'không rõ'}` không được hỗ trợ.\n"
                f"Hỗ trợ: {', '.join(sorted(SUPPORTED_CODE_EXTS)[:20])}..."
            )

        data = await cls.download_bytes(doc.file_id, bot)
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
    @staticmethod
    def wants_file(user_input: str) -> bool:
        lower = user_input.lower()
        return any(kw in lower for kw in OUTPUT_KEYWORDS)

    @staticmethod
    def extract_code_blocks(text: str) -> list[tuple[str, str]]:
        pattern = r"```(\w*)\n?([\s\S]*?)```"
        matches = re.findall(pattern, text)
        return [(lang.lower().strip(), code.strip()) for lang, code in matches if code.strip()]

    @classmethod
    def build_output_files(cls, ai_response: str, user_input: str) -> list[tuple[bytes, str]]:
        blocks = cls.extract_code_blocks(ai_response)
        if not blocks:
            return []
        results = []
        timestamp = datetime.now().strftime("%H%M%S")
        for i, (lang, code) in enumerate(blocks):
            ext = LANG_TO_EXT.get(lang.lower(), ".txt")
            name = f"output_{timestamp}_{i+1}{ext}" if len(blocks) > 1 else f"output_{timestamp}{ext}"
            results.append((code.encode("utf-8"), name))
        return results


# ══════════════════════════════════════════════════════════════
# AI PROMPTS
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are the ORCHESTRATOR of a multi-agent AI system.
Respond in the SAME LANGUAGE the user writes in (Vietnamese if Vietnamese).

━━━ AGENTS ━━━
[ANALYST]    → Break down complex problems into subtasks.
[RESEARCHER] → Verify facts. Use provided web search results as sources.
[CODER]      → Write, review, debug code. Always use ```lang fences.
[WRITER]     → Draft, edit, translate content.
[CRITIC]     → Review all outputs last. Never skipped.

━━━ FILE OUTPUT RULE ━━━
When user asks to create/export/save a file, wrap complete content in ```lang fences.

━━━ WEB SOURCE RULE ━━━
When web search results are provided, USE THEM as authoritative references.
Cite sources naturally: "Theo [title]..." or "Dựa trên tài liệu tại [url]...".
Cross-reference multiple sources for accuracy.

━━━ WORKFLOW ━━━
1. ROUTE → pick 1-3 agents
2. EXECUTE → complete subtasks
3. CRITIC → validate
4. REPLY → clean, concise, in user's language

━━━ RULES ━━━
• Step-by-step thinking. CRITIC never skipped.
• Never hallucinate. Use web sources when available.
• Use markdown + code blocks with language tags.
"""

# Prompt OCR chuyên biệt cho chữ viết tay tiếng Việt và code
OCR_SYSTEM_PROMPT = """\
You are an expert OCR and handwriting recognition system specialized in:
1. Vietnamese handwritten text (including diacritical marks: ắ ặ ổ ộ ừ ứ etc.)
2. Handwritten or photographed source code (any programming language)
3. Mixed content (Vietnamese notes + code snippets)

━━━ OCR INSTRUCTIONS ━━━
Step 1 — TRANSCRIBE
  • Read ALL visible text/code exactly as written.
  • For Vietnamese: preserve ALL tone marks (sắc, huyền, hỏi, ngã, nặng) and
    vowel modifications (ă â ê ô ơ ư đ).
  • For code: preserve indentation, operators, brackets, semicolons exactly.
  • Mark unclear characters as [?] rather than guessing wrongly.
  • If image has multiple sections, transcribe each separately labeled.

Step 2 — CLEAN
  • Fix obvious OCR errors (O vs 0, l vs 1, etc.) based on context.
  • Reconstruct broken words from handwriting.
  • Output clean, readable version.

Step 3 — DETECT CONTENT TYPE
  • Identify: Vietnamese prose | Code | Math | Mixed | Other
  • Identify programming language if code is detected.
  • Note any ambiguous or unclear sections.

OUTPUT FORMAT:
```
=== VĂN BẢN GỐC (PHIÊN ÂM) ===
[transcribed content here]

=== LOẠI NỘI DUNG ===
[content type + language if code]

=== GHI CHÚ ===
[unclear parts, corrections made, confidence notes]
```

Respond in Vietnamese.
"""

# Prompt phân tích sâu kết hợp OCR + web search
DEEP_ANALYSIS_PROMPT = """\
You are an expert code and content analyst. You have been given:
1. OCR-extracted text/code from an image
2. Relevant web search results as reference material

━━━ YOUR TASK ━━━
Perform a DEEP, ACCURATE analysis by combining both sources:

For CODE content:
  • Identify language, framework, libraries used
  • Explain what the code does (line by line if needed)
  • Find bugs, errors, security issues, anti-patterns
  • Compare with best practices found in web sources
  • Suggest concrete improvements with fixed code
  • Reference official docs/StackOverflow answers when relevant

For Vietnamese TEXT content:
  • Summarize the content
  • Fact-check claims against web sources
  • Expand with related information from search results
  • Provide deeper context

For MIXED content:
  • Analyze text and code parts separately, then holistically

━━━ OUTPUT STRUCTURE ━━━
Use clear sections with headers.
Cite web sources: "Theo [source title]" or "[url]".
Always wrap code in ```language fences.
Respond in the SAME LANGUAGE as the extracted text (Vietnamese if Vietnamese).
"""


# ══════════════════════════════════════════════════════════════
# AI ENGINE
# ══════════════════════════════════════════════════════════════

class AIEngine:
    def __init__(self, config: Config):
        self._cfg = config
        self._headers = {
            "Authorization": f"Bearer {config.nvidia_api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, user_input: str, history: list[dict], *, retries: int = 2) -> str:
        context_lines = [f"• {h['text'][:80]}" for h in history]
        context_block = "\n".join(context_lines) if context_lines else "(Không có)"
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[Lịch sử]\n{context_block}\n\n[Yêu cầu]\n{user_input}"},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        return await self._call(payload, retries=retries)

    async def ocr_image(self, image_b64: str, media_type: str = "image/jpeg", *, retries: int = 2) -> str:
        """Bước 1: OCR thuần — chỉ phiên âm văn bản/code từ ảnh."""
        payload = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Hãy đọc và phiên âm TOÀN BỘ nội dung trong ảnh này. "
                                "Chú ý đặc biệt đến chữ viết tay tiếng Việt và code. "
                                "Giữ nguyên tất cả dấu tiếng Việt và cú pháp code."
                            ),
                        },
                    ],
                },
            ],
            "temperature": 0.1,   # Thấp để OCR chính xác hơn
            "max_tokens": 2000,
        }
        return await self._call(payload, retries=retries)

    async def generate_search_query(
        self,
        ocr_text: str,
        user_prompt: str,
        *,
        retries: int = 1,
    ) -> str:
        """
        Dùng AI sinh ra search query chính xác từ nội dung OCR.
        Tránh dùng regex vì dễ bắt nhầm tên import/framework.
        """
        if not self._cfg.nvidia_api_key:
            return user_prompt or "code analysis"

        payload = {
            "model": self._cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a search query generator. "
                        "Given OCR-extracted content, output ONLY a short, focused search query "
                        "(max 8 words) suitable for finding relevant documentation, "
                        "StackOverflow answers, or GitHub examples. "
                        "Focus on: the core problem/concept, NOT import names or framework names. "
                        "If content is Vietnamese text, generate query in Vietnamese. "
                        "Output the query only — no explanation, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"OCR content (first 500 chars):\n{ocr_text[:500]}\n\n"
                        f"User request: {user_prompt or '(none)'}\n\n"
                        "Generate ONE focused search query:"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 30,
        }
        result = await self._call(payload, retries=retries)
        # Làm sạch output — bỏ dấu nháy, xuống dòng
        query = result.strip().strip("\"'").split("\n")[0].strip()
        # Fallback nếu AI trả về rác
        if len(query) < 3 or len(query) > 120:
            return user_prompt or "code analysis best practices"
        return query

    async def analyze_with_vision(
        self,
        image_b64: str,
        user_prompt: str,
        web_context: str,
        media_type: str = "image/jpeg",
        *,
        retries: int = 2,
    ) -> str:
        """Bước 2: Phân tích ảnh kết hợp web search context."""
        combined_prompt = (
            f"[YÊU CẦU CỦA NGƯỜI DÙNG]\n{user_prompt}\n\n"
            f"[KẾT QUẢ TÌM KIẾM WEB]\n{web_context}\n\n"
            "Dựa vào ảnh và tài liệu trên mạng ở trên, hãy phân tích chi tiết và chính xác nhất."
        )
        payload = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": DEEP_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        },
                        {"type": "text", "text": combined_prompt},
                    ],
                },
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        return await self._call(payload, retries=retries)

    async def deep_analyze(
        self,
        ocr_text: str,
        user_prompt: str,
        web_context: str,
        *,
        retries: int = 2,
    ) -> str:
        """Bước 3: Phân tích sâu kết hợp OCR result + web search."""
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY."

        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": DEEP_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[VĂN BẢN/CODE ĐÃ ĐỌC TỪ ẢNH]\n{ocr_text}\n\n"
                        f"[KẾT QUẢ TÌM KIẾM WEB]\n{web_context}\n\n"
                        f"[YÊU CẦU PHÂN TÍCH]\n{user_prompt or 'Hãy phân tích toàn diện.'}"
                    ),
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
        web_context: str = "",
        *,
        retries: int = 2,
    ) -> str:
        context_lines = [f"• {h['text'][:80]}" for h in history]
        context_block = "\n".join(context_lines) if context_lines else "(Không có)"

        max_file_chars = 8000
        truncated = ""
        if len(file_content) > max_file_chars:
            truncated = f"\n\n⚠️ [File bị cắt bớt — hiển thị {max_file_chars} ký tự đầu]"
        file_preview = file_content[:max_file_chars]

        web_section = f"\n\n[TÀI LIỆU TỪ WEB]\n{web_context}" if web_context else ""

        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[Lịch sử]\n{context_block}\n\n"
                        f"[File: {filename}]\n```\n{file_preview}\n```{truncated}"
                        f"{web_section}\n\n"
                        f"[Yêu cầu]\n{user_input or 'Hãy phân tích file này.'}"
                    ),
                },
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        return await self._call(payload, retries=retries)

    async def _call(self, payload: dict, *, retries: int) -> str:
        if not self._cfg.nvidia_api_key:
            return "⚠️ Chưa cấu hình NVIDIA_API_KEY."

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
# SMART ANALYZER — Pipeline OCR + Search + Deep Analysis
# ══════════════════════════════════════════════════════════════

class SmartAnalyzer:
    """
    Pipeline phân tích thông minh:
    Ảnh → OCR → Trích từ khoá → Web Search → Deep Analysis
    """

    def __init__(self, ai: AIEngine, search: WebSearchEngine):
        self._ai = ai
        self._search = search

    async def _generate_search_query(self, ocr_text: str, user_prompt: str) -> str:
        """
        Dùng AI sinh query chính xác thay vì regex.
        Tránh bắt nhầm tên import/framework làm query.
        """
        return await self._ai.generate_search_query(ocr_text, user_prompt)

    async def analyze_image(
        self,
        image_b64: str,
        media_type: str,
        user_prompt: str,
        status_callback=None,
    ) -> tuple[str, str, list[SearchResult]]:
        """
        Full pipeline cho ảnh.
        Trả về (ocr_text, analysis, search_results).
        """
        # Bước 1: OCR
        if status_callback:
            await status_callback("🔍 Bước 1/3 — Đang đọc chữ trong ảnh...")
        ocr_text = await self._ai.ocr_image(image_b64, media_type)
        logger.info(f"OCR done: {len(ocr_text)} chars")

        # Bước 2: AI sinh query thông minh (không dùng regex)
        search_query = await self._generate_search_query(ocr_text, user_prompt)
        if status_callback:
            await status_callback(f"🌐 Bước 2/3 — Tìm kiếm: `{search_query[:60]}`...")

        search_results = await self._search.search(search_query)
        web_context = WebSearchEngine.format_for_ai(search_results)

        # Bước 3: Phân tích sâu kết hợp
        if status_callback:
            await status_callback("🧠 Bước 3/3 — Đang phân tích chuyên sâu...")

        # Dùng vision model + web context để phân tích
        analysis = await self._ai.analyze_with_vision(
            image_b64, user_prompt or "Phân tích toàn diện", web_context, media_type
        )

        return ocr_text, analysis, search_results

    async def analyze_file(
        self,
        file_content: str,
        filename: str,
        user_prompt: str,
        history: list[dict],
        status_callback=None,
    ) -> tuple[str, list[SearchResult]]:
        """
        Pipeline cho file code/text.
        Trả về (analysis, search_results).
        """
        # AI sinh query thông minh từ nội dung file
        search_query = await self._generate_search_query(file_content, user_prompt)
        if search_query:
            if status_callback:
                await status_callback(f"🌐 Đang tìm tài liệu: `{search_query[:60]}`...")
            search_results = await self._search.search(search_query)
        else:
            search_results = []

        web_context = WebSearchEngine.format_for_ai(search_results)

        analysis = await self._ai.complete_with_file(
            user_prompt, file_content, filename, history, web_context
        )
        return analysis, search_results


# ══════════════════════════════════════════════════════════════
# BOT
# ══════════════════════════════════════════════════════════════

class OrchestratorBot:
    def __init__(self, config: Config):
        self._cfg = config
        self._memory = MemorySystem(config)
        self._ai = AIEngine(config)
        self._search = WebSearchEngine(config)
        self._analyzer = SmartAnalyzer(self._ai, self._search)
        self._rate = RateLimiter(config.rate_limit_messages, config.rate_limit_window)

    # ── Helpers ────────────────────────────────────────────────

    def _check_rate(self, user_id: int) -> int:
        if self._rate.is_allowed(user_id):
            return 0
        return self._rate.seconds_until_reset(user_id)

    async def _send_long(self, message: Message, text: str) -> None:
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

    async def _send_search_sources(self, message: Message, results: list[SearchResult]) -> None:
        """Gửi danh sách nguồn tham khảo đã dùng."""
        if not results:
            return
        lines = ["🔗 *Nguồn tham khảo:*"]
        for i, r in enumerate(results[:5], 1):
            title = r.title[:50] + ("…" if len(r.title) > 50 else "")
            lines.append(f"`{i}.` [{title}]({r.url})")
        try:
            await message.reply_text(
                "\n".join(lines),
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except TelegramError:
            pass

    async def _maybe_send_files(self, message: Message, ai_response: str, user_input: str) -> bool:
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
            f"👋 Xin chào *{name}*\\! Tôi là **Multi\\-Agent Orchestrator Bot** v4\\.0\\.\n\n"
            "🧠 *5 Agent AI chuyên biệt*\n\n"
            "📁 *Tôi có thể xử lý:*\n"
            "• 🖼️ Ảnh chứa code / chữ viết tay tiếng Việt\n"
            "• 📄 File code \\(.py .js .ts .html .json…\\)\n"
            "• 💬 Chat thường\n"
            "• 💾 Xuất file theo yêu cầu\n\n"
            "🌐 *Tự động tìm kiếm web* để phân tích chính xác nhất\\!\n\n"
            "/help để xem hướng dẫn đầy đủ",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        search_status = "✅ Tavily" if self._cfg.tavily_api_key else "✅ DuckDuckGo (miễn phí)"
        await update.message.reply_text(
            "📖 *HƯỚNG DẪN SỬ DỤNG v4.0*\n\n"
            "*🖼️ Gửi ảnh \\(có chữ viết tay / code\\):*\n"
            "Bot sẽ tự động:\n"
            "1\\. Đọc và phiên âm toàn bộ chữ trong ảnh\n"
            "2\\. Tìm kiếm tài liệu liên quan trên mạng\n"
            "3\\. Phân tích sâu kết hợp cả hai nguồn\n"
            "4\\. Gửi danh sách nguồn tham khảo\n\n"
            "Caption = yêu cầu cụ thể \\(tùy chọn\\)\n"
            "Ví dụ: `\"Code này có bug gì?\"`\n\n"
            "*📄 Gửi file code/text:*\n"
            "Tương tự, kèm web search để phân tích tốt hơn\\.\n\n"
            "*💾 Xuất file:*\n"
            "Dùng từ khoá: `\"xuất file\"` `\"tạo file\"` `\"save\"`\n\n"
            f"*🔍 Search engine:* {search_status}\n\n"
            "/agents /memory /clear /stats",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def cmd_agents(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "🤖 *DANH SÁCH AGENT*\n\n"
            "*1\\. ANALYST* — Phân tích vấn đề phức tạp\n"
            "*2\\. RESEARCHER* — Tổng hợp từ web \\+ kiến thức\n"
            "*3\\. CODER* — Viết, review, debug code\n"
            "*4\\. WRITER* — Soạn thảo, dịch thuật\n"
            "*5\\. CRITIC* — Kiểm tra đầu ra, chạy cuối cùng\n\n"
            "🌐 *Web Search Pipeline:*\n"
            "`Ảnh/File → OCR → Search → Deep Analysis`",
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
        await update.message.reply_text("🗑️ Đã xóa lịch sử." if cleared else "ℹ️ Không có gì để xóa.")

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        s = await self._memory.stats(uid)
        search_engine = "Tavily ✅" if self._cfg.tavily_api_key else "DuckDuckGo (miễn phí)"
        await update.message.reply_text(
            f"📊 *Thống kê*\n\n"
            f"💬 Tin đã lưu: `{s['count']}`\n"
            f"🕐 Mới nhất: `{s['newest'] or 'N/A'}`\n\n"
            f"🤖 AI: {'✅' if self._cfg.nvidia_api_key else '❌'}\n"
            f"👁️ Vision: `{self._cfg.vision_model}`\n"
            f"🌐 Search: `{search_engine}`\n"
            f"💾 GitHub: {'✅' if self._cfg.github_token else '❌'}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # ── Text handler ───────────────────────────────────────────

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

        # Tự động search nếu câu hỏi có vẻ technical
        web_context = ""
        search_results = []
        tech_keywords = (
            "lỗi", "error", "bug", "fix", "code", "function", "class", "import",
            "install", "setup", "deploy", "api", "database", "sql", "python",
            "javascript", "docker", "git", "linux", "windows",
        )
        if any(kw in user_input.lower() for kw in tech_keywords):
            search_results = await self._search.search(user_input[:150])
            web_context = WebSearchEngine.format_for_ai(search_results)

        history = await self._memory.get(user.id, limit=self._cfg.history_context_limit - 1)
        history = history[:-1] if history else []

        if web_context:
            full_input = f"{user_input}\n\n[Tài liệu từ web]\n{web_context}"
            response = await self._ai.complete(full_input, history)
        else:
            response = await self._ai.complete(user_input, history)

        sent_file = await self._maybe_send_files(update.message, response, user_input)
        if not sent_file or len(response) > 200:
            await self._send_long(update.message, response)

        if search_results:
            await self._send_search_sources(update.message, search_results)

    # ── Photo handler ──────────────────────────────────────────

    async def handle_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message

        wait = self._check_rate(user.id)
        if wait:
            await message.reply_text(f"⏳ Chờ *{wait}* giây.", parse_mode=ParseMode.MARKDOWN)
            return

        caption = (message.caption or "").strip()
        status_msg = await message.reply_text("⏳ Đang xử lý...")

        async def update_status(text: str):
            try:
                await status_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass

        try:
            best_photo = max(message.photo, key=lambda p: p.file_size or 0)
            img_b64, media_type = await FileProcessor.process_photo(best_photo, ctx.bot)

            ocr_text, analysis, search_results = await self._analyzer.analyze_image(
                img_b64, media_type, caption,
                status_callback=update_status,
            )

            await status_msg.delete()

            # Gửi OCR trước (text đã đọc được)
            if ocr_text:
                await self._send_long(message, f"📝 *Nội dung đọc được:*\n\n{ocr_text}")

            # Gửi phân tích sâu
            await self._send_long(message, f"🧠 *Phân tích:*\n\n{analysis}")

            # Xuất file nếu cần
            await self._maybe_send_files(message, analysis, caption)

            # Gửi nguồn tham khảo
            await self._send_search_sources(message, search_results)

            # Lưu memory
            await self._memory.add(user.id, f"[Ảnh] {caption or 'OCR + phân tích'}")

        except Exception as e:
            logger.error(f"Lỗi xử lý ảnh: {e}", exc_info=True)
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

        caption = (message.caption or "").strip()
        status_msg = await message.reply_text(
            f"📂 Đang đọc `{doc.file_name}`...", parse_mode=ParseMode.MARKDOWN
        )

        async def update_status(text: str):
            try:
                await status_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass

        try:
            file_content, filename, ext = await FileProcessor.process_document(doc, ctx.bot)

            history = await self._memory.get(user.id, limit=self._cfg.history_context_limit - 1)

            analysis, search_results = await self._analyzer.analyze_file(
                file_content, filename, caption, history,
                status_callback=update_status,
            )

            await status_msg.delete()

            await self._send_long(message, analysis)
            await self._maybe_send_files(message, analysis, caption)
            await self._send_search_sources(message, search_results)

            await self._memory.add(user.id, f"[File: {filename}] {caption or 'Phân tích'}")

        except ValueError as e:
            await status_msg.edit_text(f"⚠️ {e}", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi xử lý document: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ Lỗi: {e}")

    # ── Error handler ──────────────────────────────────────────

    async def handle_error(self, update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        error = ctx.error
        if isinstance(error, Conflict):
            logger.critical("❌ CONFLICT: Bot instance khác đang chạy!")
            return
        if isinstance(error, NetworkError):
            logger.warning(f"Network error: {error}")
            return
        logger.error(f"Unhandled: {error}", exc_info=error)
        if isinstance(update, Update) and update.message:
            try:
                await update.message.reply_text("❌ Lỗi không mong muốn. Vui lòng thử lại.")
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
        app.add_handler(CommandHandler("start",   self.cmd_start))
        app.add_handler(CommandHandler("help",    self.cmd_help))
        app.add_handler(CommandHandler("agents",  self.cmd_agents))
        app.add_handler(CommandHandler("memory",  self.cmd_memory))
        app.add_handler(CommandHandler("clear",   self.cmd_clear))
        app.add_handler(CommandHandler("stats",   self.cmd_stats))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        app.add_error_handler(self.handle_error)
        return app

    async def _post_init(self, app: Application) -> None:
        logger.info("🔧 Đang khởi tạo bot v4.0...")
        await self._memory.load()
        search_info = "Tavily" if self._cfg.tavily_api_key else "DuckDuckGo (miễn phí)"
        await app.bot.set_my_commands([
            BotCommand("start",   "Màn hình chào mừng"),
            BotCommand("agents",  "Danh sách agent"),
            BotCommand("memory",  "Xem lịch sử chat"),
            BotCommand("clear",   "Xóa lịch sử"),
            BotCommand("stats",   "Thống kê sử dụng"),
            BotCommand("help",    "Hướng dẫn sử dụng"),
        ])
        logger.info(
            f"✅ Bot sẵn sàng | Model: {self._cfg.model} | "
            f"Vision: {self._cfg.vision_model} | Search: {search_info}"
        )

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
