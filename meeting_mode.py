import asyncio

# ===== ROLE CONFIG (MỖI ROLE = MODEL KHÁC NHAU) =====
ROLE_CONFIG = {
    "economic": {
        "system": "Bạn là chuyên gia kinh tế.",
        "model": "meta/llama-4-maverick-17b-128e-instruct"
    },
    "technical": {
        "system": "Bạn là kỹ sư phần mềm.",
        "model": "qwen/qwen3-coder-480b-a35b-instruct"
    },
    "legal": {
        "system": "Bạn là chuyên gia pháp lý.",
        "model": "z-ai/glm-5.1"
    },
    "marketing": {
        "system": "Bạn là chuyên gia marketing.",
        "model": "qwen/qwen2.5-coder-32b-instruct"
    }
}

# ===== DETECT =====
def is_meeting_prompt(text: str):
    return "họp" in text.lower() or "meeting" in text.lower()

# ===== AUTO SEARCH HOOK =====
def maybe_search(topic):
    keywords = ["giá", "trend", "2025", "thị trường", "news"]
    if any(k in topic.lower() for k in keywords):
        return f"[SEARCH NEEDED] {topic}"
    return None

# ===== ROUND 1 =====
async def run_roles(topic, goi_nvidia):
    loop = asyncio.get_event_loop()
    tasks = []

    for role, cfg in ROLE_CONFIG.items():
        messages = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": f"""
Phân tích: {topic}

Trả lời JSON:
- summary
- risks
- solution
- score (0-10)
"""}
        ]

        tasks.append(
            loop.run_in_executor(None, goi_nvidia, cfg["model"], messages)
        )

    results = await asyncio.gather(*tasks)

    return {k: v[0] for k, v in zip(ROLE_CONFIG.keys(), results)}

# ===== DEBATE =====
async def debate(topic, results, goi_nvidia):
    loop = asyncio.get_event_loop()
    combined = "\n\n".join([f"{k}: {v}" for k, v in results.items()])

    tasks = []

    for role, cfg in ROLE_CONFIG.items():
        messages = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": f"""
Các ý kiến:
{combined}

Hãy phản biện + cải thiện solution.
"""}
        ]

        tasks.append(
            loop.run_in_executor(None, goi_nvidia, cfg["model"], messages)
        )

    res = await asyncio.gather(*tasks)
    return {k: v[0] for k, v in zip(ROLE_CONFIG.keys(), res)}

# ===== VOTING =====
async def voting(topic, results, goi_nvidia):
    loop = asyncio.get_event_loop()
    combined = "\n\n".join([f"{k}: {v}" for k, v in results.items()])

    tasks = []

    for role, cfg in ROLE_CONFIG.items():
        messages = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": f"""
Chọn giải pháp tốt nhất từ các ý kiến sau:
{combined}

Chỉ trả lời: economic / technical / legal / marketing
"""}
        ]

        tasks.append(
            loop.run_in_executor(None, goi_nvidia, cfg["model"], messages)
        )

    votes = await asyncio.gather(*tasks)

    vote_count = {}
    for v in votes:
        v = v[0].strip().lower()
        vote_count[v] = vote_count.get(v, 0) + 1

    return vote_count

# ===== FINAL AGGREGATE =====
def aggregate(topic, r1, r2, votes, goi_nvidia):
    messages = [
        {"role": "system", "content": "Bạn là CEO tổng hợp chiến lược."},
        {"role": "user", "content": f"""
Topic: {topic}

Round1:
{r1}

Round2:
{r2}

Votes:
{votes}

Hãy đưa ra:
- Chiến lược cuối
- Hành động cụ thể
- Rủi ro cần tránh
"""}
    ]

    res, _ = goi_nvidia("meta/llama-3.1-70b-instruct", messages)
    return res
