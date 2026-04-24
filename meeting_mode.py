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

def is_stop_meeting_prompt(text: str):
    lower_text = text.lower().strip()
    return lower_text == "stop" or lower_text == "ngưng"

# ===== AUTO SEARCH HOOK =====
def maybe_search(topic):
    keywords = ["giá", "trend", "2025", "thị trường", "news"]
    if any(k in topic.lower() for k in keywords):
        return f"[SEARCH NEEDED] {topic}"
    return None

async def run_roles(topic, nvidia_caller):
    tasks = []
    for role, config in ROLE_CONFIG.items():
        messages = [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": f"Chủ đề: {topic}. Đưa ra ý kiến."}
        ]
        tasks.append(nvidia_caller(config["model"], messages))
    
    results = await asyncio.gather(*tasks)
    
    role_results = {}
    for i, role in enumerate(ROLE_CONFIG.keys()):
        role_results[role] = results[i][0] if results[i][1] is None else f"Error: {results[i][0]}"
        
    return role_results

async def debate(topic, r1_results, nvidia_caller):
    debate_messages = {
        "economic_vs_technical": [
            {"role": "system", "content": "Bạn là người điều phối. Tóm tắt và so sánh ý kiến kinh tế và kỹ thuật."},
            {"role": "user", "content": f"Topic: {topic}\n\nKinh tế: {r1_results['economic']}\n\nKỹ thuật: {r1_results['technical']}\n\nHãy tranh luận."}
        ],
        "legal_vs_marketing": [
            {"role": "system", "content": "Bạn là người điều phối. Tóm tắt và so sánh ý kiến pháp lý và marketing."},
            {"role": "user", "content": f"Topic: {topic}\n\nPháp lý: {r1_results['legal']}\n\nMarketing: {r1_results['marketing']}\n\nHãy tranh luận."}
        ]
    }
    
    tasks = [
        nvidia_caller("meta/llama-3.1-8b-instruct", debate_messages["economic_vs_technical"]),
        nvidia_caller("meta/llama-3.1-8b-instruct", debate_messages["legal_vs_marketing"])
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "eco_v_tech": results[0][0] if results[0][1] is None else f"Error: {results[0][0]}",
        "leg_v_mkt": results[1][0] if results[1][1] is None else f"Error: {results[1][0]}"
    }

async def voting(topic, r1_results, nvidia_caller):
    prompt = f"Chủ đề: {topic}\n\nDựa trên các ý kiến sau, hãy bỏ phiếu cho phương án tốt nhất (kinh tế, kỹ thuật, pháp lý, marketing):\n\n"
    for role, opinion in r1_results.items():
        prompt += f"- {role.capitalize()}: {opinion[:200]}...\n"
    
    messages = [
        {"role": "system", "content": "Bạn là người bỏ phiếu. Chỉ trả về một từ: economic, technical, legal, hoặc marketing."},
        {"role": "user", "content": prompt}
    ]
    
    result, err = await nvidia_caller("meta/llama-3.1-8b-instruct", messages)
    
    if err:
        return f"Error: {result}"
        
    # Basic validation
    valid_votes = ["economic", "technical", "legal", "marketing"]
    for vote in valid_votes:
        if vote in result.lower():
            return vote
            
    return "Không quyết định được"


async def aggregate(topic, r1, r2, votes, nvidia_caller):
    prompt = f'''Chủ đề: {topic}

Tổng hợp tất cả thông tin sau thành một báo cáo cuối cùng.

**Vòng 1 - Ý kiến ban đầu:**
- Kinh tế: {r1['economic']}
- Kỹ thuật: {r1['technical']}
- Pháp lý: {r1['legal']}
- Marketing: {r1['marketing']}

**Vòng 2 - Tranh luận:**
- Kinh tế vs Kỹ thuật: {r2['eco_v_tech']}
- Pháp lý vs Marketing: {r2['leg_v_mkt']}

**Bỏ phiếu:** Phương án được chọn là: {votes}

**Yêu cầu:** Tạo báo cáo tổng hợp, giải thích tại sao phương án được chọn là tốt nhất và đề xuất các bước tiếp theo.
'''
    
    messages = [
        {"role": "system", "content": "Bạn là thư ký cuộc họp, chuyên viết báo cáo tổng kết."},
        {"role": "user", "content": prompt}
    ]
    
    result, err = await nvidia_caller("meta/llama-3.1-70b-instruct", messages, timeout=120)
    
    if err:
        return f"Error: {result}"
    return result
