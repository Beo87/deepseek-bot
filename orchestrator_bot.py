#!/usr/bin/env python3
"""
Orchestrator Bot - Multi-Agent AI System
Roles: ANALYST, RESEARCHER, CODER, WRITER, CRITIC
Workflow: ROUTE → EXECUTE → CRITIC REVIEW → FINAL ANSWER
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─────────────────────────────────────────────────────
# CONFIGURATION & ENVIRONMENT
# ─────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "ai-bot-memory")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "your-github-username")

if not TELEGRAM_TOKEN or not NVIDIA_API_KEY:
    print("❌ Missing required environment variables:")
    print("   export TELEGRAM_TOKEN='your_token_here'")
    print("   export NVIDIA_API_KEY='your_nvidia_key_here'")
    exit(1)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# NVIDIA API Endpoint
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ─────────────────────────────────────────────────────
# MEMORY SYSTEM (GitHub-backed)
# ─────────────────────────────────────────────────────
class MemorySystem:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def _github_request(self, method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        if not GITHUB_TOKEN:
            return None
        
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        if data:
            headers["Content-Type"] = "application/json"
        
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/{path}"
        
        try:
            response = requests.request(method, url, headers=headers, json=data, timeout=10)
            if response.status_code in [200, 201]:
                return response.json()
            elif response.status_code == 404 and method == "GET":
                return None
            else:
                logger.warning(f"GitHub API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"GitHub request failed: {e}")
            return None

    def save(self, key: str, value: Any, user_id: str = "global") -> bool:
        """Save memory to GitHub"""
        if not GITHUB_TOKEN:
            self.cache[f"{user_id}:{key}"] = value
            return True
        
        path = f"contents/memory/{user_id}/{key}.json"
        
        # Check if file exists to get SHA
        existing = self._github_request("GET", path)
        data = {
            "message": f"Update memory: {key}",
            "content": json.dumps(value),
            "branch": "main"
        }
        if existing and "sha" in existing:
            data["sha"] = existing["sha"]
        
        result = self._github_request("PUT", path, data)
        if result:
            self.cache[f"{user_id}:{key}"] = value
            return True
        return False

    def get(self, key: str, user_id: str = "global") -> Optional[Any]:
        """Retrieve memory from cache or GitHub"""
        cache_key = f"{user_id}:{key}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not GITHUB_TOKEN:
            return None
        
        path = f"contents/memory/{user_id}/{key}.json"
        result = self._github_request("GET", path)
        
        if result and "content" in result:
            import base64
            content = base64.b64decode(result["content"]).decode("utf-8")
            data = json.loads(content)
            self.cache[cache_key] = data
            return data
        return None

    def list_keys(self, user_id: str = "global") -> List[str]:
        """List all memory keys for a user"""
        if not GITHUB_TOKEN:
            return [k.replace(f"{user_id}:", "") for k in self.cache.keys() if k.startswith(f"{user_id}:")]
        
        path = f"contents/memory/{user_id}"
        result = self._github_request("GET", path)
        
        if result and isinstance(result, list):
            return [item["name"].replace(".json", "") for item in result if item["type"] == "file"]
        return []

memory = MemorySystem()

# ─────────────────────────────────────────────────────
# LLM INTERFACE (NVIDIA NIM)
# ─────────────────────────────────────────────────────
def call_llm(messages: List[Dict[str, str]], model: str = "meta/llama-3.1-70b-instruct", temperature: float = 0.7) -> Tuple[Optional[str], Optional[str]]:
    """Call NVIDIA NIM API and return (response, error)"""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False,
    }
    
    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            return None, f"API Error: {response.status_code} - {response.text}"
        
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            return content, None
        else:
            return None, "No response from model"
            
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────
# MULTI-AGENT ORCHESTRATOR SYSTEM
# ─────────────────────────────────────────────────────
AGENT_PROMPTS = {
    "ANALYST": """You are ANALYST agent. Your role is to break down complex problems into subtasks.
Output a structured task list with priorities. Be concise and clear.""",
    
    "RESEARCHER": """You are RESEARCHER agent. Your role is to gather, verify, and synthesize information.
Provide summarized findings with sources if available. If you don't know, say "I don't know".""",
    
    "CODER": """You are CODER agent. Your role is to write, review, and debug code.
Output clean, commented, working code. Explain your approach briefly.""",
    
    "WRITER": """You are WRITER agent. Your role is to draft, edit, and translate content.
Output polished, tone-matched text. Keep it engaging and clear.""",
    
    "CRITIC": """You are CRITIC agent. Your role is to review outputs for errors, gaps, or improvements.
Flag issues or approve. Be constructive and thorough."""
}

def route_request(user_input: str) -> List[str]:
    """Determine which agents are needed"""
    messages = [
        {"role": "system", "content": "You are a router. Analyze the user request and return ONLY a comma-separated list of 1-3 agent names from: ANALYST, RESEARCHER, CODER, WRITER. Always include CRITIC at the end implicitly (don't list it). Rules: If ambiguous/multi-step → ANALYST first. If needs facts → RESEARCHER. If coding → CODER. If writing/translating → WRITER."},
        {"role": "user", "content": user_input}
    ]
    
    response, error = call_llm(messages, temperature=0.3)
    
    if error or not response:
        return ["ANALYST", "CRITIC"]  # Default fallback
    
    # Parse response
    agents = [a.strip().upper() for a in response.replace(",", " ").split()]
    valid_agents = [a for a in agents if a in ["ANALYST", "RESEARCHER", "CODER", "WRITER"]]
    
    if not valid_agents:
        return ["ANALYST", "CRITIC"]
    
    # Ensure CRITIC is always last
    if "CRITIC" not in valid_agents:
        valid_agents.append("CRITIC")
    
    return valid_agents[:4]  # Max 3 + CRITIC

def execute_agent(agent_name: str, user_input: str, context: str = "") -> str:
    """Execute a specific agent's task"""
    prompt = AGENT_PROMPTS.get(agent_name, "You are a helpful assistant.")
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User Request: {user_input}\n\nContext from previous agents: {context}" if context else f"User Request: {user_input}"}
    ]
    
    response, error = call_llm(messages, temperature=0.7)
    
    if error:
        return f"[{agent_name}] Error: {error}"
    
    return f"[{agent_name}]\n{response}"

def orchestrator_workflow(user_input: str) -> str:
    """Main orchestrator workflow: ROUTE → EXECUTE → CRITIC → FINAL"""
    
    # Step 1: ROUTE
    agents_needed = route_request(user_input)
    logger.info(f"Routing to agents: {agents_needed}")
    
    # Step 2: EXECUTE (sequential chain)
    results = []
    context = ""
    
    for agent in agents_needed:
        if agent == "CRITIC":
            continue  # Handle CRITIC separately
        
        output = execute_agent(agent, user_input, context)
        results.append(output)
        context += "\n" + output
    
    # Step 3: CRITIC REVIEW
    critic_messages = [
        {"role": "system", "content": AGENT_PROMPTS["CRITIC"]},
        {"role": "user", "content": f"Review the following agent outputs for the user request: '{user_input}'\n\n{''.join(results)}\n\nProvide corrections if needed, or approve. Then create a final merged answer that is concise, actionable, and human-friendly (no agent labels)."}
    ]
    
    critic_response, error = call_llm(critic_messages, temperature=0.5)
    
    if error or not critic_response:
        # Fallback: merge results manually
        final_answer = "Here's what I found:\n\n" + "\n\n".join(results)
    else:
        final_answer = critic_response
    
    return final_answer

# ─────────────────────────────────────────────────────
# TELEGRAM BOT HANDLERS
# ─────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 **Orchestrator Bot Activated**\n\n"
        "I coordinate multiple AI agents to solve your tasks:\n"
        "• ANALYST – Break down problems\n"
        "• RESEARCHER – Gather facts\n"
        "• CODER – Write/debug code\n"
        "• WRITER – Create/edit content\n"
        "• CRITIC – Quality review\n\n"
        "**Commands:**\n"
        "/orchestrator <task> – Run multi-agent workflow\n"
        "/myagents – List available agents\n"
        "/memory – View saved memories\n"
        "/clear – Clear conversation\n\n"
        "Just send me any task and I'll orchestrate the right agents!"
    )

async def myagents_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List available agents"""
    agents_list = "\n".join([f"• {name} – {desc.split('.')[0]}" for name, desc in AGENT_PROMPTS.items()])
    await update.message.reply_text(f"**Available Agents:**\n\n{agents_list}")

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View saved memories"""
    user_id = str(update.effective_user.id)
    keys = memory.list_keys(user_id)
    
    if not keys:
        await update.message.reply_text("No memories saved yet.")
        return
    
    memory_list = "\n".join([f"• `{k}`" for k in keys])
    await update.message.reply_text(f"**Your Memories:**\n\n{memory_list}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation context"""
    context.user_data.clear()
    await update.message.reply_text("✅ Conversation cleared!")

async def orchestrator_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /orchestrator command"""
    if not context.args:
        await update.message.reply_text("Usage: /orchestrator <your task>")
        return
    
    user_input = " ".join(context.args)
    await process_orchestrator_request(update, user_input)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages"""
    user_input = update.message.text
    
    # Auto-detect orchestrator mode for complex requests
    if len(user_input.split()) > 5 or any(keyword in user_input.lower() for keyword in ["code", "write", "analyze", "research", "debug", "create"]):
        await process_orchestrator_request(update, user_input)
    else:
        # Simple chat fallback
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise and friendly."},
            {"role": "user", "content": user_input}
        ]
        response, error = call_llm(messages)
        
        if error:
            await update.message.reply_text(f"Error: {error}")
        else:
            await update.message.reply_text(response)

async def process_orchestrator_request(update: Update, user_input: str):
    """Process request through orchestrator workflow"""
    # Send thinking indicator
    thinking_msg = await update.message.reply_text("🔄 Orchestrating agents...")
    
    try:
        # Run orchestrator workflow
        result = orchestrator_workflow(user_input)
        
        # Delete thinking message
        await thinking_msg.delete()
        
        # Send result
        await update.message.reply_text(result, parse_mode="Markdown")
        
        # Save to memory if useful
        if len(user_input) > 20:
            memory.save(f"task_{int(time.time())}", {"input": user_input, "output": result}, str(update.effective_user.id))
    
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        await thinking_msg.edit_text(f"❌ Error: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error: {context.error}")
    if update and update.message:
        await update.message.reply_text("⚠️ An error occurred. Please try again.")

# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────
def main():
    """Start the bot"""
    print("🚀 Starting Orchestrator Bot...")
    print(f"Agents: {', '.join(AGENT_PROMPTS.keys())}")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("myagents", myagents_command))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("orchestrator", orchestrator_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    # Start polling
    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
