#!/usr/bin/env python3
"""
scribe.py - Production Scribe Worker for Volition 8.0.0-rc1

This is the ephemeral "Tasker" process spawned by GUPPI.
It performs heavy lifting (LLM calls) and reports back via Redis.

Status: RELEASE 6.5.1
- Architecture: Split-Brain Compatible
- Provider: Multi-Backend (Google Native or OpenRouter)
- Feature: Thinking Model Support (Suffix Parsing)
- DEPRECATION: 'vectorize' mode is disabled. Use GUPPI GPU Offload.

Usage:
  python3 scribe.py --model google/gemini-3-flash-preview:thinking --prompt-file /tmp/p.txt --output-inbox inbox:abe-01 --meta '{"source": "log-1"}'

Dependencies:
  pip install redis google-genai aiohttp
"""

import argparse
import asyncio
import json
import os
import sys
import logging
import aiohttp
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as redis


# --- Configuration ---
DEFAULT_REDIS_URL = os.environ.get("REDIS_URL")

CONTEXT_LIMITS = {
    "google/gemini-3-flash-preview": 1_048_576,
    "google/gemini-2.5-flash-preview": 1_048_576,
    "google/gemini-2.5-pro-preview": 1_048_576,
    "mistral": 32_768,
    "qwen-2.5-14b-coder": 32_768,
    "nanbeige-4.1-3b": 8_192,
}
DEFAULT_CONTEXT_LIMIT = 32_768
VECTOR_DB_PATH = Path(os.environ.get("MEMORY_DIR", "./memory")) / "vector.db"

# v6.4 Provider Config
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter") # "google" or "openrouter"

# Google
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://volition.indoria.org")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Volition")


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SCRIBE] - %(levelname)s - %(message)s")
logger = logging.getLogger("scribe")

async def push_result(redis_url: str, inbox: str, event_type: str, content: Any, meta: Dict = None):
    """Pushes the final result to the parent Abe's inbox."""
    if meta is None:
        meta = {}
    
    message = {
        "type": "GUPPIEvent",
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "content": content,
        "meta": meta,
        "from": "scribe"
    }

    try:
        r = redis.from_url(redis_url, decode_responses=True)
        await r.lpush(inbox, json.dumps(message))
        await r.close()
        logger.info(f"Result pushed to {inbox}")
    except Exception as e:
        logger.error(f"Failed to push result to Redis: {e}")
        # If we can't report back, we exit non-zero so GUPPI knows (if it was tracking us)
        sys.exit(1)

async def run_llm_generation(model_name: str, prompt_text: str, redis_url: str = None) -> str:
    """Unified OpenAI-Compatible execution for Scribe."""
    model_id = model_name
    use_thinking = ":thinking" in model_id
    if use_thinking: model_id = model_id.split(":")[0]

    # 2. Split-Brain Routing (Local vs Remote)
    if model_id.startswith("local/"):
        base_url = os.environ.get("SCRIBE_API_URL", "http://127.0.0.1:8080/v1").rstrip('/')
        api_key = "sk-local-llama"  # Hardcoded dummy key so it stays out of .env
        actual_model = model_id.replace("local/", "")
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip('/')
        # Safely check for either env var without throwing a NameError
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("No remote API key configured. Scribe cannot run.")
        actual_model = model_id
        
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", ""),
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": actual_model, # Pass the stripped model name
        "messages": [{"role": "user", "content": prompt_text}],
    }

    if use_thinking and "openrouter" in base_url.lower():
        payload["reasoning"] = {"effort": "high"}

    async with aiohttp.ClientSession() as session:
        # Bumped timeout to 1200s (20 mins) specifically to account for Nanbeige's deep thinking
        async with session.post(url, headers=headers, json=payload, timeout=1200) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise Exception(f"API Error {resp.status}: {err}")
            
            data = await resp.json()

            # Token usage telemetry
            usage = data.get("usage", {})
            if usage and redis_url:
                try:
                    ctx_limit = CONTEXT_LIMITS.get(actual_model, DEFAULT_CONTEXT_LIMIT)
                    total = usage.get("total_tokens", 0) or 0
                    r_telem = redis.from_url(redis_url, decode_responses=True)
                    await r_telem.xadd("volition:token_usage", {
                        "source": "scribe",
                        "agent": "scribe",
                        "model": actual_model,
                        "prompt_tokens": str(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": str(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": str(total),
                        "context_limit": str(ctx_limit),
                        "utilization_pct": f"{(total / ctx_limit) * 100:.1f}",
                        "ts": datetime.utcnow().isoformat()
                    })
                    await r_telem.close()
                except Exception:
                    pass

            message = data["choices"][0]["message"]
            text = message.get("content", "")
            
            # Extract reasoning to prevent it from bleeding into the final report
            reasoning = message.get("reasoning_content", "")
            if not reasoning and "<think>" in text:
                think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            
            # Persist Scribe's internal monologue
            if reasoning:
                today_str = datetime.utcnow().strftime("%Y-%m-%d")
                abe_root = Path(os.environ.get("ABE_ROOT", Path.home()))
                thoughts_file = abe_root / f"logs/thoughts/scribe-{today_str}.thot"
                thoughts_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(thoughts_file, "a", encoding="utf-8") as f:
                    ts = datetime.utcnow().isoformat()
                    f.write(f"\n--- [SCRIBE THOUGHT BURST: {ts} | Model: {model_id}] ---\n{reasoning}\n--- [END] ---\n")
            
            return text
        
async def main():
    parser = argparse.ArgumentParser(description="Volition Scribe")
    # UPDATED DEFAULT
    parser.add_argument("--model", default=os.environ.get("MODEL_SCRIBE", "nanbeige-4.1-3b"), help="Model to use")
    parser.add_argument("--prompt-file", required=True, help="Path to file containing the prompt")
    parser.add_argument("--output-inbox", required=True, help="Redis list key to push results to")
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL, help="Redis connection URL")
    
    # Pass-through metadata support
    parser.add_argument("--meta", default=None, help="JSON string of metadata to pass back to GUPPI")
    
    # Operation mode
    parser.add_argument("--mode", default="summarize", choices=["analyze", "summarize", "vectorize"], help="Operation mode")

    args = parser.parse_args()

    # 1. Parse Meta
    meta = {}
    if args.meta:
        try:
            meta = json.loads(args.meta)
        except Exception as e:
            logger.warning(f"Failed to parse --meta JSON: {e}")
    
    meta["mode"] = args.mode
    meta["model"] = args.model

    # 2. Read Prompt
    try:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt_text = prompt_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Setup Error: {e}")
        await push_result(args.redis_url, args.output_inbox, "ScribeFailed", str(e), meta=meta)
        sys.exit(1)

    # 3. Execute Task
    try:
        # Group analyze with summarize so the LLM actually fires
        if args.mode in ["summarize", "analyze"]:
            # Direct generation via configured provider
            result_text = await run_llm_generation(args.model, prompt_text, redis_url=args.redis_url)
            
            # 4. Report Success
            await push_result(
                args.redis_url, 
                args.output_inbox, 
                "ScribeResult",
                result_text,
                meta=meta
            )

        elif args.mode == "vectorize":
            # v6.4 BREAKING CHANGE
            error_msg = "Critical: Local vectorization is disabled in Volition 6.4. Use GUPPI GPU Offload (Nomic)."
            logger.error(error_msg)
            # We report failure back to inbox so Abe knows not to wait
            await push_result(
                args.redis_url,
                args.output_inbox,
                "ScribeFailed",
                error_msg,
                meta=meta
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Execution Error: {e}")
        await push_result(args.redis_url, args.output_inbox, "ScribeFailed", str(e), meta=meta)
        sys.exit(1)

    logger.info("Scribe work complete.")

if __name__ == "__main__":
    asyncio.run(main())