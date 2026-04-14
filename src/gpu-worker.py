#!/usr/bin/env python3
"""
Volition GPU Worker - "The Muscle"
Runs on: Sense-of-Proportion (Workstation) with 5070 Ti

Purpose:
  Offloads heavy compute tasks from the LXC containers to the GPU.
  Primary tasks:
  1. Generating Vector Embeddings (Nomic) for RAG/Memory.
  2. Generating Summaries (Mistral/Llama) for Scribes/Logs.

Usage:
  export REDIS_HOST=10.0.0.175
  export OLLAMA_URL=http://localhost:11434
  python3 gpu_worker.py
"""

import asyncio
import json
import os
import sys
import logging
import aiohttp
import redis.asyncio as redis
from typing import Dict, Any, Optional

# --- Configuration ---
# Network
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

# Models
MODEL_EMBED = os.environ.get("MODEL_EMBED", "local/nomic-embed-text")
MODEL_SUMMARIZE = os.environ.get("MODEL_SUMMARIZE", "local/mistral") 

# Routing URLs & Fallbacks
raw_ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
clean_ollama_base = raw_ollama_url.rstrip("/")
if clean_ollama_base.endswith("/api"):
    clean_ollama_base = clean_ollama_base[:-4]
elif clean_ollama_base.endswith("/v1"):
    clean_ollama_base = clean_ollama_base[:-3]
legacy_ollama = f"{clean_ollama_base}/v1"

PRO_API_URL = os.environ.get("PRO_API_URL", legacy_ollama).rstrip('/')
# FIX: Safely fallback to Ollama
EMBED_API_URL = os.environ.get("EMBED_API_URL", legacy_ollama).rstrip('/')
SUMMARIZE_API_URL = os.environ.get("SUMMARIZE_API_URL", legacy_ollama).rstrip('/')

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip('/')
REMOTE_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU-WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_worker")

async def check_api_status(session: aiohttp.ClientSession, base_url: str, label: str):
    """Pings the standard OpenAI-compatible /models endpoint to verify the API is alive."""
    # OpenRouter doesn't need health checks, only ping local/custom URLs
    if "openrouter" in base_url.lower():
        return True
        
    url = f"{base_url}/models"
    try:
        # Quick 5-second timeout so it doesn't hang forever
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                logger.info(f"{label} API healthy at {base_url}")
                return True
            else:
                logger.warning(f"⚠️ {label} API returned status {resp.status} at {base_url}")
                return False
    except Exception as e:
        logger.warning(f"❌ {label} API unreachable at {base_url}: {e}")
        return False

async def run_embedding(session: aiohttp.ClientSession, text: str) -> Optional[list]:
    """Unified OpenAI-Compatible embedding generation."""
    if MODEL_EMBED.startswith("local/"):
        base_url = EMBED_API_URL
        api_key = "sk-local-llama"
        actual_model = MODEL_EMBED.replace("local/", "")
        req_timeout = 1200
    else:
        base_url = OPENAI_BASE_URL
        api_key = REMOTE_API_KEY
        actual_model = MODEL_EMBED
        req_timeout = 120
        if not api_key:
            logger.error("No remote API key configured for embeddings.")
            return None

    payload = {
        "model": actual_model,
        "input": text
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{base_url}/embeddings"

    try:
        async with session.post(url, headers=headers, json=payload, timeout=req_timeout) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Embedding failed ({resp.status}): {err}")
                return None
            data = await resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding exception: {e}")
        return None

async def run_summary(session: aiohttp.ClientSession, text: str) -> Optional[str]:
    """Unified OpenAI-Compatible summary generation for the worker."""
    if MODEL_SUMMARIZE.startswith("local/"):
        base_url = SUMMARIZE_API_URL
        api_key = "sk-local-llama"
        actual_model = MODEL_SUMMARIZE.replace("local/", "")
        req_timeout = 1200
    else:
        base_url = OPENAI_BASE_URL
        api_key = REMOTE_API_KEY
        actual_model = MODEL_SUMMARIZE
        req_timeout = 120
        if not api_key:
            logger.error("No remote API key for GPU worker summarization.")
            return None

    prompt = f"Summarize the following text concisely, focusing on key events and technical details:\n\n{text}"
    payload = {
        "model": actual_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{base_url}/chat/completions"

    try:
        async with session.post(url, headers=headers, json=payload, timeout=req_timeout) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Summary failed ({resp.status}): {err}")
                return None
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Summary exception: {e}")
        return None
    
async def process_task(r: redis.Redis, session: aiohttp.ClientSession, raw_task: str):
    """Parses a task from the queue and routes it."""
    try:
        task = json.loads(raw_task)
    except json.JSONDecodeError:
        logger.error("Received malformed JSON task")
        return

    task_id = task.get("task_id", "unknown")
    task_type = task.get("type")
    content = task.get("content")
    reply_to = task.get("reply_to")

    logger.info(f"Processing Task {task_id} [{task_type}] -> {reply_to}")

    result_data = None
    error_msg = None

    # --- Router ---
    # v6.4 FIX: Check for empty/whitespace content to prevent 400 errors
    if not content or not isinstance(content, str) or not content.strip():
        error_msg = f"No valid content provided for {task_type}"
    else:
        if task_type == "embed":
            vector = await run_embedding(session, content)
            if vector:
                result_data = {"vector": vector}
            else:
                error_msg = "Embedding generation failed."


        elif task_type == "summarize":
            summary = await run_summary(session, content)
            if summary:
                result_data = {"summary": summary}
            else:
                error_msg = "Summary generation failed."

        else:
            error_msg = f"Unknown task type: {task_type}"

    # --- Reply ---
    if reply_to:
        response_payload = {
            "type": "GUPPIEvent",
            "event": "ScribeResult", # Standardized event type for GUPPI ingestion
            "task_id": task_id,
            "status": "success" if result_data else "error",
            "content": result_data if result_data else {"error": error_msg},
            "meta": {
                "worker": "gpu_5070ti",
                "model": MODEL_EMBED if task_type == "embed" else MODEL_SUMMARIZE
            }
        }
        
        try:
            # We explicitly push to the reply_to list (usually an inbox:abe-XX or temp:req:XX)
            await r.lpush(reply_to, json.dumps(response_payload))
            logger.info(f"Response pushed to {reply_to}")
        except Exception as e:
            logger.error(f"Failed to push response to Redis: {e}")

async def main():
    logger.info("Initializing GPU Worker...")
    logger.info(f"Models :: Embed: {MODEL_EMBED} | Sum: {MODEL_SUMMARIZE}")

    # Verify Redis
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        await r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}")
    except Exception as e:
        logger.critical(f"Redis connection failed: {e}")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        # Run health checks
        await check_api_status(session, EMBED_API_URL, "Embedding")
        if SUMMARIZE_API_URL != EMBED_API_URL:
            await check_api_status(session, SUMMARIZE_API_URL, "Summarization")

        logger.info("Listening on queue:gpu_heavy ...")
        while True:
            try:
                # BLPOP blocks until a task is available
                # 0 means block indefinitely
                _, raw_task = await r.blpop("queue:gpu_heavy", timeout=0)
                
                # Execute
                await process_task(r, session, raw_task)

            except asyncio.CancelledError:
                logger.info("Worker stopping...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(1) # Prevent tight loop on error

    await r.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
