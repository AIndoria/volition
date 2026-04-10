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
from datetime import datetime
from typing import Dict, Any, Optional

# --- Configuration ---
# Network
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"



# Ollama Endpoint
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api")

# Models
# 'nomic-embed-text' is the standard for RAG in Volition 6.4
MODEL_EMBED = os.environ.get("MODEL_EMBED", "nomic-embed-text")
# 'mistral' (7B) is VRAM safe. 'mistral-small' (22B) requires ~14GB VRAM.
MODEL_SUMMARIZE = os.environ.get("MODEL_SUMMARIZE", "mistral") 

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "ollama").lower()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

CONTEXT_LIMITS = {
    "google/gemini-embedding-001": 8_192,
    "mistral": 32_768,
    "nomic-embed-text": 8_192,
}
DEFAULT_CONTEXT_LIMIT = 32_768
OPENROUTER_MODEL_EMBED = os.environ.get(
    "OPENROUTER_MODEL_EMBED",
    "google/gemini-embedding-001"
)


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU-WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_worker")

async def check_ollama_status():
    """Verifies Ollama is reachable and models are loaded."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_URL.replace('/api', '')}") as resp:
                if resp.status == 200:
                    logger.info(f"Ollama connected at {OLLAMA_URL}")
                    return True
    except Exception as e:
        logger.critical(f"Ollama connection failed: {e}")
        return False
    return False

async def run_embedding(session: aiohttp.ClientSession, text: str, r: redis.Redis = None) -> Optional[list]:
    """Calls Ollama to generate a vector embedding."""
    payload = {
        "model": MODEL_EMBED,
        "prompt": text
    }
    try:
        async with session.post(f"{OLLAMA_URL}/embeddings", json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Embedding failed ({resp.status}): {err}")
                return None
            data = await resp.json()
            if r:
                try:
                    prompt_toks = data.get("prompt_eval_count", 0) or 0
                    if prompt_toks > 0:
                        ctx_limit = CONTEXT_LIMITS.get(MODEL_EMBED, DEFAULT_CONTEXT_LIMIT)
                        await r.xadd("volition:token_usage", {
                            "source": "gpu-worker",
                            "agent": "gpu-worker",
                            "model": MODEL_EMBED,
                            "prompt_tokens": str(prompt_toks),
                            "completion_tokens": "0",
                            "total_tokens": str(prompt_toks),
                            "context_limit": str(ctx_limit),
                            "utilization_pct": f"{(prompt_toks / ctx_limit) * 100:.1f}",
                            "ts": datetime.utcnow().isoformat()
                        })
                except Exception:
                    pass
            return data.get("embedding")
    except Exception as e:
        logger.error(f"Embedding exception: {e}")
        return None

async def run_embedding_openrouter(session: aiohttp.ClientSession, text: str, r: redis.Redis = None) -> Optional[list]:
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set for openrouter embedding")
        return None

    payload = {
        "model": OPENROUTER_MODEL_EMBED,
        "input": text
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with session.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json=payload
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"OpenRouter embedding failed ({resp.status}): {err}")
                return None

            data = await resp.json()
            usage = data.get("usage", {})
            if usage and r:
                try:
                    ctx_limit = CONTEXT_LIMITS.get(OPENROUTER_MODEL_EMBED, DEFAULT_CONTEXT_LIMIT)
                    total = usage.get("total_tokens", 0) or 0
                    await r.xadd("volition:token_usage", {
                        "source": "gpu-worker",
                        "agent": "gpu-worker",
                        "model": OPENROUTER_MODEL_EMBED,
                        "prompt_tokens": str(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": "0",
                        "total_tokens": str(total),
                        "context_limit": str(ctx_limit),
                        "utilization_pct": f"{(total / ctx_limit) * 100:.1f}",
                        "ts": datetime.utcnow().isoformat()
                    })
                except Exception:
                    pass
            return data["data"][0]["embedding"]

    except Exception as e:
        logger.error(f"OpenRouter embedding exception: {e}")
        return None


async def run_summary(session: aiohttp.ClientSession, text: str, r: redis.Redis = None) -> Optional[str]:
    """Calls Ollama to generate a summary."""
    # Context window management is up to the model, but we keep the prompt simple.
    prompt = f"Summarize the following text concisely, focusing on key events and technical details:\n\n{text}"
    payload = {
        "model": MODEL_SUMMARIZE,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192 # Attempt to use larger context if model supports it
        }
    }
    try:
        async with session.post(f"{OLLAMA_URL}/generate", json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Summary failed ({resp.status}): {err}")
                return None
            data = await resp.json()
            if r:
                try:
                    prompt_toks = data.get("prompt_eval_count", 0) or 0
                    comp_toks = data.get("eval_count", 0) or 0
                    total = prompt_toks + comp_toks
                    if total > 0:
                        ctx_limit = CONTEXT_LIMITS.get(MODEL_SUMMARIZE, DEFAULT_CONTEXT_LIMIT)
                        await r.xadd("volition:token_usage", {
                            "source": "gpu-worker",
                            "agent": "gpu-worker",
                            "model": MODEL_SUMMARIZE,
                            "prompt_tokens": str(prompt_toks),
                            "completion_tokens": str(comp_toks),
                            "total_tokens": str(total),
                            "context_limit": str(ctx_limit),
                            "utilization_pct": f"{(total / ctx_limit) * 100:.1f}",
                            "ts": datetime.utcnow().isoformat()
                        })
                except Exception:
                    pass
            return data.get("response")
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
            if EMBEDDING_BACKEND == "openrouter":
                vector = await run_embedding_openrouter(session, content, r=r)
                backend_name = OPENROUTER_MODEL_EMBED
            else:
                vector = await run_embedding(session, content, r=r)
                backend_name = MODEL_EMBED

            if vector:
                result_data = {"vector": vector}
            else:
                error_msg = f"Embedding generation failed ({EMBEDDING_BACKEND})"


        elif task_type == "summarize":
            summary = await run_summary(session, content, r=r)
            if summary:
                result_data = {"summary": summary}
            else:
                error_msg = "Ollama summary generation failed"

        else:
            error_msg = f"Unknown task type: {task_type}"


    model_meta = (
            backend_name
            if task_type == "embed"
            else MODEL_SUMMARIZE
        )
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
                "model": model_meta
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

    # Verify Ollama
    if EMBEDDING_BACKEND == "ollama":
        if not await check_ollama_status():
            logger.warning("Ollama not currently reachable. Waiting...")
    else:
        logger.info("Embedding backend set to OpenRouter (no Ollama required)") # Openrouter fallback

    
    async with aiohttp.ClientSession() as session:
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