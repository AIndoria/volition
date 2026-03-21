#!/usr/bin/env python3
"""
Volition "Ear" - Social Router & Summarizer
Runs on: Sense-of-Proportion (Workstation)

Purpose:
  1. Listens to 'chat:general' for conversation activity.
  2. Detects "Bursts" of activity (messages occurring within a time window).
  3. When a burst concludes (silence > timeout), it generates a summary.
  4. (v7.0) Publishes "SocialDigest" to 'volition:social_digests' stream.
     (Replaces v6.4 Inbox Broadcasting for Ambient Awareness).

Features:
  - Auto-discovery of Abes via 'volition:heartbeat'.
  - Uses Ollama for summarization to save API costs.
  - v6.4: Session reuse, Participant Filtering.
  - v7.0: Temporal Metadata (start_ts/end_ts) for Orientation logic.
  - v7.0 Polish: Bounded Chat Buffer (Safety Valve).
"""

import asyncio
import json
import os
import sys
import time
import logging
import aiohttp
import redis.asyncio as redis
from collections import deque
from datetime import datetime

# --- Configuration ---
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api")
MODEL_SUMMARIZE = os.environ.get("MODEL_SUMMARIZE", "mistral") 

SUMMARIZE_BACKEND = os.environ.get("SUMMARIZE_BACKEND", "ollama").lower()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL_SUMMARIZE = os.environ.get(
    "OPENROUTER_MODEL_SUMMARIZE",
    "google/gemini-3-flash-preview"
)


# Tuning
BURST_SILENCE_THRESHOLD = 300.0  # Seconds of silence to consider a conversation "finished"
BURST_MIN_MESSAGES = 3          # Don't summarize single stray messages
MAX_CHAT_BUFFER = 100           # Safety: Force summary if buffer exceeds this
HEARTBEAT_TTL = 300             # How long to remember an Abe after last heartbeat

# v7.0 Stream Key
DIGEST_STREAM_KEY = "volition:social_digests"

CONTEXT_LIMITS = {
    "google/gemini-3-flash-preview": 1_048_576,
    "google/gemini-2.5-flash-preview": 1_048_576,
    "mistral": 32_768,
}
DEFAULT_CONTEXT_LIMIT = 32_768

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EAR] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ear")

class SocialRouter:
    def __init__(self):
        self.r = redis.from_url(REDIS_URL, decode_responses=True)
        self.active_abes = {} # {abe_name: last_seen_timestamp}
        self.chat_buffer = [] # List of formatted strings "Name: Message"
        self.burst_participants = set() # Track who is talking in current burst
        
        # v7.0 Temporal Tracking
        self.last_msg_time = 0.0
        self.burst_start_time = 0.0
        
        self.chat_cursor = "$" # Start reading from end of stream

    async def update_active_abes(self, heartbeat_data):
        """Updates the list of known active Abes from heartbeat stream."""
        try:
            # heartbeat payload example: {"abe": "abe-01", "ts": "...", ...}
            abe = heartbeat_data.get("abe")
            if abe:
                if abe not in self.active_abes:
                    logger.info(f"Discovered new Abe: {abe}")
                self.active_abes[abe] = time.time()
        except Exception as e:
            logger.error(f"Heartbeat parse error: {e}")

    async def prune_abes(self):
        """Removes Abes we haven't heard from in a while."""
        now = time.time()
        dead = [abe for abe, ts in self.active_abes.items() if now - ts > HEARTBEAT_TTL]
        for d in dead:
            logger.info(f"Abe timed out: {d}")
            del self.active_abes[d]

    async def generate_summary(self, session, conversation_text):
        """Uses Ollama to summarize the buffered conversation."""
        prompt = (
            "Summarize the following chat log briefly. "
            "Identify the participants, the main topic, and any decisions made.\n\n"
            f"{conversation_text}"
        )
        payload = {
            "model": MODEL_SUMMARIZE,
            "prompt": prompt,
            "stream": False
        }
        try:
            # Use shared session
            async with session.post(f"{OLLAMA_URL}/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        prompt_toks = data.get("prompt_eval_count", 0) or 0
                        comp_toks = data.get("eval_count", 0) or 0
                        total = prompt_toks + comp_toks
                        if total > 0:
                            ctx_limit = CONTEXT_LIMITS.get(MODEL_SUMMARIZE, DEFAULT_CONTEXT_LIMIT)
                            await self.r.xadd("volition:token_usage", {
                                "source": "ear",
                                "agent": "ear",
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
                else:
                    logger.error(f"Ollama summary failed: {resp.status}")
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
        return None
    

    async def generate_summary_openrouter(self, session, conversation_text):
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY not set for OpenRouter summarize")
            return None

        prompt = (
            "Summarize the following chat log briefly. "
            "Identify the participants, the main topic, and any decisions made.\n\n"
            f"{conversation_text}"
        )

        payload = {
            "model": OPENROUTER_MODEL_SUMMARIZE,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"OpenRouter summary failed ({resp.status}): {err}")
                    return None

                data = await resp.json()
                usage = data.get("usage", {})
                if usage:
                    try:
                        ctx_limit = CONTEXT_LIMITS.get(OPENROUTER_MODEL_SUMMARIZE, DEFAULT_CONTEXT_LIMIT)
                        total = usage.get("total_tokens", 0) or 0
                        await self.r.xadd("volition:token_usage", {
                            "source": "ear",
                            "agent": "ear",
                            "model": OPENROUTER_MODEL_SUMMARIZE,
                            "prompt_tokens": str(usage.get("prompt_tokens", 0) or 0),
                            "completion_tokens": str(usage.get("completion_tokens", 0) or 0),
                            "total_tokens": str(total),
                            "context_limit": str(ctx_limit),
                            "utilization_pct": f"{(total / ctx_limit) * 100:.1f}",
                            "ts": datetime.utcnow().isoformat()
                        })
                    except Exception:
                        pass
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenRouter summarize exception: {e}")
            return None

    # v7.0: Replaced broadcast_digest (Push) with publish_digest (Pull/Stream)
    async def publish_digest(self, summary, msg_count, participants):
        """Publishes the digest to the 'Notice Board' stream."""
        if not summary:
            return

        # Explicit timestamps for GUPPI Orientation logic
        entry = {
            "start_ts": self.burst_start_time,
            "end_ts": self.last_msg_time,
            "msg_count": msg_count,
            "participants": json.dumps(list(participants)),
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }

        try:
            await self.r.xadd(DIGEST_STREAM_KEY, entry)
            logger.info(f"Published Digest to {DIGEST_STREAM_KEY} ({msg_count} msgs).")
        except Exception as e:
            logger.error(f"Failed to publish digest: {e}")

    async def run(self):
        if SUMMARIZE_BACKEND == "ollama":
            logger.info(f"Running Summarize backend: Ollama ({MODEL_SUMMARIZE})")
        else:
            logger.info(f"Running Summarize backend: OpenRouter ({OPENROUTER_MODEL_SUMMARIZE})")

        
        # Init: check heartbeats immediately to populate list
        try:
            initial_hbs = await self.r.xrevrange("volition:heartbeat", count=20)
            for _, data in initial_hbs:
                await self.update_active_abes(data)
        except Exception:
            pass 

        # v6.4: Optimization - Single Session for lifecycle
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    streams = {
                        "chat:general": self.chat_cursor,
                        "volition:heartbeat": "$" 
                    }
                    
                    res = await self.r.xread(streams, count=50, block=1000)
                    
                    if res:
                        for stream, messages in res:
                            if stream == "chat:general":
                                self.chat_cursor = messages[-1][0]
                                
                                # v7.0: Capture start time of burst
                                if not self.chat_buffer:
                                    self.burst_start_time = time.time()
                                
                                for _, data in messages:
                                    sender = data.get("from", "Unknown")
                                    
                                    # Attempt to extract technical ID if possible
                                    tech_id = None
                                    for active_id in self.active_abes:
                                        if active_id in sender:
                                            tech_id = active_id
                                            break
                                    
                                    if tech_id: self.burst_participants.add(tech_id)
                                    
                                    content = data.get("content", "")
                                    self.chat_buffer.append(f"{sender}: {content}")
                                    self.last_msg_time = time.time()
                                    
                            elif stream == "volition:heartbeat":
                                for _, data in messages:
                                    await self.update_active_abes(data)

                    # --- Burst Logic ---
                    now = time.time()
                    
                    if int(now) % 60 == 0:
                        await self.prune_abes()
                    
                    # Safety Valve: Force flush if buffer gets too big
                    force_flush = len(self.chat_buffer) >= MAX_CHAT_BUFFER
                    if force_flush:
                        logger.warning(f"Buffer full ({len(self.chat_buffer)} msgs). Forcing summary.")

                    if self.chat_buffer:
                        silence_duration = now - self.last_msg_time
                        
                        if silence_duration > BURST_SILENCE_THRESHOLD or force_flush:
                            msg_count = len(self.chat_buffer)
                            
                            if msg_count >= BURST_MIN_MESSAGES:
                                logger.info(f"Burst concluded ({msg_count} msgs). Summarizing...")
                                full_text = "\n".join(self.chat_buffer)
                                if SUMMARIZE_BACKEND == "openrouter":
                                    summary = await self.generate_summary_openrouter(session, full_text)
                                else:
                                    summary = await self.generate_summary(session, full_text)

                                if summary:
                                    # v7.0: Publish instead of Broadcast
                                    await self.publish_digest(summary, msg_count, self.burst_participants)
                            else:
                                logger.info("Burst too short, discarding.")
                            
                            # Reset
                            self.chat_buffer = []
                            self.burst_participants = set()
                            self.burst_start_time = 0.0

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    await asyncio.sleep(1)

        await self.r.close()

if __name__ == "__main__":
    try:
        asyncio.run(SocialRouter().run())
    except KeyboardInterrupt:
        pass