#!/usr/bin/env python3
"""
Volition Black Box Logger
Role: Immutable Audit Trail
Function: Subscribes to ALL comms and actions, writes to append-only log.
"""

import asyncio
import json
import os
import signal
from datetime import datetime
from pathlib import Path

import redis.asyncio as redis

# CONFIG
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
LOG_DIR = Path(os.environ.get("VOLITION_LOG_DIR", "/var/log/volition"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))


async def logger_daemon():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Use from_url to handle auth consistent with GUPPI, or add password= arg
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
    
    # We use two tasks: one for PubSub (Inboxes), one for Streams (Actions/Chat)
    pubsub = r.pubsub()
    await pubsub.psubscribe("inbox:*")
    
    print(f"[*] Volition Logger Active. Writing to {LOG_DIR}")

    async def _write(category, data):
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        fpath = LOG_DIR / f"volition-audit-{date_str}.log"
        ts = datetime.utcnow().isoformat()
        entry = f"[{ts}] [{category.upper()}] {json.dumps(data)}\n"

        try:
            fd = os.open(
                fpath,
                os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                0o640
            )
            with os.fdopen(fd, "a") as f:
                f.write(entry)

        except Exception as e:
            print(f"CRITICAL: Failed to write audit log {fpath}: {e}")



    # Task 1: Stream Reader (Actions, Chat)
    async def stream_reader():
        last_ids = {"volition:action_log": "$", "chat:general": "$", "chat:synchronous": "$", "volition:token_usage": "$"}
        while True:
            try:
                streams = await r.xread(last_ids, block=5000)
                if streams:
                    for stream_name, messages in streams:
                        last_ids[stream_name] = messages[-1][0] # Update ID
                        for msg_id, data in messages:
                            await _write(stream_name, data)
            except Exception as e:
                print(f"Stream Error: {e}")
                await asyncio.sleep(5)

    # Task 2: PubSub Reader (Inboxes)
    async def inbox_reader():
        async for msg in pubsub.listen():
            if msg["type"] == "pmessage":
                channel = msg["channel"]
                data = msg["data"]
                # Inboxes usually contain raw JSON strings, try to parse for cleaner logging
                try: data = json.loads(data)
                except: pass
                await _write(channel, data)

    await asyncio.gather(stream_reader(), inbox_reader())

if __name__ == "__main__":
    try:
        asyncio.run(logger_daemon())
    except KeyboardInterrupt:
        print("Logger Stopped.")