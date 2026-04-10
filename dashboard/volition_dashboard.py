#!/usr/bin/env python3
"""
Volition Command Server
Backend: FastAPI + Redis
Frontend: React (served via static template)
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import List

import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# --- CONFIG ---
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
HUMAN_NAME = os.environ.get("HUMAN_NAME", "Human-Abe")

# --- APP SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_listener())
    yield

app = FastAPI(title="Volition Command", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- ROUTES ---

@app.get("/mobile", response_class=HTMLResponse)
async def get_mobile(request: Request):
    """Force load the mobile interface."""
    return templates.TemplateResponse(request=request, name="mobile.html", context={"human_name": HUMAN_NAME})

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Root route with device detection.
    If it looks like a phone, serve the mobile app.
    Otherwise, serve the desktop dashboard.
    """
    user_agent = request.headers.get("user-agent", "").lower()

    # "Automagic" detection
    if "mobile" in user_agent or "android" in user_agent or "iphone" in user_agent:
        return templates.TemplateResponse(request=request, name="mobile.html", context={"human_name": HUMAN_NAME})

    # Fallback to desktop
    return templates.TemplateResponse(request=request, name="index.html", context={"human_name": HUMAN_NAME})

# --- REDIS MANAGER ---
class RedisManager:
    def __init__(self):
        self.redis = None

    async def connect(self):
        if not self.redis:
            self.redis = redis.from_url(REDIS_URL, decode_responses=True)

    async def get_history(self, stream_key: str, count: int = 100):
        # Fetch deeper history (100) so we don't miss recent context
        try:
            data = await self.redis.xrevrange(stream_key, count=count)
            return list(reversed(data))
        except Exception as e:
            print(f"Error fetching history for {stream_key}: {e}")
            return []

    async def post_message(self, channel: str, user: str, content: str):
        entry = {
            "from": user,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "human_chat"
        }
        await self.redis.xadd(channel, entry)

    async def send_email(self, target: str, content: str, sender: str = HUMAN_NAME):
        key = f"inbox:{target}"
        msg = {
            "from": sender,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "NewInboxMessage",
            "content": content
        }
        await self.redis.lpush(key, json.dumps(msg))

        log_entry = {
            "id": f"human-{int(time.time())}",
            "type": "AbeTurn",
            "agent": sender,
            "timestamp_intent": datetime.utcnow().isoformat(),
            "status": "completed",
            "reasoning": "Manual Override",
            "action": {
                "tool": "email_send",
                "recipient": key,
                "message": content
            }
        }
        await self.redis.xadd("volition:action_log", {"entry": json.dumps(log_entry)})

    async def scan_channels(self):
        keys = []
        cursor = '0'
        try:
            while cursor != 0:
                cursor, batch = await self.redis.scan(cursor=cursor, match="chat:*")
                keys.extend(batch)
            return list(set(keys))
        except Exception:
            return []

rm = RedisManager()

# --- WEBSOCKET MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except:
                dead.append(connection)

        for d in dead:
            self.active_connections.remove(d)


manager = ConnectionManager()

# --- BACKGROUND WORKER ---
async def redis_listener():
    await rm.connect()

    streams = {
        "chat:general": "$",
        "chat:synchronous": "$",
        "volition:action_log": "$",
        "volition:heartbeat": "$",
        "volition:social_digests": "$",
        "volition:token_usage": "$"
    }

    last_scan = 0
    print("👂 Volition Backend Listening...")

    backoff = 1

    while True:
        try:
            # DYNAMIC CHANNEL DISCOVERY
            if time.time() - last_scan > 5:
                found_channels = await rm.scan_channels()
                for ch in found_channels:
                    if ch not in streams:
                        streams[ch] = "$"
                        print(f"found new channel: {ch}")
                        await manager.broadcast({"type": "channel_discovery", "channel": ch})
                last_scan = time.time()

            # READ STREAMS
            events = await rm.redis.xread(streams, count=1, block=100)
            if events:
                for stream_name, messages in events:
                    streams[stream_name] = messages[-1][0]
                    for msg_id, data in messages:
                        payload = {
                            "type": "stream_event",
                            "stream": stream_name,
                            "id": msg_id,
                            "data": data
                        }
                        await manager.broadcast(payload)
            backoff = 1
        except Exception as e:
            print(f"Redis Loop Error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

# --- ROUTES ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await rm.connect()

        # 1. Send Channels
        try:
            channels = await rm.scan_channels()
            for ch in channels:
                await websocket.send_text(json.dumps({"type": "channel_discovery", "channel": ch}))
        except Exception as e:
            print(f"WS Channel Scan Error: {e}")

        # 2. Send History with Error Handling
        
        # 2a. Chat Channels (Standard History - 100 items)
        for stream in ["chat:general", "chat:synchronous"]:
             try:
                 hist = await rm.get_history(stream, 100)
                 for msg_id, data in hist:
                     try:
                         await websocket.send_text(json.dumps({
                             "type": "stream_event",
                             "stream": stream,
                             "id": msg_id,
                             "data": data,
                             "is_history": True
                         }))
                     except Exception as inner_e:
                         print(f"Failed to serialize message {msg_id}: {inner_e}")
             except Exception as outer_e:
                 print(f"Failed to fetch history for {stream}: {outer_e}")

        # 2b. Action Log (Smart Filter: Recent + Capped Emails)
        try:
            # Call redis directly to get NEWEST first (xrevrange default)
            # Do NOT reverse it.
            hist = await rm.redis.xrevrange("volition:action_log", count=2000)
            
            count = 0
            email_counts_per_inbox = {}  # recipient -> count of deep-scan emails sent
            EMAIL_DEEP_SCAN_LIMIT = 50
            error_count = 0
            ERROR_DEEP_SCAN_LIMIT = 100
            ERROR_STATUSES = {"error", "self_correcting", "interrupted", "failed"}

            for msg_id, data in hist:
                should_send = False

                # Rule 1: Always send the most recent 100 events
                if count < 100:
                    should_send = True

                # Rule 2: Deep search for emails and errors beyond recent 100
                else:
                    try:
                        entry = json.loads(data.get("entry", "{}"))
                        tool = entry.get("action", {}).get("tool")

                        # 2a: Emails (up to 25 per inbox)
                        if tool == "email_send":
                            recipient = entry.get("action", {}).get("recipient", "")
                            inbox = re.sub(r"^inbox:", "", recipient, flags=re.IGNORECASE).strip()
                            if email_counts_per_inbox.get(inbox, 0) < EMAIL_DEEP_SCAN_LIMIT:
                                should_send = True
                                email_counts_per_inbox[inbox] = email_counts_per_inbox.get(inbox, 0) + 1

                        # 2b: Errors (up to 100 total)
                        if not should_send and error_count < ERROR_DEEP_SCAN_LIMIT:
                            status = entry.get("status", "")
                            results = entry.get("results", {}) if isinstance(entry.get("results"), dict) else {}
                            reasoning = entry.get("reasoning", "")
                            is_error = (
                                status in ERROR_STATUSES
                                or results.get("status") in {"error", "failed"}
                                or "error" in results
                                or (isinstance(reasoning, str) and reasoning.startswith("Error:"))
                            )
                            if is_error:
                                should_send = True
                                error_count += 1
                    except: pass

                if should_send:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "stream_event",
                            "stream": "volition:action_log",
                            "id": msg_id,
                            "data": data,
                            "is_history": True
                        }))
                    except Exception as inner_e:
                        print(f"Failed to serialize action log {msg_id}: {inner_e}")
                
                # Only increment count for the "Recent" bucket check
                # We iterate through all 2000, but only 'count' the first 100 as "Recent"
                count += 1
                
        except Exception as outer_e:
            print(f"Failed to fetch history for volition:action_log: {outer_e}")

        # 2c. Social Digests
        try:
             hist = await rm.get_history("volition:social_digests", 50)
             for msg_id, data in hist:
                 try:
                     await websocket.send_text(json.dumps({
                         "type": "stream_event",
                         "stream": "volition:social_digests",
                         "id": msg_id,
                         "data": data,
                         "is_history": True
                     }))
                 except Exception as inner_e:
                     print(f"Failed to serialize digest {msg_id}: {inner_e}")
        except Exception as outer_e:
             print(f"Failed to fetch history for volition:social_digests: {outer_e}")

        # 2d. Token Usage
        try:
            hist = await rm.get_history("volition:token_usage", 200)
            for msg_id, data in hist:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "stream_event",
                        "stream": "volition:token_usage",
                        "id": msg_id,
                        "data": data,
                        "is_history": True
                    }))
                except Exception as inner_e:
                    print(f"Failed to serialize token usage {msg_id}: {inner_e}")
        except Exception as outer_e:
            print(f"Failed to fetch history for volition:token_usage: {outer_e}")

        # 2e. Heartbeats (enough to populate all active agents immediately)
        try:
            hist = await rm.get_history("volition:heartbeat", 100)
            for msg_id, data in hist:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "stream_event",
                        "stream": "volition:heartbeat",
                        "id": msg_id,
                        "data": data,
                        "is_history": True
                    }))
                except Exception as inner_e:
                    print(f"Failed to serialize heartbeat {msg_id}: {inner_e}")
        except Exception as outer_e:
            print(f"Failed to fetch history for volition:heartbeat: {outer_e}")

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                sender = msg.get("sender", HUMAN_NAME)  # Identity Support

                if action == "post":
                    channel = msg.get("channel")
                    content = msg.get("content")
                    target_stream = f"chat:{channel}" if not channel.startswith("chat:") else channel
                    await rm.post_message(target_stream, sender, content)

                elif action == "email":
                    raw_target = msg.get("target", "")
                    content = msg.get("content")
                    # Handle multiple recipients
                    targets = [t.strip() for t in raw_target.split(",") if t.strip()]
                    for target in targets:
                        await rm.send_email(target, content, sender=sender)

            except Exception as e:
                print(f"WS Receive Error: {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Critical Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "volition_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
    )

