#!/usr/bin/env python3
"""
Volition Heartbeat Monitor (Hardened + Governance)
Watches 'volition:heartbeat'. 
1. If an Abe goes silent for > 3 minutes, triggers HIGH PRIORITY Ntfy alert.
2. ALSO notifies Abe-01 (The Steward) via inbox so he can investigate.
"""

import time
import os
import json
import requests
import redis
import sys
from datetime import datetime, timezone

# CONFIG
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
NTFY_URL = os.environ.get("NTFY_URL")

# FAIL LOUD: No default token in code. Export NTFY_TOKEN in your shell/systemd.
NTFY_TOKEN = os.environ.get("NTFY_TOKEN")
NTFY_ENABLED = bool(NTFY_URL)


# TUNING
ALERT_THRESHOLD = 180        # Alert if silent for 3 minutes
CHECK_INTERVAL = 10          # How often to check for dead Abes (approx)

def send_alert(abe_name, last_seen_str):
    """Sends a high-priority panic alert to Humans."""
    if not NTFY_ENABLED:
        print("[!] NTFY disabled; skipping human alert")
        return

    print(f"[!] FLATLINE: {abe_name} (Last seen: {last_seen_str})")
    try:
        headers = {
            "Title": f"Abe Down: {abe_name}",
            "Priority": "urgent",
            "Tags": "skull,rotating_light",
        }
        if NTFY_TOKEN:
            headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
        requests.post(
            NTFY_URL,
            data=f"🚑 FLATLINE DETECTED: {abe_name} has not reported in since {last_seen_str}.",
            headers=headers,
            timeout=5
        )
    except Exception as e:
        print(f"Failed to send Ntfy alert: {e}")


def notify_steward(r, dead_abe, last_seen_str):
    """Notifies Abe-01 that a peer has died."""
    # Don't ask Abe-01 to investigate himself.
    if dead_abe == "abe-01":
        return

    msg = {
        "from": "Volition-Overseer",
        "event_type": "SystemAlert",
        "content": f"CRITICAL: Heartbeat lost for {dead_abe}. Last seen: {last_seen_str}. Investigate status immediately.",
        "priority": "high"
    }
    
    try:
        r.lpush("inbox:abe-01", json.dumps(msg))
        print(f"[*] Notified Abe-01 to investigate {dead_abe}.")
    except Exception as e:
        print(f"Failed to notify Abe-01: {e}")

def main():
    print(f"[*] Heartbeat Monitor Active. Connecting to {REDIS_HOST}...")
    
    # Hardened Connection
    r = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        password=REDIS_PASSWORD, 
        decode_responses=True,
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    # State: { "abe-01": timestamp_float }
    known_abes = {}
    
    # We start at '$' (only new beats). 
    # If you restart this monitor, it learns about alive Abes as they beat.
    last_id = "$"
    
    while True:
        try:
            # 1. READ NEW BEATS (Block for 1s to keep loop tight but efficient)
            streams = r.xread({"volition:heartbeat": last_id}, count=100, block=1000)
            
            if streams:
                for _, messages in streams:
                    for msg_id, data in messages:
                        last_id = msg_id
                        abe = data.get("abe")
                        if abe:
                            if abe not in known_abes:
                                print(f"[+] Discovered new heartbeat: {abe}")
                            known_abes[abe] = time.time()

            # 2. CHECK FOR DEAD ABES
            now = time.time()
            # Iterate copy to allow modification
            for abe, last_ts in list(known_abes.items()):
                delta = now - last_ts
                
                if delta > ALERT_THRESHOLD:
                    # FLATLINE CONFIRMED
                    last_seen_str = datetime.fromtimestamp(last_ts, timezone.utc).strftime('%H:%M:%S UTC')
                    
                    # Action 1: Alert Humans
                    send_alert(abe, last_seen_str)
                    
                    # Action 2: Alert Abe-01
                    notify_steward(r, abe, last_seen_str)
                    
                    # Remove from active monitoring to prevent alert spam
                    # (Will re-add automatically if they heartbeat again)
                    print(f"[-] Removing {abe} from active monitoring until next beat.")
                    del known_abes[abe]

        except KeyboardInterrupt:
            print("\nMonitor Stopped.")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()