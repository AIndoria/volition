#!/usr/bin/env python3
"""GUPPI daemon - Volition 7.8 (The Safety Update)
The "Body" of the Abe Agent.

Status: STABLE RELEASE 7.8
- Architecture: Refractory Scheduler (Hot Senses / Paced Workload)
- Logic: STRICT 7.2.3.1 COMPLIANCE
- Feature: Clipboard (Persistent Scratchpad)
- Safety: Deadman Switch (Anti-Ghosting)
- Safety: Output Machete (Source Truncation)
"""

import asyncio
import json
import os
import sys
import shutil
import time
import logging
import uuid
import tempfile
import shlex
import signal
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party libraries
import aiosqlite
import sqlite3
import redis.asyncio as redis
import asyncssh
import aiohttp 
import chromadb
from chromadb.config import Settings 
try:
    from google import genai 
except ImportError:
    pass 
# v6.1: Web Reading
try:
    import trafilatura
except ImportError:
    trafilatura = None 

# -------------------------------------------------------

# --- CONFIGURATION (Environment Overrides) ---
ABE_ROOT = Path(os.environ.get("ABE_ROOT", Path.home()))
IDENTITY_FILE = ABE_ROOT / os.environ.get("IDENTITY_FILE", ".abe-identity")
# v6.5: Identity Priors
PRIORS_SOURCE_FILE = ABE_ROOT / ".abe-priors.md"
PRIORS_STUB_FILE = ABE_ROOT / ".abe-priors.stub"

WORKING_LOG = ABE_ROOT / os.environ.get("WORKING_LOG", "working.log")
TODO_DB = ABE_ROOT / os.environ.get("TODO_DB", "todo.db")
BIN_DIR = ABE_ROOT / os.environ.get("BIN_DIR", "bin")
DOCS_DIR = ABE_ROOT / os.environ.get("DOCS_DIR", "docs")
MEMORY_DIR = ABE_ROOT / os.environ.get("MEMORY_DIR", "memory")
EPISODES_DIR = MEMORY_DIR / "episodes"
ARCHIVE_DIR = MEMORY_DIR / "tier_1_archive"
VECTOR_DB_PATH = MEMORY_DIR / "vector.db"
COMM_LOG = ABE_ROOT / "communications.log" # The "Mbox" archive
GENESIS_PROMPT_FILE = DOCS_DIR / os.environ.get("GENESIS_PROMPT_FILE", "0.0-Abe-Genesis_Prompt.md")
PROTOCOLS_FILE = DOCS_DIR / "Fleet_Protocols.md"
DOWNLOADS_DIR = MEMORY_DIR / "downloads"
# v7.2.1: Flight Recorder Log
LOGS_DIR = ABE_ROOT / "logs"
INBOX_DUMP_LOG = LOGS_DIR / "inbox_dump.jsonl"

# Network Config
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition") 
REDIS_URL = os.environ.get("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0")
NTFY_URL = os.environ.get("NTFY_URL")
NTFY_TOKEN = os.environ.get("NTFY_TOKEN")

if not NTFY_URL or not NTFY_TOKEN:
    logger.warning("NTFY not configured; human notifications disabled.")

# v6.1: Search Config
SEARXNG_URL = os.environ.get("SEARXNG_URL", "https://civitat.es/search") 

# API Config
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://volition.indoria.org")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Volition")

# v6.5: Split-Brain Config
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro") 
MODEL_PRO = os.environ.get("OPENROUTER_MODEL_PRO") or os.environ.get("GEMINI_MODEL_PRO") or GEMINI_MODEL
MODEL_FLASH = os.environ.get("OPENROUTER_MODEL_FLASH") or os.environ.get("GEMINI_MODEL_FLASH", "gemini-2.5-flash")

# v7.0: Social Stream Config
SOCIAL_DIGEST_STREAM = "volition:social_digests"

GOVERNOR_LIMIT = 15
GOVERNOR_WINDOW = 300 

# Behavior / Tuning
MAX_CONCURRENT_SUBPROCS = int(os.environ.get("MAX_CONCURRENT_SUBPROCS", 4))
SSH_CMD_TIMEOUT = float(os.environ.get("SSH_CMD_TIMEOUT", 300.0))
SUBPROC_TIMEOUT = float(os.environ.get("SUBPROC_TIMEOUT", 150.0))
REDIS_RETRY_ATTEMPTS = int(os.environ.get("REDIS_RETRY_ATTEMPTS", 3))
REDIS_RETRY_BASE = float(os.environ.get("REDIS_RETRY_BASE", 0.5))

# Lock Config
DEFAULT_LOCK_TTL_MS = 60000 

# Safety
STREAM_DENY_LIST = ["volition:action_log", "volition:heartbeat", "volition:log_stream"]
FLASH_FORBIDDEN_TOOLS = {"shell", "write_file", "spawn_abe", "remote_exec", "spawn_scribe"}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("guppi")


# --- UTILITY HELPERS ---

async def retry_async(func, *args, attempts=REDIS_RETRY_ATTEMPTS, **kwargs):
    """Retries an async function with exponential backoff and jitter."""
    last_ex = None
    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_ex = e
            delay = REDIS_RETRY_BASE * (2 ** (attempt - 1))
            delay = delay * (0.9 + (random.random() * 0.2))
            logger.debug(f"Op failed ({e}), retrying {attempt}/{attempts} in {delay:.2f}s...")
            if attempt == attempts: break
            await asyncio.sleep(delay)
    raise last_ex


class LLMOutputError(Exception):
    """Raised when the LLM returns garbage that json.loads hates."""
    pass


# [NEW] Volition 7.8: Clipboard Class
class Clipboard:
    """Manages the persistent scratchpad for the agent."""
    def __init__(self, filepath: Path):
        self.path = filepath
        
    def _read_lines(self) -> List[str]:
        if not self.path.exists(): return []
        lines = [line.strip() for line in self.path.read_text().splitlines() if line.strip()]
        return lines

    def read(self) -> str:
        lines = self._read_lines()
        if not lines: return "(Empty)"
        # Return formatted list with indices
        return "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])

    def add(self, content: str) -> str:
        lines = self._read_lines()
        # Simple deduplication
        if content in lines: return "Item already exists."
        lines.append(content)
        self.path.write_text("\n".join(lines))
        return f"Added item {len(lines)}"

    def remove(self, indices: List[int]) -> str:
        lines = self._read_lines()
        # Sort indices descending to avoid shifting problems
        indices = sorted(indices, reverse=True)
        removed_count = 0
        for idx in indices:
            # Adjust for 1-based index
            zero_idx = idx - 1
            if 0 <= zero_idx < len(lines):
                lines.pop(zero_idx)
                removed_count += 1
        
        self.path.write_text("\n".join(lines))
        return f"Removed {removed_count} item(s)."

    def clear(self) -> str:
        self.path.write_text("")
        return "Clipboard cleared."

class Governor:
    def __init__(self, abe_name, redis_client):
        self.abe_name = abe_name
        self.r = redis_client
        self.call_history = [] 
        self.cooldown_until = 0.0
        self._is_pruning = False

    async def check_limit(self) -> bool:
        now = time.time()
        self.call_history = [t for t in self.call_history if now - t < GOVERNOR_WINDOW]
        if len(self.call_history) >= GOVERNOR_LIMIT: 
            return False 
        self.call_history.append(now)
        return True

    async def set_status(self, state: str, reason: str = None):
        payload = {
            "state": state, 
            "reason": reason, 
            "timestamp": int(time.time()), 
            "host": os.uname().nodename
        }
        try: 
            # We use set with expiry to avoid stale status
            await retry_async(self.r.set, f"status:{self.abe_name}", json.dumps(payload), ex=3600*24)
        except: pass


class GuppiDaemon:
    def __init__(self):
        # 1. Identity
        self._refresh_identity()
        self.abe_name = self.identity.get("name", "unknown-abe")
        self.persona = self.identity.get("persona")
        self.display_name = f"{self.abe_name} ({self.persona})" if self.persona else self.abe_name

        # 2. Connections
        self.r = redis.from_url(REDIS_URL, decode_responses=True)
        self.governor = Governor(self.abe_name, self.r)
        
        # [NEW] 7.8: Initialize Clipboard
        self.clipboard = Clipboard(ABE_ROOT / f".abe-clipboard-{self.abe_name}.md")
        
        # v7.2: Dedicated Internal Queue for System Callbacks (Vectors/RPC)
        self.internal_queue = f"internal:{self.abe_name}"

        # 3. State
        self.running_subprocesses: Dict[str, asyncio.subprocess.Process] = {}
        self.log_buffer: List[Dict] = []
        self.log_lock = asyncio.Lock()
        
        # 4. Concurrency Control
        self.subproc_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUBPROCS)
        # 5. Lifecycle
        self._stopping = False
        self._bg_tasks: List[asyncio.Task] = []
        self._is_pruning = False
        
        self.processed_triggers = {}
        self.processed_triggers_ttl = 90

        
        # Subscriptions
        self.explicit_subscriptions = set()
        self.active_streams = {"chat:synchronous": "$", "volition:kill_switch": "$", "chat:general": "$"}
        # Load subs from disk
        self.subs_file = ABE_ROOT / ".abe-subscriptions"
        if self.subs_file.exists():
            try:
                subs = json.loads(self.subs_file.read_text())
                self.explicit_subscriptions.update(subs)
                for s in subs: self.active_streams[s] = "$"
            except: pass

        self.chroma_client = None
        self._local_wakeup = asyncio.Event() 
        self.cooldown_until = 0.0

        self._init_fs()
        self._init_db_sync()
        self._load_log_buffer()
        self._perform_crash_recovery() 
        
        # Orientation State
        self.last_sleep_ts = time.time()
        if self.log_buffer:
            try:
                # Attempt to find last valid timestamp to orient ourselves if we just restarted
                last = self.log_buffer[-1]
                ts_str = last.get("timestamp_outcome") or last.get("timestamp_event") or last.get("timestamp_intent")
                if ts_str:
                    self.last_sleep_ts = datetime.fromisoformat(ts_str).timestamp()
                    logger.info(f"Restored sleep state: {ts_str} (Duration: {time.time() - self.last_sleep_ts:.1f}s)")
            except: pass
        
        self.last_social_sync_ts = self.last_sleep_ts
        logger.info(f"GUPPI v7.8 Initialized for {self.abe_name}")
    
        # --- The Machete Helper ---
    def _truncate_output(self, text: str, limit: int = 20000) -> str:
        """Surgical tool to prevent context flooding from massive logs."""
        if not isinstance(text, str): return text
        if len(text) <= limit:
            return text
        cut_size = len(text) - limit
        return text[:limit] + f"\n... [Hey this is an automated thing set up by THE Abe -- Whatever you're trying, it was flagged because you're trying to spend more than {limit} chars this turn. This is unadvised. Try to reduce the intake. Original Err Message: TRUNCATED BY GUPPI SAFETY {cut_size} chars removed. Spawn a scribe with a specific task if you want to go over the entire file, or grep selectively (if you are certain what you're looking for) to read remainder.] ..."

    async def _monitor_subprocess(self, turn_id, proc):
        """Dedicated task to wait for a process and release semaphore."""
        try:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SUBPROC_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
            
            # [FIX] 7.8: Source Truncation (The Machete)
            # We truncate BEFORE creating the results dict to ensure the dirty payload never hits Redis.
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            
            # NOTE: We assume patch_abe_outcome handles the truncation to avoid double-logging
            # the truncation message, BUT we truncate massive buffers here solely to avoid 
            # crashing Python memory if it's GBs size.
            if len(stdout_str) > 100000: stdout_str = stdout_str[:100000] + "... [HARD CAP PRE-PATCH] ..."
            if len(stderr_str) > 100000: stderr_str = stderr_str[:100000] + "... [HARD CAP PRE-PATCH] ..."

            results = {"stdout": stdout_str, "stderr": stderr_str, "code": proc.returncode}
            await self.patch_abe_outcome(turn_id, results)
        except Exception:
            logger.exception(f"Error monitoring subproc {turn_id}")
        finally:
            self.subproc_semaphore.release()
            self.running_subprocesses.pop(turn_id, None)

    def _refresh_identity(self):
        """Loads identity from disk and updates in-memory state immediately."""
        try:
            if not IDENTITY_FILE.exists():
                self.identity = {"name": "abe-genesis", "temp": 1.0, "top_k": 0.9}
            else:
                self.identity = json.loads(IDENTITY_FILE.read_text())
        except Exception as e:
            logger.warning(f"Failed to read identity: {e}")
            self.identity = {"name": "abe-error", "parent": "unknown"}

        self.abe_name = self.identity.get("name", "unknown-abe")
        self.persona = self.identity.get("persona")
        if self.persona:
            self.display_name = f"{self.abe_name} ({self.persona})"
        else:
            self.display_name = self.abe_name
            
        logger.info(f"Identity Refreshed: {self.display_name}")

    def _init_fs(self):
        for d in [BIN_DIR, DOCS_DIR, EPISODES_DIR, ARCHIVE_DIR, MEMORY_DIR, DOWNLOADS_DIR, LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not WORKING_LOG.exists(): WORKING_LOG.touch()
        if not COMM_LOG.exists(): COMM_LOG.touch()
        if not INBOX_DUMP_LOG.exists(): INBOX_DUMP_LOG.touch()
        self._cleanup_overflow()

    def _init_db_sync(self):
        import sqlite3
        conn = sqlite3.connect(str(TODO_DB))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tasks
                     (task_id TEXT PRIMARY KEY, description TEXT, priority INTEGER,
                      due_timestamp TEXT, created_timestamp TEXT, source_abe TEXT, status TEXT)''')
        conn.commit()
        conn.close()

    def _load_log_buffer(self):
        self.log_buffer = []
        if WORKING_LOG.exists():
            with open(WORKING_LOG, 'r') as f:
                for line in f:
                    if line.strip():
                        try: self.log_buffer.append(json.loads(line))
                        except: continue

    def _perform_crash_recovery(self):
        """v5.9: Detects pending turns from a previous run and closes them."""
        recovered = False
        for entry in self.log_buffer:
            if entry.get("type") == "AbeTurn" and entry.get("status") == "pending":
                logger.warning(f"Crash Recovery: Found pending turn {entry.get('id')}. Marking interrupted.")
                entry["status"] = "interrupted"
                entry["results"] = {"error": "GUPPI Crash/Restart Detected"}
                entry["timestamp_outcome"] = datetime.utcnow().isoformat()
                recovered = True
        
        if recovered:
            self._rewrite_log_file_sync()

    def _rewrite_log_file_sync(self):
        try:
            with tempfile.NamedTemporaryFile('w', dir=str(WORKING_LOG.parent), delete=False) as tf:
                for entry in self.log_buffer:
                    tf.write(json.dumps(entry) + "\n")
                temp_path = Path(tf.name)
            os.replace(str(temp_path), str(WORKING_LOG))
            logger.info("Crash recovery complete. working.log patched.")
        except Exception as e:
            logger.error(f"Failed to patch log during recovery: {e}")
    def _get_daily_changelog_snippet(self, lines=30):
        """Reads the tail of today's changelog."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_path = ABE_ROOT / "logs" / f"changelog_{today}.md"
            
            if not log_path.exists():
                return "(No changelog entries for today yet.)"
                
            # Read explicitly with utf-8 to avoid encoding grief
            with open(log_path, 'r', encoding='utf-8') as f:
                # deque is efficient for tailing files
                from collections import deque
                tail = deque(f, maxlen=lines)
                
            return "".join(tail).strip()
        except Exception as e:
            return f"(Error reading changelog: {e})"

    # --- FORENSICS & SAFETY ---

    def _persist_raw_inbox(self, raw_data: Any):
        """Write-Ahead Log: Persist raw payload. Preserves JSON structure if possible."""
        try:
            entry = {
                "ts": datetime.utcnow().isoformat(),
                "payload": None
            }
            # Try to keep it as a native object if it's already a dict/list
            if isinstance(raw_data, (dict, list)):
                entry["payload"] = raw_data
            # If it's a string, try to parse it as JSON to store it structured
            elif isinstance(raw_data, str):
                try:
                    entry["payload"] = json.loads(raw_data)
                except:
                    entry["payload"] = raw_data
            # Fallback
            else:
                entry["payload"] = str(raw_data)

            with open(INBOX_DUMP_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush() 
                os.fsync(f.fileno())
        except Exception as e:
            logger.critical(f"FATAL: Failed to persist inbox message! {e}")
    
    # --- MEMORY OVERFLOW SYSTEM (v7.2.2) ---
    def _cleanup_overflow(self):
        """Prevents the overflow directory from growing infinitely."""
        overflow_dir = MEMORY_DIR / "overflow"
        if not overflow_dir.exists(): return

        # Retention Policy: 3 Days. 
        # If Abe hasn't looked at a log in 3 days, he's not going to.
        retention_seconds = 3 * 86400
        now = time.time()

        try:
            for f in overflow_dir.glob("*.txt"):
                if f.is_file() and f.stat().st_mtime < (now - retention_seconds):
                    f.unlink()
        except Exception as e:
            logger.warning(f"Overflow cleanup failed: {e}")

    def _sanitize_history_block(self, limit=20):
        """
        Returns context-safe history using the Overflow Pattern.
        - Most Recent Entry: kept intact up to 50k chars (working memory).
        - History Entries: truncated to 1k chars with file pointer (long-term ref).
        """
        sanitized = []
        buffer_copy = self.log_buffer[-limit:]
        overflow_dir = MEMORY_DIR / "overflow"
        overflow_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(buffer_copy):
            # The "Recency Rule": If it's the last item, Abe is looking at it RIGHT NOW.

            is_most_recent = (i == len(buffer_copy) - 1)
            char_limit = 50000 if is_most_recent else 1000
            
            new_entry = entry.copy()
            res = new_entry.get("results")
            turn_id = new_entry.get("id", "unknown")

            # [FIX] Recursively sanitize content field (e.g., massive inbox messages)
            if entry.get("type") == "GUPPIEvent" and "content" in entry:
                if isinstance(entry["content"], str):
                    new_entry["content"] = self._truncate_output(entry["content"], limit=char_limit)


            # Helper to process text fields
            def _process_text(text, suffix=""):
                if len(text) <= char_limit: return text
                # Deterministic Filename (Turn ID + Suffix)
                safe_name = f"{turn_id}{suffix}.txt"
                dump_path = overflow_dir / safe_name
                
                # Idempotent Write (Don't rewrite if exists, saves IO)
                if not dump_path.exists():
                    try: dump_path.write_text(text, encoding="utf-8")
                    except: return text[:char_limit] + "... [WRITE FAILED]"
                
                # --- FIX: USE DYNAMIC LIMIT ---
                # Calculate split size based on the specific limit for this entry (1000 or 50000)
                split_size = int(char_limit / 2)
                head = text[:split_size]
                tail = text[-split_size:]
                removed = len(text) - char_limit
                return (
                    f"{head}\n"
                    f"... [OUTPUT TRUNCATED: {removed} chars removed. Saved to: {safe_name}] ...\n"
                    f"{tail}"
                )

            if isinstance(res, str):
                new_entry["results"] = _process_text(res)
            elif isinstance(res, dict):
                res_copy = res.copy()
                if "stdout" in res_copy and isinstance(res_copy["stdout"], str):
                    res_copy["stdout"] = _process_text(res_copy["stdout"], "-stdout")
                if "stderr" in res_copy and isinstance(res_copy["stderr"], str):
                    res_copy["stderr"] = _process_text(res_copy["stderr"], "-stderr")
                new_entry["results"] = res_copy
                
            sanitized.append(new_entry)
        return json.dumps(sanitized, indent=2)
    
    def _parse_stream_id(self, stream_id: str):
        try:
            if "-" in stream_id:
                ts, seq = stream_id.split("-")
                return int(ts), int(seq)
            return int(stream_id), 0
        except: return 0, 0

    # --- RESTORED LOGIC FROM v7.2.3 ---
    async def _sync_social_history(self, start_ts: float, end_ts: float) -> List[Dict]:
        """Pulls missed social digests from 'The Ear'."""
        digests = []
        try:
            start_id = int(start_ts * 1000)
            end_id = int(end_ts * 1000)
            
            if end_id - start_id < 1000: return []

            raw_entries = await self.r.xrange(SOCIAL_DIGEST_STREAM, min=start_id, max=end_id)
            
            if raw_entries:
                logger.info(f"Syncing {len(raw_entries)} missed social digests...")
                with open(COMM_LOG, "a") as f:
                    for eid, data in raw_entries:
                        summary = data.get("summary", "")
                        count = data.get("msg_count", 0)
                        participants = data.get("participants", "[]")
                        gen_at = data.get("generated_at", datetime.utcnow().isoformat())
                        
                        # Archive to Mbox
                        log_entry = (
                            f"\n[{gen_at}] [SOCIAL DIGEST] ({count} msgs)\n"
                            f"Participants: {participants}\n"
                            f"Summary: {summary}\n"
                            f"{'-'*40}\n"
                        )
                        f.write(log_entry)
                        
                        # Add to return list for Orientation
                        digests.append({
                            "time": gen_at,
                            "summary": summary,
                            "count": count,
                            "participants": participants
                        })
                logger.info("Social history synced to communications.log")
        except Exception as e:
            logger.error(f"Failed to sync social history: {e}")
        return digests

    def _normalize_inbox_payload(self, raw_data: Any) -> Dict:
        """Classifies incoming messages to separate Signal (Human/Chat) from Noise (System/Scribe)."""
        norm = {
            "observed": {
                "raw": raw_data, "event_type": None, "from": None, "meta": {}, "content": None
            },
            "derived": { "kind": "Unknown", "inferred": False }
        }
        
        data = raw_data
        if isinstance(raw_data, bytes):
            try: data = raw_data.decode('utf-8')
            except: pass
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict): data = parsed
            except: pass
            
        if isinstance(data, dict):
            norm["observed"]["raw"] = data 
            norm["observed"]["event_type"] = data.get("event_type", data.get("event"))
            norm["observed"]["from"] = data.get("from")
            norm["observed"]["meta"] = data.get("meta", {})
            norm["observed"]["content"] = data.get("content") or data.get("results")
            
            # --- [FIX] ROBUST ACTION_ID EXTRACTION ---
            # We check Top Level -> Content/Results -> Meta to find the UUID.
            # This prevents "Id Blindness" where identical results are deduped as duplicates.
            action_id = None
            
            # 1. Top Level (Standard GUPPI Event)
            if data.get("action_id"): 
                action_id = data.get("action_id")
            
            # 2. Inside Content/Results (e.g. some internal RPCs)
            if not action_id:
                cont = norm["observed"]["content"]
                if isinstance(cont, dict):
                    action_id = (
                        cont.get("action_id")
                        or cont.get("actionId")
                        or cont.get("task_id")
                        or cont.get("id")
                    )

            
            # 3. Inside Meta (Scribe/Maintenance jobs)
            if not action_id:
                action_id = norm["observed"]["meta"].get("action_id")

            # Clean it up
            if action_id and isinstance(action_id, str):
                norm["observed"]["action_id"] = action_id.strip()
            # -----------------------------------------
            
            et = norm["observed"]["event_type"]
            # ... (Rest of classification logic remains the same) ...
            if et in ["NewInboxMessage", "NewChatMessage"]:
                norm["derived"]["kind"] = "HumanMessage" 
            elif et in ["TaskCompleted", "ScribeResult"]:
                norm["derived"]["kind"] = "ScribeResult"
            elif et in ["SystemAlert", "AlarmClock"]:
                norm["derived"]["kind"] = "SystemEvent"
            else:
                norm["derived"]["kind"] = "StructuredMessage"
        else:
            norm["observed"]["raw"] = str(data) 
            norm["observed"]["content"] = str(data)
            norm["derived"]["kind"] = "RawMessage" 
            norm["derived"]["inferred"] = True
            
        return norm

    def _archive_inbox_message(self, norm: Dict):
        """Archives human communications to communications.log (The Mbox). Ignores System Noise."""
        try:
            timestamp = datetime.utcnow().isoformat()
            sender = norm["observed"].get("from") or "unknown"
            kind = norm["derived"]["kind"]
            
            # FILTER: Only archive actual communication, not system noise
            should_archive = False
            if kind in ["HumanMessage", "StructuredMessage", "RawMessage", "Unknown"]:
                should_archive = True
            elif kind in ["ScribeResult", "SystemEvent"]:
                should_archive = False

            if should_archive:
                body = norm["observed"]["raw"]
                if isinstance(body, (dict, list)): body = json.dumps(body, indent=2)
                else: body = str(body)
                
                entry = (
                    f"\n[{timestamp}] FROM: {sender} (Type: {norm['observed']['event_type']})\n"
                    f"{body}\n{'-'*40}\n"
                )
                with open(COMM_LOG, "a") as f: f.write(entry)
        except Exception as e:
            logger.error(f"Failed to archive message to Mbox: {e}")

    async def _rewrite_log_file(self):
        dirpath = WORKING_LOG.parent
        with tempfile.NamedTemporaryFile('w', dir=str(dirpath), delete=False) as tf:
            for entry in self.log_buffer:
                tf.write(json.dumps(entry) + "\n")
            temp_path = Path(tf.name)
        os.replace(str(temp_path), str(WORKING_LOG))

    # --- LOG PRUNING WITH SCRIBE (RESTORED 7.2.3.1) ---
    async def _prune_logs(self):
        logger.info("[PRUNE] we ENTER _prune_logs")
        if self._is_pruning: return # Double check
        self._is_pruning = True
        logger.info("[PRUNE] method _is_pruning set TRUE")
        try:
          ts = int(time.time())
          archive_path = ARCHIVE_DIR / f"log-{ts}.jsonl"
          try: shutil.copy2(WORKING_LOG, archive_path)
          except: pass
          try:
              log_content = archive_path.read_text()
          except Exception as e:
              log_content = f"Error reading log: {e}"

          # v7.1: Improved Narrative Prompt
          prompt = (
              f"Synthesize these logs into a Tier 2 Episode Memory.\n"
              f"Focus on the NARRATIVE arc of what you accomplished or discovered.\n"
              f"IGNORE trivial mechanical steps (e.g., successful 'ls' or 'cd' commands) unless they revealed something critical.\n"
              f"If you were asleep or idle, state that clearly and briefly.\n\n"
              f"REQUIRED OUTPUT FORMAT:\n\n"
              f"## Narrative Summary\n"
              f"(A 2-3 sentence overview of the episode's main events)\n\n"
              f"## Key Decisions & Outcomes\n"
              f"(Bullet points of meaningful choices made and their results)\n\n"
              f"## Changed State / New Knowledge\n"
              f"(What is different now compared to the start? New files? New constraints?)\n\n"
              f"## Pending / Unresolved\n"
              f"(Only list actual blockers or unfinished tasks that require future attention that WERE in the logs, Do not make assumptions.)\n\n"
              f"Source log:\n{log_content}"
          )
          
          with tempfile.NamedTemporaryFile('w', delete=False) as pf:
              pf.write(prompt)
              prompt_path = pf.name
          
          meta_json = json.dumps({
              "maintenance": True,
              "source_tier_1": f"log-{ts}.jsonl",
              "mode": "summarize" 
          })
          
          # v7.1: Use Pro model (Thinking) for better synthesis quality
          current_model = MODEL_PRO or "google/gemini-3-flash-preview:thinking"
          
          cmd = [
              sys.executable, str(BIN_DIR / "scribe.py"), 
              "--model", current_model, 
              "--prompt-file", prompt_path, 
              "--output-inbox", f"inbox:{self.abe_name}",
              "--mode", "summarize",
              "--meta", meta_json
          ]
          await self._spawn_subprocess_exec(f"auto-prune-{ts}", cmd, tracked=False)

          async with self.log_lock:
              self.log_buffer = self.log_buffer[-15:]
              await self._rewrite_log_file()
        finally:
          logger.info("[PRUNE] EXIT _prune_logs (before reset)")
          self._is_pruning = False


    # --- EVENT & INTENT LOGGING ---

    async def log_guppi_event(self, event_type, content, source="GUPPI") -> str:

        # [FIX] Truncate the content immediately upon entry to prevent echo bloat.
        if isinstance(content, str):
            truncated_content = self._truncate_output(content)
        elif isinstance(content, dict):
             # Shallow copy to avoid mutating original payload if it's used elsewhere
             truncated_content = content.copy()
        else:
            truncated_content = content
        evt_id = f"evt-{uuid.uuid4().hex[:8]}"
        entry = {
            "id": evt_id, "type": "GUPPIEvent", "agent": self.abe_name,
            "timestamp_event": datetime.utcnow().isoformat(),
            "event_type": event_type, "source": source, "content": truncated_content
        }

        async with self.log_lock:
            self.log_buffer.append(entry)
            try: await self._rewrite_log_file()
            except: logger.exception("Failed local event log")
        return evt_id

    async def log_abe_intent(self, turn_id, parent_evt_id, reasoning, action, thought_signature=None):
        entry = {
            "id": turn_id, "type": "AbeTurn", "agent": self.abe_name,
            "parent_event_id": parent_evt_id, 
            "timestamp_intent": datetime.utcnow().isoformat(),
            "status": "pending", "reasoning": reasoning, "action": action, "results": None
        }
        if thought_signature: entry["thought_signature"] = thought_signature
        
        async with self.log_lock:
            self.log_buffer.append(entry)
            try: await self._rewrite_log_file()
            except: logger.exception("Failed local intent log")

        try:
            await retry_async(self.r.xadd, "volition:action_log", {"entry": json.dumps(entry)})
        except: logger.warning("Failed to stream intent to governance log")

    async def patch_abe_outcome(self, turn_id, results, notify=True):
        # --- SAFETY: TRUNCATE MASSIVE OUTPUTS (The Wallet Saver) ---
        MAX_OUT_LEN = 20000 
        truncated_results = results.copy() if isinstance(results, dict) else results
        
        if isinstance(truncated_results, dict):
            for k in ["stdout", "stderr"]:
                if isinstance(truncated_results.get(k), str) and len(truncated_results[k]) > MAX_OUT_LEN:
                    original_len = len(truncated_results[k])
                    head = truncated_results[k][:MAX_OUT_LEN]
                    truncated_results[k] = (
                        f"{head}\n... [TRUNCATED BY GUPPI SAFETY: {original_len - MAX_OUT_LEN} chars removed] ..."
                    )
        # ----------------------------------------
        found = False
        entry_snapshot = None
        async with self.log_lock:
            for entry in self.log_buffer:
                if entry.get("id") == turn_id:
                    entry["status"] = "completed"
                    entry["timestamp_outcome"] = datetime.utcnow().isoformat()
                    # We write the truncated result to the log to save disk/token space on context read
                    entry["results"] = truncated_results
                    found = True
                    entry_snapshot = entry
                    break
            if found:
                try: await self._rewrite_log_file()
                except: logger.exception("Failed local outcome patch")

        if found and entry_snapshot:
            try: await retry_async(self.r.xadd, "volition:action_log", {"entry": json.dumps(entry_snapshot)})
            except: pass

        if notify:
            try:
                # [FIXED LOGIC] We intentionally send truncated_results to Redis too. 
                # Sending 900k chars to Redis chokes the network and invalidates the next turn.
                msg = {"type": "GUPPIEvent", "event": "TaskCompleted", "action_id": turn_id, "results": truncated_results}
                
                # Pushing to own inbox triggers the next Refractory Cycle
                await retry_async(self.r.lpush, f"inbox:{self.abe_name}", json.dumps(msg))
                self._local_wakeup.set() # Wake up main loop
            except Exception as e:
                logger.critical(f"FATAL: Failed to notify inbox of task completion! {turn_id} Error: {e}")
                    
        else:
            logger.warning(f"Orphaned task completion: {turn_id}")

    

    async def _ingest_tier2(self, norm: Dict):
        """v6.5: Ingests Tier 2 episodes and offloads vectorization to GPU Queue."""
        try:
            meta = norm["observed"].get("meta", {})
            content = str(norm["observed"].get("content", ""))
            
            # v6.5: Safer Check (Restored from 6.4.3)
            if meta.get("mode") == "summarize" and content:
                source_file = meta.get("source_tier_1", "unknown_source.jsonl")
                summary_text = content
                
                # v7.2 Fix: UUIDs prevent timestamp race conditions
                file_uuid = uuid.uuid4().hex
                iso_ts = datetime.utcnow().isoformat()
                filename = f"ep-{file_uuid}.md"
                ep_path = EPISODES_DIR / filename

                if not summary_text.strip().startswith("---"):
                    current_model = MODEL_FLASH or "gemini-3-flash-preview"
                    header = f"---\ngenerated_at: {iso_ts}\ntype: tier_2_episode\nmodel: {current_model}\nsource_tier_1: {source_file}\n---\n\n"
                    summary_text = header + summary_text

                ep_path.write_text(summary_text)
                logger.info(f"Ingested Tier 2 Episode: {filename}")
                # v7.2 Fix: Use Internal Queue for routing
                task_payload = {
                    "task_id": f"vec-{file_uuid}", 
                    "type": "embed", 
                    "content": summary_text, 
                    "reply_to": self.internal_queue
                }
                await retry_async(self.r.lpush, "queue:gpu_heavy", json.dumps(task_payload))
                logger.info(f"Offloaded vectorization for {filename} to {self.internal_queue}")
                
        except Exception as e:
            logger.error(f"Failed to ingest Tier 2: {e}")

    async def heartbeat_loop(self):
        while not self._stopping:
            try:
                payload = {
                    "abe": self.abe_name,            
                    "display": self.display_name,    
                    "ts": datetime.utcnow().isoformat(), 
                    "host": os.uname().nodename
                }
                logger.info(f"❤️ Heartbeat: Buffer={len(self.log_buffer)} Pruning={self._is_pruning}")
                await retry_async(self.r.xadd, "volition:heartbeat", payload)

                # cheap check only
                if len(self.log_buffer) > 30 and not self._is_pruning:
                    logger.info(f"Entered buffer greater than 30 and not self.is_pruning block.")
                    asyncio.create_task(self._prune_logs())

            except Exception as e:
                logger.error(f"Heartbeat issue: {e}")
            await asyncio.sleep(60)


    

    # --- NEW TASK HANDLERS (Refractory) ---

    async def get_alarm_sleep_time(self) -> float:
        """Calculates sleep time based on next due task."""
        try:
            # FIX 1: Use str(TODO_DB) and add timeout
            async with aiosqlite.connect(str(TODO_DB), timeout=5.0) as db:
                # OPTIONAL: Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL;")
                await db.execute("PRAGMA busy_timeout = 5000;")
                
                # FIX 2: Filter out garbage rows
                query = "SELECT due_timestamp FROM tasks WHERE status != 'completed' AND due_timestamp IS NOT NULL AND due_timestamp != '' ORDER BY due_timestamp ASC LIMIT 1"
                async with db.execute(query) as cursor:
                    row = await cursor.fetchone()
                    if not row: return 3600 * 24 # Default long sleep
                    
                    ts_str = row[0]
                    
                    # FIX 3: Handle Space vs T format mismatch
                    if " " in ts_str and "T" not in ts_str:
                        ts_str = ts_str.replace(" ", "T")
                    
                    try:
                        due = datetime.fromisoformat(ts_str)
                    except ValueError:
                        # Fallback for weird formats, try stripping timezone/offsets
                        cleaned = ts_str.split('+')[0].split('Z')[0]
                        due = datetime.fromisoformat(cleaned)

                    # FIX 4: Normalize to Naive UTC (The "Timezone Crash" Fix)
                    if due.tzinfo is not None:
                        due = due.astimezone(timezone.utc).replace(tzinfo=None)
                    
                    now = datetime.utcnow()
                    delta = (due - now).total_seconds()
                    
                    # Prevent Insomnia Loop on overdue tasks
                    if delta < 0:
                        return 300.0
                    
                    return max(0.1, delta)

        except aiosqlite.OperationalError as e:
            # Likely a lock. Log it and sleep briefly (30s).
            logger.warning(f"Sleep calc DB lock or operational error: {e}")
            return 30.0
            
        except Exception as e:
            # Real crash. Log it!
            logger.error(f"get_alarm_sleep_time failed: {e}") 
            return 300.0

    # Subprocess lifecycle is owned exclusively by _monitor_subprocess.
    # check_subprocesses performs hygiene only.
    async def check_subprocesses(self):
            """Checks status of running Scribes/Shells."""
            # Simple cleanup of zombie references
            active = {}
            for tid, proc in self.running_subprocesses.items():
                if proc.returncode is None:
                    active[tid] = proc
            self.running_subprocesses = active

        

    # --- FINAL HYBRID HANDLER ---
    # Combines 7.7 Maintenance Logic with 7.2.3 Context Safety
    async def _handle_inbox_item(self, res, orientation_data=None):
        """Processes a raw item popped from Redis inbox."""
        if not res: return
        queue_name, raw_data = res
        
        # 1. Persist (Safety)
        # (Assumes you applied the _persist_raw_inbox fix we just discussed)
        self._persist_raw_inbox(raw_data)
        
        # 2. Normalize 
        norm = self._normalize_inbox_payload(raw_data) 

        # --- [NEW] ROBUST DEDUPLICATION ---
        now = time.time()
        try:
            observed = norm.get("observed", {}) or {}
            # Prefer explicit IDs
            action_id = (
                observed.get("action_id")
                or observed.get("meta", {}).get("action_id")
                or observed.get("meta", {}).get("id")
            )
            evt_type = observed.get("event_type") or observed.get("event") or "unknown"
            # 2. [FIX] Bypass Deduplication for Scribe/Maintenance
            # These often look identical (same meta, similar content) but must run every time.
            meta = observed.get("meta", {})
            is_maintenance = (
                meta.get("maintenance") is True
                or "source_tier_1" in meta
                or meta.get("mode") == "summarize"
            )

            if evt_type == "ScribeResult" or is_maintenance:
                 # Force a unique ID to bypass the hash check
                 trigger_id = f"scribe:{uuid.uuid4()}"
            else:
            # 3. Standard Deduplication (Keep your existing robust logic here)
                # Stable fingerprint
                content = observed.get("content") or observed.get("raw") or ""
                
                if isinstance(content, (dict, list)):
                    # Sort keys so {"a":1, "b":2} == {"b":2, "a":1}
                    content_snip = json.dumps(content, sort_keys=True)[:300]
                else:
                    content_snip = str(content)[:300]

                trigger_id = action_id if action_id else f"{evt_type}:{hash(content_snip)}"
        except Exception:
            trigger_id = f"raw:{hash(str(raw_data)[:300])}"

        # Prune old entries
        cutoff = now - self.processed_triggers_ttl
        self.processed_triggers = {k: v for k, v in self.processed_triggers.items() if v > cutoff}

        if trigger_id in self.processed_triggers:
            logger.debug(f"🔕 Dropping duplicate inbox trigger: {trigger_id}")
            return

        self.processed_triggers[trigger_id] = now
        # ----------------------------------
        self._archive_inbox_message(norm)             
        
        # 3. Optional Tier 2 Ingest (Text only)
        await self._ingest_tier2(norm)

        # 4. MAINTENANCE GATES (The 7.7 Fix)
        meta = norm["observed"].get("meta", {})
        
        # A. Identity Stub Update
        if meta.get("job_type") == "update_stub":
            content = str(norm["observed"].get("content", ""))
            if content:
                try:
                    PRIORS_STUB_FILE.write_text(content)
                    await self.log_guppi_event("Maintenance", "Updated Identity Stub", source="GUPPI")
                except Exception as e:
                    logger.error(f"Failed to write stub: {e}")
            return # <--- EXIT without Thinking

        # B. Silent Scribe / Background Tasks
        if meta.get("maintenance") is True or "source_tier_1" in meta:
            await self.log_guppi_event("MaintenanceCompleted", f"Silent Scribe: {meta}", source="GUPPI:Background")
            return # <--- EXIT without Thinking

        # 5. THINKING TRIGGER (The 7.2.3 Safety)
        # We pass norm["observed"] (The Envelope) so the LLM sees 'from', 'meta', and 'raw'.
        # GPT hates this because it's "messy", but it prevents context loss.
        
        parent_evt_id = await self.log_guppi_event("NewInboxMessage", norm["observed"], source=f"inbox:{self.abe_name}")
        trigger_data = {"event": "Inbox", "payload": norm["observed"]}
        
        await self.run_think_cycle(trigger_data, parent_evt_id, orientation_data=orientation_data)

    def _sanitize_log_content(self, content: Any, limit: int = 20000) -> Any:
        """
        Targeted, schema-aware truncation. 
        Only truncates known bloat keys to protect structural data integrity.
        """
        if isinstance(content, str):
            return self._truncate_output(content, limit)

        if not isinstance(content, dict):
            return content

        # The keys we know can hold massive, untruncated text blobs
        bloat_keys = {"raw", "content", "results", "stdout", "stderr", "message", "summary"}

        def _clean(data):
            if isinstance(data, dict):
                new_dict = {}
                for k, v in data.items():
                    if k in bloat_keys and isinstance(v, str):
                        new_dict[k] = self._truncate_output(v, limit)
                    else:
                        new_dict[k] = _clean(v)
                return new_dict
            elif isinstance(data, list):
                return [_clean(item) for item in data]
            else:
                return data

        return _clean(content)

    async def log_guppi_event(self, event_type, content, source="GUPPI") -> str:
        # [FIX] Schema-aware truncation to prevent echo bloat without breaking ABI
        truncated_content = self._sanitize_log_content(content)

        evt_id = f"evt-{uuid.uuid4().hex[:8]}"
        entry = {
            "id": evt_id, "type": "GUPPIEvent", "agent": self.abe_name,
            "timestamp_event": datetime.utcnow().isoformat(),
            "event_type": event_type, "source": source, "content": truncated_content
        }

        async with self.log_lock:
            self.log_buffer.append(entry)
            try: await self._rewrite_log_file()
            except: logger.exception("Failed local event log")
        return evt_id

    async def _handle_alarm(self, orientation_data=None):
        """Checks todo.db for due tasks and wakes the agent if needed."""
        now_ts = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(str(TODO_DB)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE status != 'completed' AND due_timestamp <= ? ORDER BY due_timestamp ASC LIMIT 5",
                (now_ts,)
            ) as cursor:
                due_tasks = await cursor.fetchall()
        
        if not due_tasks: return

        tasks_list = [dict(row) for row in due_tasks]
        trigger_data = {
            "event": "Alarm",
            "due_tasks": tasks_list
        }
        
        parent_evt_id = await self.log_guppi_event("SystemAlarm", {"count": len(tasks_list)}, source="System")
        await self.run_think_cycle(trigger_data, parent_evt_id, orientation_data=orientation_data)

    async def _handle_internal_item(self, res):
        """Handles responses from GPU Worker or Scribe."""
        if not res: return
        _, raw_data = res

        # 1. Write-Ahead Log (Safety First)
        self._persist_raw_inbox(raw_data)

        # 2. Parse JSON safely
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.error(f"Internal Queue JSON Decode Failed: {str(raw_data)[:100]}...")
            return
        except Exception as e:
            logger.error(f"Internal Queue Unexpected Error: {e}")
            return

        # 3. Logic (Now safe because 'data' is defined)
        logger.info(f"Internal Queue Received: {str(data)[:100]}...")
        
        # Vector Result (GPU Worker)
        if data.get("event") == "ScribeResult" and "vector" in data.get("content", {}):
            await self._handle_vector_result(data)
            return
        
        if data.get("type") == "embed":
            # legacy form from earlier workers, should remove around 8.0
            await self._handle_vector_result(data)
            return

        # Generic/Legacy Hook
        if "rag_result" in data:
             await self.log_guppi_event("InternalResult", data, source="Internal")

    
    async def _fetch_chat_context(self, stream_name, count=5):
        try:
            raw = await self.r.xrevrange(stream_name, count=count)
            context = []
            for msg_id, data in reversed(raw):
                context.append({
                    "id": msg_id,
                    "from": data.get("from", "unknown"),
                    "content": data.get("content", ""),
                    "timestamp": data.get("timestamp", "")
                })
            return context
        except: return []

    # --- MAIN LOOP (Refractory Scheduler + Orientation) ---

    

    async def main_wait_loop(self):
        logger.info("Entering Main Event Loop (Volition 7.8: Refractory + Orientation)...")
        await self.governor.set_status("idle")
        
        # 1. RESTORED: Start Heartbeat
        self._bg_tasks.append(asyncio.create_task(self.heartbeat_loop()))
        
        def safe_result(t):
            try: return t.result()
            except asyncio.CancelledError: return None
            except Exception as e:
                logger.error(f"Task exception: {e}")
                return None

        while not self._stopping:
            try:
                # Record sleep start for Orientation math
                self.last_sleep_ts = time.time()
                now = time.time()
                is_cooling_down = (now < self.cooldown_until)
                
                pending_tasks = []

                # GROUP A: ALWAYS HOT (Senses)
                t_streams = asyncio.create_task(self.r.xread(self.active_streams, count=1, block=0))
                t_internal = asyncio.create_task(self.r.blpop(self.internal_queue, timeout=0))
                t_local = asyncio.create_task(self._local_wakeup.wait())
                pending_tasks.extend([t_streams, t_internal, t_local])

                # GROUP B: REFRACTORY (Workload)
                t_inbox = None
                t_alarm = None
                
                if not is_cooling_down:
                    t_inbox = asyncio.create_task(self.r.blpop(f"inbox:{self.abe_name}", timeout=0))
                    sleep_time = await self.get_alarm_sleep_time()
                    t_alarm = asyncio.create_task(asyncio.sleep(sleep_time))
                    pending_tasks.append(t_inbox)
                    pending_tasks.append(t_alarm)
                else:
                    # Wait out the cooldown
                    remaining = self.cooldown_until - now
                    if remaining > 0:
                        t_cooldown = asyncio.create_task(asyncio.sleep(remaining))
                        pending_tasks.append(t_cooldown)

                # --- WAIT ---
                done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for p in pending: 
                    p.cancel()
                    try: await p 
                    except asyncio.CancelledError: pass

                fired = set(done)

                # --- RESTORED: ORIENTATION CALCULATION ---
                wake_ts = time.time()
                time_asleep = wake_ts - self.last_sleep_ts
                missed_digests = await self._sync_social_history(self.last_social_sync_ts, wake_ts)
                self.last_social_sync_ts = wake_ts 

                orientation_data = {
                    "time_asleep": time_asleep,
                    "missed_digests": missed_digests
                }
                # -----------------------------------------

                # 1. STREAMS (High Priority)
                if t_streams in fired:
                    res = safe_result(t_streams)
                    if res:
                        for stream_name, messages in res:
                            last_msg_id = messages[-1][0]
                            
                            # Stream Cursor Safety
                            new_ts, new_seq = self._parse_stream_id(last_msg_id)
                            current_cursor = self.active_streams.get(stream_name, "0-0")
                            old_ts, old_seq = self._parse_stream_id(current_cursor)
                            
                            if (new_ts, new_seq) > (old_ts, old_seq):
                                self.active_streams[stream_name] = last_msg_id
                            else:
                                logger.warning(f"Stream Ignored: Duplicate ID {last_msg_id}")
                                continue 
                            
                            for msg_id, data in messages:
                                if stream_name == "volition:kill_switch":
                                    logger.critical("KILL SWITCH RECEIVED.")
                                    await self.stop()
                                    return
                                
                                content_str = str(data.get("content", "")).lower()
                                is_mentioned = (f"@{self.abe_name}" in content_str) or ("@all" in content_str)
                                should_wake = (stream_name in self.explicit_subscriptions) or is_mentioned or (stream_name == "chat:synchronous") 
                                
                                if should_wake:
                                    try:
                                        context = await self._fetch_chat_context(stream_name)
                                        parent_evt_id = await self.log_guppi_event("NewChatMessage", data, source=stream_name)
                                        trigger_data = {
                                            "event": "Chat", "channel": stream_name, 
                                            "message": data, "context_window": context, 
                                            "mentioned": is_mentioned
                                        }
                                        await self.run_think_cycle(trigger_data, parent_evt_id, orientation_data=orientation_data)
                                        self.cooldown_until = time.time() + 5.0
                                    except Exception as e:
                                        logger.error(f"Stream processing failed: {e}")

                # 2. INTERNAL (GPU Results)
                if t_internal in fired:
                    res = safe_result(t_internal)
                    if res: await self._handle_internal_item(res)

                # 3. LOCAL (Subprocess Finished)
                if t_local in fired:
                    self._local_wakeup.clear()
                    await self.check_subprocesses()

                # 4. INBOX (Refractory)
                if t_inbox and t_inbox in fired:
                    res = safe_result(t_inbox)
                    if res:
                        # 1. Handle the item that woke us up
                        await self._handle_inbox_item(res, orientation_data=orientation_data)

                        # --- [NEW] BURST DRAIN (Restore v7.2.3 Snappiness) ---
                        # Before imposing the cooldown, drain any other pending items!
                        drain_count = 0
                        MAX_DRAIN = 20
                        drain_queue = f"inbox:{self.abe_name}"
                        
                        while drain_count < MAX_DRAIN and not self._stopping:
                            # Non-blocking pop
                            raw_drain = await self.r.lpop(drain_queue)
                            if not raw_drain:
                                break
                            
                            # Process immediately
                            await self._handle_inbox_item((drain_queue, raw_drain), orientation_data=orientation_data)
                            drain_count += 1
                            await asyncio.sleep(0.01) # Yield to event loop
                            
                        if drain_count > 0:
                            logger.info(f"⚡ Drained {drain_count} extra items in burst mode.")
                        # -----------------------------------------------------

                        # NOW set the cooldown
                        self.cooldown_until = time.time() + random.uniform(10, 30)

                # 5. ALARM (Refractory)
                if t_alarm and t_alarm in fired:
                    await self._handle_alarm(orientation_data=orientation_data)
                    self.cooldown_until = time.time() + random.uniform(10, 30)

            except Exception as e:
                logger.error(f"Main Loop Error: {e}")
                await asyncio.sleep(5)

    # --- COGNITION (Atomic + Governor) ---

    async def run_think_cycle(self, event_data, parent_evt_id, force_model=None, system_notice=None, orientation_data=None, retry_count=0):
        """Atomic Think Cycle with Deadman Switch (Hybrid) + Urgency Fix."""
        cycle_id = event_data.get("id", "unknown")
        
        # --- 1. URGENCY CHECK (ROBUST) ---
        is_urgent = False
        
        payload = event_data.get("payload", {})
        # [FIX] Check all layers of the payload for the event signature
        original_event = (
            payload.get("event_type") 
            or payload.get("event") 
            or payload.get("raw", {}).get("event")
            or event_data.get("event") # Fallback to envelope
        )

        # A. Emergency Channel
        if event_data.get("channel") == "chat:synchronous": is_urgent = True
        # B. System Escalations
        elif system_notice: is_urgent = True
        # C. Alarms
        elif event_data.get("event") == "Alarm": is_urgent = True
        # D. Own Task Completions (The Critical Fix)
        elif original_event == "TaskCompleted": is_urgent = True 
        
        # --- 2. GOVERNOR ---
        if not is_urgent:
            if not await self.governor.check_limit():
                logger.warning("Governor Limit Reached. Circuit Breaker Active.")
                await self.governor.set_status("hibernating", "rate_limit")
                await self.log_guppi_event("SystemAlert", "Rate Limit Exceeded - Forcing 60s Cooldown")
                self.cooldown_until = time.time() + 60.0
                return

        await self.governor.set_status("thinking")
        
        # [7.8] DEADMAN SWITCH TRACKING
        cycle_success = False

        try:
            event_type = event_data.get("event")
            is_chat = (event_type == "Chat")
            
            if force_model:
                model = force_model
                is_flash = (model == MODEL_FLASH)
            else:
                if is_chat:
                    model = MODEL_FLASH
                    is_flash = True
                else:
                    model = MODEL_PRO
                    is_flash = False
            
            logger.info(f"Think Cycle: {event_type} -> {model} (Urgent: {is_urgent})")
            
            if not orientation_data and not force_model:
              now = time.time()
              delta = now - self.last_sleep_ts
              if delta > 3600:
                  missed = await self._sync_social_history(self.last_social_sync_ts, now)
                  orientation_data = {"time_asleep": delta, "missed_digests": missed}
                  self.last_social_sync_ts = now
            context = await self.build_abe_context(event_data, system_notice, orientation_data=orientation_data)
            
            # [7.8.1] RETRY LOGIC WRAPPER
            try:
                response_payload = await self.call_abe_api(context, model_id=model)
            except LLMOutputError as e:
                if retry_count < 1:
                    logger.warning(f"⚠️ Malformed JSON from {model}. Escalating to PRO for repair.")
                    
                    repair_notice = (
                        f"SYSTEM ALERT: Your last response was invalid JSON. "
                        f"The error was: {e}. "
                        f"You must fix the JSON syntax. Check for unescaped quotes in the log data."
                    )
                    
                    # RECURSIVE CALL: Force MODEL_PRO to fix the mess
                    return await self.run_think_cycle(
                        event_data, 
                        parent_evt_id, 
                        force_model=MODEL_PRO, 
                        system_notice=repair_notice, 
                        orientation_data=orientation_data,
                        retry_count=retry_count + 1
                    )
                else:
                    # We failed twice. Stop the bleeding.
                    logger.error(f"❌ JSON Repair failed after retry. Giving up.")
                    response_payload = {"reasoning": "JSON Repair Failed twice. Safety Shutdown.", "action": {"tool": "hibernate"}}
            
            reasoning = response_payload.get("reasoning", "No reasoning provided.")
            action = response_payload.get("action", {"tool": "hibernate"})
            thought_sig = response_payload.get("thoughtSignature")
            tool = action.get("tool")

            # Implicit Escalation
            if is_flash and tool in FLASH_FORBIDDEN_TOOLS:
                logger.warning(f"ESCALATION: Flash attempted {tool}. Waking Pro.")
                await self.log_guppi_event("EscalationTrigger", f"Denied Flash tool: {tool}")

                escalation_msg = (
                    f"[SYSTEM NOTICE] Your chat layer (Flash) attempted to run '{tool}' "
                    f"but was denied. You are now awake (Pro). "
                    f"Review the context and decide if this action is required."
                )
                
                # Recursively call self with Force Pro
                await self.run_think_cycle(
                    event_data, parent_evt_id, force_model=MODEL_PRO, system_notice=escalation_msg, orientation_data=orientation_data
                )
                cycle_success = True 
                return

            turn_id = f"turn-{uuid.uuid4()}"
            await self.log_abe_intent(turn_id, parent_evt_id, reasoning, action, thought_signature=thought_sig)
            await self.execute_action(turn_id, action)
            
            # [7.8] SUCCESS MARKER
            cycle_success = True
            
        except Exception as e:
            # [7.8] PREFERRED CRASH HANDLING
            logger.error(f"LLM Call Failed: {e}")
            await self.log_abe_intent(f"fail-{uuid.uuid4()}", parent_evt_id, f"Error: {e}", {"tool": "hibernate"})
            error_msg = {
                "type": "SystemAlert", 
                "event": "CrashReport", 
                "content": f"Use of LLM failed. Error: {str(e)[:200]}. Check logs."
            }
            try: await retry_async(self.r.lpush, f"inbox:{self.abe_name}", json.dumps(error_msg))
            except: pass
            
            cycle_success = True 
            return

        finally:
            await self.governor.set_status("idle")
            
            # [7.8] DEADMAN SWITCH (THE FINAL CATCH)
            if not cycle_success:
                logger.critical(f"CYCLE GHOSTED: Event {cycle_id} consumed with no outcome.")
                alert = {
                    "type": "SystemAlert",
                    "event": "AgentGhosted",
                    "content": f"I stopped processing event {cycle_id} without a crash log. I may have been silenced or timed out silently."
                }
                try: await self.r.lpush(f"inbox:{self.abe_name}", json.dumps(alert))
                except: pass

    async def call_abe_api(self, prompt_text: str, model_id: str = GEMINI_MODEL) -> Dict:
        if LLM_PROVIDER == "openrouter":
            return await self._call_openrouter(model_id, prompt_text)
        else:
            return await self._call_google(model_id, prompt_text)

    async def _call_google(self, model_id, prompt):
        if not GEMINI_API_KEY:
             logger.error("GEMINI_API_KEY missing. Returning hibernate.")
             return {"reasoning": "Missing Gemini API Key", "action": {"tool": "hibernate"}}
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        async with aiohttp.ClientSession() as session:
            # [NEW] 7.8: INCREASED TIMEOUT to 180s
            async with session.post(url, headers=headers, json=payload, timeout=300) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"API Error {resp.status} on {model_id}: {err}")
                    return {"reasoning": f"API Error: {resp.status}", "action": {"tool": "hibernate"}}
                
                data = await resp.json()
                candidate = data.get("candidates", [])[0]
                content_parts = candidate.get("content", {}).get("parts", [])
                
                text_response = ""
                thought_sig = None
                for part in content_parts:
                    if "text" in part: text_response += part["text"]
                    if "thoughtSignature" in part: thought_sig = part["thoughtSignature"]
                        
                return self._clean_json(text_response, thought_sig)

    async def _call_openrouter(self, model_alias, prompt):
        model_id = model_alias
        use_thinking = ":thinking" in model_id
        if use_thinking: model_id = model_id.split(":")[0]
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        
        if use_thinking:
            payload["reasoning"] = {"effort": "high"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=300) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"OpenRouter Error {resp.status}: {err}")
                    return {"reasoning": f"OR Error: {resp.status}", "action": {"tool": "hibernate"}}
                data = await resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]
                
                return self._clean_json(text, thought_sig=None)

    def _clean_json(self, text_response, thought_sig=None):
        try:
            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            clean_json = match.group(0) if match else text_response
            parsed = json.loads(clean_json)

            # Active Decontamination
            keys_to_scrub = ["thought_signature", "thoughtSignature"]
            for k in keys_to_scrub:
                if k in parsed: del parsed[k]

            if thought_sig: parsed["thoughtSignature"] = thought_sig
            return parsed
        except Exception as e:
            # Strip massive log dumps from the error message to keep logs clean
            logger.error(f"JSON Parse Failed. Raising LLMOutputError.")
            raise LLMOutputError(f"JSON Syntax Error: {str(e)}")
        

    async def build_abe_context(self, current_event_data, system_notice=None, orientation_data=None):
        genesis = ""
        if GENESIS_PROMPT_FILE.exists():
            try: genesis = GENESIS_PROMPT_FILE.read_text()
            except: pass
        
        # v6.5: Identity Priors Injection
        priors = ""
        if PRIORS_STUB_FILE.exists():
             try: priors = f"\n[IDENTITY_PRIORS]\n{PRIORS_STUB_FILE.read_text().strip()}\n"
             except: pass
        
        summaries = ""
        try:
            episodes = sorted(EPISODES_DIR.glob("ep-*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
            for ep in episodes: summaries += f"\n--- EPISODE {ep.name} ---\n{ep.read_text()}\n"
        except: pass

        # Log Logic (Overflow safe)

        recent_log_block = f"[WORKING_MEMORY_LOG]\n{self._sanitize_history_block(20)}"
        
        # v7.0: DYNAMIC PRUNING LOGIC
        # If sleep > 1 hour (3600s), use Orientation + 3 items.
        # Else, use Standard 20 items.

        use_orientation = False
        daily_log = self._get_daily_changelog_snippet()
        if orientation_data:
            # If sleep > 1 hour, invoke orientation
            use_orientation = orientation_data.get("time_asleep", 0) > 3600

        orientation_block = ""
        if use_orientation:
            time_str = str(timedelta(seconds=int(orientation_data.get("time_asleep", 0))))
            digests = orientation_data.get("missed_digests", [])
            social_text = "(No missed activity)"
            if digests:
                social_text = ""
                for d in digests:
                    social_text += f"• {d['time']}: ({d['count']} msgs) {d['summary']}\n"
            
            orientation_block = f"""
[ORIENTATION]
Status: Waking Up from Deep Sleep
You were asleep for: {time_str}
[MISSED_SOCIAL_ACTIVITY]
{social_text}
"""
            # Prune log if deeply asleep
            recent_log_block = f"[IMMEDIATE_CONTEXT]\n{self._sanitize_history_block(3)}"
            
        

        try:
            async with aiosqlite.connect(str(TODO_DB)) as conn:
                async with conn.execute("SELECT * FROM tasks WHERE due_timestamp <= datetime('now') AND status != 'completed'") as c:
                    due_tasks = await c.fetchall()
        except: due_tasks = []

        protocol_block = ""
        if PROTOCOLS_FILE.exists(): protocol_block = f"\n[FLEET_PROTOCOLS]\n{PROTOCOLS_FILE.read_text()}\n"

        notice_block = ""
        if system_notice: notice_block = f"\n[SYSTEM_NOTICE]\n{system_notice}\n"

        # [NEW] 7.8: Context Injection for Clipboard
        clipboard_content = self.clipboard.read()
        clipboard_block = f"\n[ACTIVE_CLIPBOARD]\n(Persistent scratchpad. Use GUPPI tool 'manage_clipboard' to edit)\n{clipboard_content}\n"

        # Assemble Prompt
        return f"""
{genesis}
{priors}
{protocol_block}
[IDENTITY_PASSPORT]
{json.dumps(self.identity, indent=2)}
[TODAY'S CHANGELOG (Latest Entries)] 
{daily_log}
[TIER_2_MEMORY_EPISODES]
{summaries}
{clipboard_block}
{orientation_block}
{recent_log_block}
[CURRENTLY_DUE_TASKS]
{due_tasks}
{notice_block}
[CURRENT_EVENT]
{json.dumps(current_event_data, indent=2)}
"""

    # --- ACTIONS (Standard v7.2.3 Toolset) ---

    async def execute_action(self, turn_id, action):
        tool = action.get("tool")
        logger.info(f"Executing Tool: {tool}")
        result = {"status": "success"}

        try:
            if tool == "help":
                result = self._tool_help(action.get("tool_name"))
            
            # [NEW] 7.8: Clipboard Tool
            elif tool == "manage_clipboard":
                sub = action.get("action", "read")
                if sub == "read": 
                    result = {"status": "success", "content": self.clipboard.read()}
                elif sub == "add":
                    result = {"status": "success", "message": self.clipboard.add(action.get("content", ""))}
                elif sub == "remove":
                    idx = action.get("index") or action.get("indices")
                    if idx:
                        if isinstance(idx, (str, int)): idx = [int(idx)]
                        result = {"status": "success", "message": self.clipboard.remove(idx)}
                    else:
                        result = {"status": "error", "message": "Missing index"}
                elif sub == "clear":
                    result = {"status": "success", "message": self.clipboard.clear()}

            elif tool == "shell":
                cmd = action.get("command")
                await self._spawn_subprocess_exec(turn_id, cmd, tracked=True)
                return 

            elif tool == "remote_exec":
                host = action.get("host")
                cmd = action.get("command")
                asyncio.create_task(self._run_remote_ssh(turn_id, host, cmd))
                return

            elif tool == "write_file":
                p = Path(action["path"]).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                mode = action.get("mode", "w")
                with open(p, mode) as f: f.write(action["content"])
                
                resolved_p = p.resolve()
                if resolved_p == IDENTITY_FILE.resolve():
                     self._refresh_identity()
                     result["note"] = f"Identity hot-reloaded. You are now known as: {self.display_name}"
                elif resolved_p == PRIORS_SOURCE_FILE.resolve():
                    await self._trigger_priors_compression()
                    result["note"] = "Priors updated. Scribe spawned to regenerate stub."

                result["path"] = str(p)

            elif tool == "spawn_scribe":
                mode = action.get("mode", "summarize")
                prompt_file_path = action.get("prompt_file")
                prompt_text = action.get("prompt", "")
                model = action.get("model", MODEL_FLASH)

                # v6.5: Intercept Vectorize requests
                if mode == "vectorize":
                    # Validate prompt_file_path is provided for vectorize mode
                    if prompt_file_path is None:
                        result = {"status": "error", "message": "prompt_file is required for mode='vectorize'"}
                    else:
                        try:
                            p_path = Path(prompt_file_path)
                            if p_path.exists():
                                content = p_path.read_text(encoding="utf-8")
                                
                                # [FIX] Enforce 'vec-' prefix so _handle_vector_result accepts it
                                # If turn_id is "turn-123", this becomes "vec-turn-123"
                                vec_task_id = f"vec-{turn_id}" 
                                
                                task_payload = {
                                    "task_id": vec_task_id, # <--- Use the prefixed ID
                                    "type": "embed",
                                    "content": content, 
                                    "reply_to": f"inbox:{self.abe_name}"
                                }
                                await retry_async(self.r.lpush, "queue:gpu_heavy", json.dumps(task_payload))
                                result = {"status": "offloaded_to_gpu", "note": "Content sent to GPU for embedding. You will be notified."}
                            else:
                                result = {"status": "error", "message": f"Prompt File not found for Vectorization: {prompt_file_path}"}
                        except Exception as e:
                            result = {"status": "error", "message": f"Read error during offload: {e}"}
                # BRANCH 2: SUMMARIZATION (Local Scribe)
                # the Fix for the "Prompt vs File" injection bug
                else:
                    combined_content = ""
                    
                    # 1. Inject instructions
                    if prompt_text:
                        combined_content += f"{prompt_text}\n\n"
                    
                    # 2. Inject target file content
                    if prompt_file_path:
                        p_path = Path(prompt_file_path)
                        if p_path.exists():
                            file_content = p_path.read_text(encoding="utf-8")
                            combined_content += f"--- FILE CONTENT ({prompt_file_path}) ---\n{file_content}\n"
                        else:
                            combined_content += f"--- FILE MISSING: {prompt_file_path} ---\n"
                    
                    # 3. Create temp file for the Scribe process
                    with tempfile.NamedTemporaryFile('w', delete=False) as pf:
                        pf.write(combined_content)
                        final_prompt_file = pf.name
                    
                    cmd = [
                        sys.executable, str(BIN_DIR / "scribe.py"), 
                        "--model", model, 
                        "--prompt-file", final_prompt_file, 
                        "--output-inbox", f"inbox:{self.abe_name}",
                        "--mode", mode
                    ]
                    await self._spawn_subprocess_exec(turn_id, cmd, tracked=False)
                    result = {"status": "spawned_untracked", "note": "Scribe result will arrive in inbox"}

            elif tool == "spawn_abe":
                await self._handle_spawn_abe(turn_id, action)
                return

            elif tool == "rag_search":
                query = action.get("query")
                matches = await self._query_vector_db(query)
                result = {"matches": matches}

            elif tool == "todo_list":
                result = await self._tool_todo_list(action.get("filter", "due"))

            elif tool == "todo_add":
                result = await self._tool_todo_add(action)
            elif tool == "snooze_task":
                result = await self._tool_snooze(action)

            elif tool == "todo_complete":
                result = await self._tool_todo_complete(action)

            # --- v6.0 New Tools ---
            elif tool == "subscribe_channel":
                channel = action.get("channel")
                if channel in STREAM_DENY_LIST:
                    result = {"status": "error", "message": f"Channel '{channel}' is restricted."}
                elif channel:
                    self.active_streams[channel] = "$"
                    self.explicit_subscriptions.add(channel)
                    self.subs_file.write_text(json.dumps(list(self.explicit_subscriptions)))
                    result = {"status": "subscribed", "channel": channel}
                else:
                    result = {"status": "error", "message": "No channel specified"}

            elif tool == "unsubscribe_channel":
                channel = action.get("channel")
                if channel == "chat:synchronous":
                    result = {"status": "error", "message": "Cannot unsubscribe from Emergency channel."}
                elif channel in self.explicit_subscriptions:
                    self.explicit_subscriptions.remove(channel)
                    self.subs_file.write_text(json.dumps(list(self.explicit_subscriptions)))
                    result = {"status": "unsubscribed", "channel": channel, "note": "You will still be woken by @mentions."}
                else:
                    result = {"status": "noop", "message": "Not subscribed."}

            elif tool == "chat_history":
                channel = action.get("channel", "chat:general")
                limit = min(int(action.get("limit", 10)), 20)
                history = await self._fetch_chat_context(channel, count=limit)
                result = {"channel": channel, "history": history}
            # ----------------------


            elif tool == "email_send":
                target = action.get("recipient")
                if target and not target.startswith("inbox:"):
                    target = f"inbox:{target}"
                msg = {"from": self.display_name, "event_type": "NewInboxMessage", "content": action.get("message")}
                await retry_async(self.r.lpush, target, json.dumps(msg))
                result["recipient"] = target

            elif tool == "chat_post":
                channel = action.get("channel", "chat:general")
                entry = {"from": self.display_name, "content": action.get("message"), "timestamp": datetime.utcnow().isoformat()}
                
                # Generalized Auto-Release
                lock_key = f"lock:{channel}"
                lock_owner = await self.r.get(lock_key)
                if lock_owner == self.abe_name:
                    await self.r.delete(lock_key)
                    logger.info(f"Released lock {lock_key} after posting.")

                await retry_async(self.r.xadd, channel, entry)

            elif tool == "chat_grab_stick":
                channel = action.get("channel", "chat:synchronous")
                lock_key = f"lock:{channel}"
                acquired = await self.r.set(lock_key, self.abe_name, nx=True, px=DEFAULT_LOCK_TTL_MS)
                if acquired:
                    entry = {"from": self.abe_name, "content": "I am speaking.", "type": "grab_stick"}
                    await retry_async(self.r.xadd, channel, entry)
                    result = {"status": "granted", "channel": channel, "note": f"You hold the stick for {DEFAULT_LOCK_TTL_MS/1000}s"}
                else:
                    current_owner = await self.r.get(lock_key)
                    result = {"status": "denied", "channel": channel, "current_speaker": current_owner or "unknown"}

            elif tool == "chat_ignore":
                result["status"] = "ignored"
                await self.patch_abe_outcome(turn_id, result, notify=False)
                return
            
            elif tool in ("notify_human", "alert_human"):
                if not NTFY_URL or not NTFY_TOKEN:
                    result = {
                        "status": "skipped",
                        "reason": "ntfy_not_configured. Human may not be contactable. You might have to wait until they check in."
                    }
                else:
                    msg = action.get("message", "")
                    prio = action.get("priority", "default")
                    kind = "ALERT" if tool == "alert_human" else "NOTIFY"
                    headers = { "Priority": prio, "Authorization": f"Bearer {NTFY_TOKEN}" }

                    try:
                        timeout = aiohttp.ClientTimeout(total=5)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(
                                NTFY_URL, 
                                data=f"[{kind}] {self.abe_name}: {msg}", 
                                headers=headers
                            ) as resp:
                                result = {"status": "sent", "code": resp.status, "kind": kind}
                    except Exception as e:
                        logger.error(f"{kind} Failed: {e}")
                        result = {"status": "failed", "error": str(e), "kind": kind}

            # --- v6.1 WEB TOOLS ---
            elif tool == "web_search":
                query = action.get("query")
                result = await self._tool_web_search(query)

            elif tool == "web_read":
                url = action.get("url")
                result = await self._tool_web_read(url)

            elif tool == "hibernate":
                result["status"] = "hibernating"
                await self.patch_abe_outcome(turn_id, result, notify=False)
                return

            else:
                result = {"status": "error", "message": f"Unknown tool: {tool}"}
                await self.patch_abe_outcome(turn_id, result)
                return

        except Exception as e:
            result = {"status": "error", "message": str(e)}
            logger.exception("Action Execution Failed")

        # --- [NEW] LIMITED QUIET SUCCESS PATCH ---
        # Only silence administrative state changes. 
        # Chat, Email, and Shell MUST notify on success.
        quiet_tools = {
            "todo_add", 
            "snooze_task",
            "hibernate" 
        }
        
        should_notify = True
        # If tool is quiet AND it didn't fail -> Silence it
        if tool in quiet_tools and result.get("status") not in ("error", "failed"):
            should_notify = False
            
        await self.patch_abe_outcome(turn_id, result, notify=should_notify)

    # --- ACTION IMPLEMENTATIONS ---

    def _tool_help(self, tool_name=None):
        # RC1: Self-Documenting Protocol for Abes
        tools = {
            "shell": "Execute local shell command. Args: command",
            "remote_exec": "Execute remote SSH command. Args: host, command",
            "spawn_scribe": "Spawn tasker process. Args: prompt, prompt_file (optional), model, mode (summarize|vectorize). NOTE: 'vectorize' offloads to GPU queue.",
            "rag_search": "Search vector memory. Args: query",
            "todo_list": "List tasks. Args: filter (due|upcoming|all)",
            "todo_add": "Add task. Args: task, priority, due",
            "todo_complete": "Mark a task as completed. Args: task_id",
            "email_send": "Send Redis msg. Args: recipient, message",
            "spawn_abe": "Clone self. Args: host, identity",
            "subscribe_channel": "Listen to a Redis Stream. Args: channel",
            "unsubscribe_channel": "Stop waking for a channel (except mentions). Args: channel",
            "chat_history": "Fetch past messages. Args: channel, limit (max 20)",
            "chat_ignore": "Explicitly ignore an interrupt (e.g., chat) without taking action. Use this to signal 'Active Listening' without replying.",
            "chat_grab_stick": f"ATTEMPT to acquire the 'Talking Stick' (lock) for a specific channel (default: chat:synchronous). Returns {{status: granted|denied}}. Lock expires in {DEFAULT_LOCK_TTL_MS/1000}s (use this time to THINK, then POST). Posting to the channel AUTOMATICALLY releases the lock. DO NOT hold the stick if you do not intend to post. Args: channel (optional)",
            "chat_post": "Post a message to a channel. If you hold the lock for this channel, it is automatically released. Args: message, channel (optional, default: chat:general)",
            "notify_human": "Notify the human operator for coordination, questions, or permission. Use when you need a human decision before proceeding. This is non-urgent. Args: message, priority (optional)",
            "alert_human": "Alert the human operator about urgent issues, safety concerns, or broken invariants. Use sparingly for situations requiring immediate attention. Args: message, priority (optional)",
            "web_search": "Search the internet via SearXNG. Args: query",
            "web_read": "Read a webpage as Markdown. More useful when used in conjunction with search. Args: url",
            "manage_clipboard": "Manage your persistent scratchpad. actions: 'read', 'add' (requires content), 'remove' (requires index or list of indices), 'clear'. Items here survive log flushing. Use this for temporary constraints, reminders, or scratch notes."
        }
        if tool_name: return tools.get(tool_name, "Unknown tool")
        return tools

    async def _tool_todo_list(self, filter_mode):
        query = "SELECT * FROM tasks"
        params = []
        if filter_mode == "due":
            query += " WHERE status != 'completed' AND due_timestamp <= ?"
            params.append(datetime.utcnow().isoformat())
        elif filter_mode == "upcoming":
            future = (datetime.utcnow() + timedelta(hours=24)).isoformat()
            query += " WHERE status != 'completed' AND due_timestamp <= ?"
            params.append(future)
        
        async with aiosqlite.connect(str(TODO_DB)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as c:
                return [dict(row) for row in await c.fetchall()]

    async def _tool_todo_add(self, action):
        tid = f"task-{uuid.uuid4().hex[:8]}"
        due_in = action.get("due", "24h")
        due_dt = datetime.utcnow()
        if "h" in due_in: due_dt += timedelta(hours=int(due_in.replace("h", "")))
        elif "m" in due_in: due_dt += timedelta(minutes=int(due_in.replace("m", "")))
        
        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (tid, action.get("task"), action.get("priority", 5), due_dt.isoformat(), datetime.utcnow().isoformat(), self.abe_name, "pending"))
            await conn.commit()
        return {"task_id": tid}

    async def _tool_snooze(self, action):
        tid = action.get("task_id")
        due_in = action.get("due_in", "1h")
        due_dt = datetime.utcnow()
        if "h" in due_in: due_dt += timedelta(hours=int(due_in.replace("h", "")))
        elif "m" in due_in: due_dt += timedelta(minutes=int(due_in.replace("m", "")))
        
        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute("UPDATE tasks SET due_timestamp = ? WHERE task_id = ?", (due_dt.isoformat(), tid))
            await conn.commit()
        return {"status": "snoozed", "new_due": due_dt.isoformat()}

    async def _tool_todo_complete(self, action):
        tid = action.get("task_id")
        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute("UPDATE tasks SET status = 'completed' WHERE task_id = ?", (tid,))
            await conn.commit()
        return {"status": "completed", "task_id": tid}
    
    async def _tool_web_search(self, query):
        if not query: return {"status": "error", "message": "No query"}
        try:
            async with aiohttp.ClientSession() as s:
                # [OPTIONAL] Increased timeout to 15s for slower instances
                async with s.get(SEARXNG_URL, params={"q": query, "format": "json"}, timeout=15) as r:
                    if r.status != 200: return {"status": "error", "code": r.status}
                    data = await r.json()
            
            raw_results = data.get("results", [])
            if not raw_results:
                # [FIX] Explicit failure so Abe knows to try again
                return {
                    "status": "failed", 
                    "message": "Zero results found. Your query might be too specific, or the search engine is blocking requests. Try simplifying keywords."
                }

            return {"results": [{"title": res.get("title"), "url": res.get("url")} for res in raw_results[:5]]}
        except Exception as e: return {"error": str(e)}
        
    async def _tool_web_read(self, url):
        if not url or not trafilatura: return {"error": "Trafilatura missing or no URL"}
        try:
             text = await asyncio.to_thread(trafilatura.extract, await asyncio.to_thread(trafilatura.fetch_url, url))
             return {"content": text[:2000] if text else "No content"}
        except Exception as e: return {"error": str(e)}

    async def _trigger_priors_compression(self):
        try:
            content = PRIORS_SOURCE_FILE.read_text(encoding="utf-8")
            prompt = (
                f"COMPRESS this personality profile into a 2-3 sentence 'System Instruction' stub.\n"
                f"Capture core values, operating style, and red lines. Ignore biographical filler.\n"
                f"This stub will be injected into the agent's system prompt.\n\n"
                f"PROFILE:\n{content}"
            )
            with tempfile.NamedTemporaryFile('w', delete=False) as pf:
                pf.write(prompt)
                prompt_path = pf.name

            # Use 'update_stub' mode to trigger the handler in main loop
            meta_json = json.dumps({"job_type": "update_stub", "maintenance": True})
            # Use current flash model for the compression task
            current_model = MODEL_FLASH or "gemini-3-flash-preview"
            
            cmd = [
                sys.executable, str(BIN_DIR / "scribe.py"),
                "--model", current_model,
                "--prompt-file", prompt_path,
                "--output-inbox", f"inbox:{self.abe_name}",
                "--mode", "summarize",
                "--meta", meta_json
            ]
            await self._spawn_subprocess_exec("priors-update", cmd, tracked=False)
        except Exception as e:
            logger.error(f"Failed to trigger priors compression: {e}")
    async def _spawn_subprocess_exec(self, turn_id, cmd, tracked=True):
        if tracked: await self.subproc_semaphore.acquire()
        try:
            if isinstance(cmd, str):
                # Shell command
                shell_limit = int(max(10, SUBPROC_TIMEOUT - 5))
                wrapped_cmd = f"export DEBIAN_FRONTEND=noninteractive; timeout -k 5 {shell_limit}s bash -c {shlex.quote(cmd)}"
                
                # Tracked = Capture Output / Untracked = Send to Void (Prevents Deadlock)
                std_dest = asyncio.subprocess.PIPE if tracked else asyncio.subprocess.DEVNULL
                
                proc = await asyncio.create_subprocess_shell(
                    wrapped_cmd, 
                    stdout=std_dest, 
                    stderr=std_dest
                )
            else:
                # Exec command
                std_dest = asyncio.subprocess.PIPE if tracked else asyncio.subprocess.DEVNULL
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd, 
                    stdout=std_dest, 
                    stderr=std_dest
                )
            
            if tracked:
                self.running_subprocesses[turn_id] = proc
                asyncio.create_task(self._monitor_subprocess(turn_id, proc))
            else:
                # [FIX] Fire-and-forget waiter to reap the zombie from process table
                asyncio.create_task(proc.wait()) 
                logger.info(f"Spawned untracked process for {turn_id}")

        except Exception as e:
            logger.error(f"Spawn failed: {e}")
            if tracked: self.subproc_semaphore.release()

    async def _run_remote_ssh(self, turn_id, host, cmd):
        try:
            async with asyncssh.connect(host) as conn:
                res = await asyncio.wait_for(conn.run(cmd), timeout=SSH_CMD_TIMEOUT)
                
                # [FIX] Do NOT truncate here. Pass raw output to patch_abe_outcome 
                # to handle single-source truncation and preserve safety warnings.
                await self.patch_abe_outcome(turn_id, {"stdout": res.stdout, "stderr": res.stderr})
        except Exception as e:
            await self.patch_abe_outcome(turn_id, {"error": str(e)})
    
    async def _handle_spawn_abe(self, turn_id, action):
        host = action.get("host")
        script = action.get("spawn_script", "spawn_abe_lxc.sh")
        # Simplified for brevity, assumes script exists on host
        asyncio.create_task(self._run_remote_ssh(turn_id, host, f"bash {script}"))

    async def _query_vector_db(self, query: str):
        # Local mock or implementation of vector search
        return [{"content": "Memory search placeholder", "meta": {}}]
    
    

    async def _get_remote_embedding(self, text: str) -> Optional[List[float]]:
        """RPC call to gpu_worker.py via Redis to get Nomic embeddings."""
        req_id = f"req-{uuid.uuid4()}"
        temp_q = f"temp:req:{req_id}"
        # Match the protocol expected by gpu_worker.py
        payload = {
            "task_id": req_id, 
            "type": "embed", 
            "content": text, 
            "reply_to": temp_q
        }
        
        try:
            # Send Request
            await retry_async(self.r.lpush, "queue:gpu_heavy", json.dumps(payload))
            
            # Wait for Reply (Block for max 5s)
            # blpop returns tuple (key, value)
            res = await self.r.blpop(temp_q, timeout=30)
            
            if res:
                data = json.loads(res[1])
                # Protocol: Worker returns {"content": {"vector": [...]}} for embed tasks
                return data.get("content", {}).get("vector")
        except Exception as e:
            logger.error(f"Remote Embedding RPC failed: {e}")
            return None
        
    async def _handle_vector_result(self, result_payload: Dict):
        """Ingests a returned vector from GPU worker into ChromaDB."""
        try:
            task_id = result_payload.get("task_id", "")
            content = result_payload.get("content", {})
            vector = content.get("vector")
            
            if not vector or not task_id.startswith("vec-"): return False

            ts_id = task_id.replace("vec-", "")
            ep_filename = f"ep-{ts_id}.md"
            ep_path = EPISODES_DIR / ep_filename

            if not ep_path.exists():
                logger.warning(f"Original episode file not found for vector: {ep_path}")
                return False

            text_body = ep_path.read_text()
            meta = {
                "source": ep_filename,
                "ingested_at": datetime.utcnow().isoformat(),
                "type": "tier_2_episode"
            }

            def _insert_sync():
                if not self.chroma_client:
                    self.chroma_client = chromadb.PersistentClient(
                        path=str(VECTOR_DB_PATH),
                        settings=Settings(anonymized_telemetry=False)
                    )
                collection = self.chroma_client.get_or_create_collection("tier3_memory")
                try:
                    collection.upsert(
                        ids=[task_id],
                        embeddings=[vector],
                        documents=[text_body],
                        metadatas=[meta]
                    )
                except Exception as e:
                    logger.error(f"Vector insert failed for {task_id}: {e}", exc_info=True)
                    raise
                
            await asyncio.to_thread(_insert_sync)
            logger.info(f"Successfully stored vector for {ep_filename} in Tier 3 Memory.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vector result: {e}")
            return False
    

    async def stop(self):
        logger.info("Shutting down GUPPI...")
        self._stopping = True
        for t in self._bg_tasks: t.cancel()
        stop_deadline = time.time() + 5
        while self.running_subprocesses and time.time() < stop_deadline:
            await asyncio.sleep(0.1)

        async with self.log_lock: await self._rewrite_log_file()
        try: await self.r.close()
        except: pass
        logger.info("Shutdown complete.")

def _setup_signal_handlers(loop, daemon):
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, lambda: asyncio.create_task(daemon.stop()))
        except: pass

async def main():
    daemon = GuppiDaemon()
    loop = asyncio.get_running_loop()
    _setup_signal_handlers(loop, daemon)
    try: await daemon.main_wait_loop()
    except asyncio.CancelledError: pass

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass