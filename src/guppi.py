#!/usr/bin/env python3
"""GUPPI daemon - Volition 8.0.0-rc2 (The Roamer/Scribe Update...Fix)
Status: STABLE
- Feature: Allow Roamers and Scribes Separately
- Feature: Merge different Provider calls into one
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
FLEET_DIR = ABE_ROOT / os.environ.get("FLEET_DIR", "fleet")
SCRIPT_REGISTRY_FILE = FLEET_DIR / "script_registry.json"

# v7.2.1: Flight Recorder Log
LOGS_DIR = ABE_ROOT / "logs"
INBOX_DUMP_LOG = LOGS_DIR / "inbox_dump.jsonl"

# Network Config
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "volition")
REDIS_URL = os.environ.get("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0")
NTFY_URL = os.environ.get("NTFY_URL", "")
NTFY_TOKEN = os.environ.get("NTFY_TOKEN", "")



# v6.1: Search Config
SEARXNG_URL = os.environ.get("SEARXNG_URL", "https://civitat.es/search")

# API Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://volition.indoria.org")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Volition")

# v6.5: Split-Brain Config
# Defaulting to standard model names so they map cleanly via OpenRouter or Local
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
MODEL_PRO = os.environ.get("MODEL_PRO", "google/gemini-3-flash-preview:thinking")
MODEL_FLASH = os.environ.get("MODEL_FLASH", "google/gemini-3-flash-preview")
MODEL_SUMMARIZE = os.environ.get("MODEL_SUMMARIZE", "local/mistral")

# Qwen3.6 preserve-thinking support.
# - auto: enabled only for Qwen 3.6-family model names.
# - off: never enabled
# - on/force/always: enabled regardless of model name for local experiments
PRESERVE_THINKING_MODE = os.environ.get("GUPPI_PRESERVE_THINKING", "auto").strip().lower()

try:
    PRESERVE_THINKING_TURNS = int(os.environ.get("GUPPI_PRESERVE_THINKING_TURNS", "4"))
except (TypeError, ValueError):
    PRESERVE_THINKING_TURNS = 4
PRESERVE_THINKING_TURNS = max(0, min(PRESERVE_THINKING_TURNS, 8))


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

# Chat stream policy
# - chat:general: passive town-square; wakes on @mentions or explicit subscription.
# - chat:watercooler: moderated morning/social room; wakes all agents, but stays non-urgent.
# - chat:synchronous: emergency/moot channel; wakes all agents and bypasses governor.

DEFAULT_CHAT_STREAMS = ("chat:general", "chat:watercooler", "chat:synchronous")
WAKE_ALL_CHAT_STREAMS = {"chat:watercooler", "chat:synchronous"}
URGENT_CHAT_STREAMS = {"chat:synchronous"}
MODERATED_CHAT_STREAMS = {"chat:watercooler", "chat:synchronous"}

# Safety
STREAM_DENY_LIST = ["volition:action_log", "volition:heartbeat", "volition:log_stream"]
FLASH_FORBIDDEN_TOOLS = {
    "shell",
    "write_file",
    "spawn_abe",
    "remote_exec",
    "spawn_scribe",
    "spawn_roamer",
    "manage_clipboard",
    "manage_script_registry",
}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("guppi")

if not NTFY_URL:
    logger.warning("NTFY not configured; human notifications disabled.")

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

class ContextLengthExceededError(Exception):
    """Raised when the LLM API returns a 400 Context Length Exceeded error."""
    pass

# 8.1 : New clipboard
class Clipboard:
    """Manages the persistent scratchpad for the agent.

    Storage model:
    - One logical clipboard item per non-empty line.
    - User-facing indices are 1-based.
    - Multi-line content is normalized into multiple items.
    """
    VALID_STATUSES = {
        "IN PROGRESS",
        "DONE",
        "BLOCKED",
        "FAILED",
        "CANCELLED",
    }

    STATUS_PREFIX_RE = re.compile(
        r"^\[(?: |x|X|IN PROGRESS|DONE|BLOCKED|FAILED|CANCELLED)\]\s*"
    )

    def __init__(self, filepath: Path):
        self.path = filepath

    def _normalize_items(self, content: Any) -> List[str]:
        if content is None:
            return []
        return [line.strip() for line in str(content).splitlines() if line.strip()]

    def _read_lines(self) -> List[str]:
        if not self.path.exists():
            return []
        return self._normalize_items(self.path.read_text(encoding="utf-8"))

    def _write_lines(self, lines: List[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(self.path.parent),
            delete=False,
            encoding="utf-8",
        ) as tf:
            if lines:
                tf.write("\n".join(lines) + "\n")
            temp_path = Path(tf.name)
        os.replace(temp_path, self.path)

    def _coerce_index(self, index: Any) -> int:
        try:
            idx = int(index)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid clipboard index: {index!r}")

        if idx < 1:
            raise ValueError("Clipboard indices are 1-based; index must be >= 1.")

        return idx

    def read(self) -> str:
        lines = self._read_lines()
        if not lines:
            return "(Empty)"
        return "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])

    def add(self, content: str) -> str:
        lines = self._read_lines()
        new_items = self._normalize_items(content)

        if not new_items:
            return "No content provided."

        added = 0
        skipped = 0

        for item in new_items:
            if item in lines:
                skipped += 1
                continue
            lines.append(item)
            added += 1

        self._write_lines(lines)
        return f"Added {added} item(s). Skipped {skipped} duplicate(s)."

    def set(self, content: str) -> str:
        new_items = self._normalize_items(content)
        self._write_lines(new_items)
        return f"Clipboard set to {len(new_items)} item(s)."

    def insert(self, index: int, content: str) -> str:
        lines = self._read_lines()
        new_items = self._normalize_items(content)

        if not new_items:
            return "No content provided."

        try:
            idx = self._coerce_index(index)
        except ValueError as e:
            return str(e)
        zero_idx = min(idx - 1, len(lines))
        lines[zero_idx:zero_idx] = new_items

        self._write_lines(lines)
        return f"Inserted {len(new_items)} item(s) at index {idx}."

    def replace(self, index: int, content: str) -> str:
        lines = self._read_lines()
        new_items = self._normalize_items(content)

        if not new_items:
            return "No replacement content provided."

        try:
            idx = self._coerce_index(index)
        except ValueError as e:
            return str(e)

        zero_idx = idx - 1

        if zero_idx >= len(lines):
            return f"Index {idx} out of range. Clipboard has {len(lines)} item(s)."

        lines[zero_idx:zero_idx + 1] = new_items
        self._write_lines(lines)
        return f"Replaced item {idx} with {len(new_items)} item(s)."

    def mark(self, index: int, status: str = "DONE") -> str:
        lines = self._read_lines()
        try:
            idx = self._coerce_index(index)
        except ValueError as e:
            return str(e)

        zero_idx = idx - 1

        if zero_idx >= len(lines):
            return f"Index {idx} out of range. Clipboard has {len(lines)} item(s)."

        clean_status = str(status or "DONE").strip().upper()
        if clean_status not in self.VALID_STATUSES:
            allowed = ", ".join(sorted(self.VALID_STATUSES))
            return f"Invalid status {clean_status!r}. Allowed: {allowed}."

        old_line = lines[zero_idx]
        stripped_line = self.STATUS_PREFIX_RE.sub("", old_line).strip()
        lines[zero_idx] = f"[{clean_status}] {stripped_line}"

        self._write_lines(lines)
        return f"Marked item {idx} as [{clean_status}]."

    def remove(self, indices: List[int]) -> str:
        lines = self._read_lines()

        clean_indices = []
        for raw_idx in indices:
            try:
                clean_indices.append(self._coerce_index(raw_idx))
            except ValueError:
                continue

        clean_indices = sorted(set(clean_indices), reverse=True)
        removed_count = 0

        for idx in clean_indices:
            zero_idx = idx - 1
            if 0 <= zero_idx < len(lines):
                lines.pop(zero_idx)
                removed_count += 1

        self._write_lines(lines)
        return f"Removed {removed_count} item(s)."

    def clear(self, confirm: bool = False) -> str:
        lines = self._read_lines()

        if lines and not confirm:
            return (
                "Refusing to clear non-empty clipboard without confirm=true. "
                "Use mark/replace/remove for normal plan maintenance."
            )

        self._write_lines([])
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
        self.pending_vector_tasks = {}
        self._prune_started_at = 0.0
        self._current_prune_id = None
        self.SCRIBE_SUCCESS_EVENTS = {"TaskCompleted", "ScribeResult"} # this is what happens when code evolves more than the plandocs

        self.processed_triggers = {}
        self.processed_triggers_ttl = 90


        # Subscriptions
        self.explicit_subscriptions = set()
        #self.active_streams = {"chat:synchronous": "$", "volition:kill_switch": "$", "chat:general": "$"}
        self.active_streams = {stream: "$" for stream in DEFAULT_CHAT_STREAMS}
        self.active_streams["volition:kill_switch"] = "$"
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
        logger.info(f"GUPPI v8.0.0-rc2 Initialized for {self.abe_name}")

        # --- The Machete Helper ---
    def _looks_control_heavy_text(self, text: str, sample_size: int = 4096) -> tuple[bool, dict]:
        """Detect text that will explode when JSON-escaped, e.g. binary journals full of NULs."""
        sample = text[:sample_size]
        if not sample:
            return False, {"sample_len": 0, "nul_count": 0, "control_count": 0, "control_ratio": 0.0}

        nul_count = sample.count("\x00")
        control_count = sum(
            1 for ch in sample
            if ch != "\x00" and ord(ch) < 32 and ch not in ("\n", "\r", "\t")
        )
        control_ratio = control_count / max(len(sample), 1)

        is_bad = nul_count > 0 or (control_count >= 32 and control_ratio > 0.05)

        return is_bad, {
            "sample_len": len(sample),
            "nul_count": nul_count,
            "control_count": control_count,
            "control_ratio": round(control_ratio, 4),
        }

    def _looks_control_heavy_bytes(self, data: bytes, sample_size: int = 4096) -> tuple[bool, dict]:
        """Detect binary/control-heavy bytes before decode and JSON escaping."""
        sample = data[:sample_size]
        if not sample:
            return False, {"sample_len": 0, "nul_count": 0, "control_count": 0, "control_ratio": 0.0}

        nul_count = sample.count(b"\x00")
        control_count = sum(
            1 for b in sample
            if b != 0 and b < 32 and b not in (9, 10, 13)
        )
        control_ratio = control_count / max(len(sample), 1)

        is_bad = nul_count > 0 or (control_count >= 32 and control_ratio > 0.05)

        return is_bad, {
            "sample_len": len(sample),
            "nul_count": nul_count,
            "control_count": control_count,
            "control_ratio": round(control_ratio, 4),
        }

    def _binary_suppression_notice(self, label: str, total_len: int | None, stats: dict) -> str:
        size = f"{total_len} bytes/chars" if total_len is not None else "unknown size"
        return (
            f"[BINARY / CONTROL-HEAVY {label.upper()} SUPPRESSED BY GUPPI SAFETY]\n"
            f"Hey, this is an automated thing set up by THE Abe -- Whatever you're trying, "
            f"the output looks binary or control-heavy and would expand dangerously when JSON-escaped.\n"
            f"Size: {size}; sample={stats}\n"
            "This commonly happens when tail/cat is used on binary files like systemd .journal files. "
            "Use a decoder/filter instead: journalctl --file=..., strings, grep, tail/head on decoded text, "
            "or spawn a scribe with a specific task if you need analysis over the whole file."
        )

    def _truncate_output(self, text: str, limit: int = 20000, label: str = "output") -> str:
        """Surgical tool to prevent context flooding from massive logs and JSON-escape bombs."""
        if not isinstance(text, str):
            return text

        is_bad, stats = self._looks_control_heavy_text(text)
        if is_bad:
            return self._binary_suppression_notice(label, len(text), stats)

        if len(text) <= limit:
            return text

        cut_size = len(text) - limit
        return (
            text[:limit]
            + f"\n... [Hey this is an automated thing set up by THE Abe -- "
              f"Whatever you're trying, it was flagged because you're trying to spend more than "
              f"{limit} chars this turn. This is unadvised. Try to reduce the intake. "
              f"Original Err Message: TRUNCATED BY GUPPI SAFETY {cut_size} chars removed. "
              f"Spawn a scribe with a specific task if you want to go over the entire file, "
              f"or grep selectively (if you are certain what you're looking for) to read remainder.] ..."
        )

    def _decode_tool_output(self, raw: bytes | str | None, label: str, prepatch_cap: int = 100000) -> str:
        """
        Convert subprocess/SSH output into text safely before it ever reaches
        working.log, Redis, inbox, or json.dumps().
        """
        if raw is None:
            return ""

        if isinstance(raw, bytes):
            is_bad, stats = self._looks_control_heavy_bytes(raw)
            if is_bad:
                return self._binary_suppression_notice(label, len(raw), stats)

            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)

        is_bad, stats = self._looks_control_heavy_text(text)
        if is_bad:
            return self._binary_suppression_notice(label, len(text), stats)

        if len(text) > prepatch_cap:
            cut_size = len(text) - prepatch_cap
            return (
                text[:prepatch_cap]
                + f"\n... [Hey this is an automated thing set up by THE Abe -- "
                  f"Whatever you're trying, this tool output exceeded the pre-patch hard cap "
                  f"of {prepatch_cap} chars before it could safely enter the log/Redis path. "
                  f"Original Err Message: HARD CAP PRE-PATCH {cut_size} chars removed. "
                  f"Use grep/tail/head, decode binary formats first, or spawn a scribe with a specific task.] ..."
            )

        return text

    def _atomic_write_json(self, path: Path, data: dict):
        """Safely writes JSON to avoid corruption during simultaneous Abe updates."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tf:
            json.dump(data, tf, indent=2, sort_keys=True)
            tf.write("\n")
            temp_path = Path(tf.name)
        os.replace(temp_path, path)

    async def _monitor_subprocess(self, turn_id, proc):
        """Dedicated task to wait for a process and release semaphore."""
        try:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SUBPROC_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()

            # 8.1: Decode outputs safely with pre-patch cap to prevent memory issues and log flooding
            stdout_str = self._decode_tool_output(stdout, "stdout")
            stderr_str = self._decode_tool_output(stderr, "stderr")

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
        for d in [BIN_DIR, DOCS_DIR, EPISODES_DIR, ARCHIVE_DIR, MEMORY_DIR, DOWNLOADS_DIR, LOGS_DIR, FLEET_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not WORKING_LOG.exists(): WORKING_LOG.touch()
        if not COMM_LOG.exists(): COMM_LOG.touch()
        if not INBOX_DUMP_LOG.exists(): INBOX_DUMP_LOG.touch()
        if not SCRIPT_REGISTRY_FILE.exists(): SCRIPT_REGISTRY_FILE.write_text("{}")
        self._cleanup_overflow()

    def _init_db_sync(self):
        import sqlite3
        conn = sqlite3.connect(str(TODO_DB))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tasks
                     (task_id TEXT PRIMARY KEY, description TEXT, priority INTEGER,
                      due_timestamp TEXT, created_timestamp TEXT, source_abe TEXT, status TEXT,
                      recurrence TEXT DEFAULT '')''')
        c.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in c.fetchall()}
        if "recurrence" not in columns:
            c.execute("ALTER TABLE tasks ADD COLUMN recurrence TEXT DEFAULT ''")
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

    def _sanitize_history_block(self, limit=20, buffer_override: Optional[List[Dict]] = None):
        """
        Returns context-safe history using the Overflow Pattern.
        - Most Recent Entry: kept intact up to 50k chars (working memory).
        - History Entries: truncated to 1k chars with file pointer (long-term ref).
        """
        sanitized = []
        buffer_copy = buffer_override[-limit:] if buffer_override is not None else self.log_buffer[-limit:]
        overflow_dir = MEMORY_DIR / "overflow"
        overflow_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(buffer_copy):
            # The "Recency Rule": If it's the last item, Abe is looking at it RIGHT NOW.

            is_most_recent = (i == len(buffer_copy) - 1)
            char_limit = 50000 if is_most_recent else 1000

            new_entry = entry.copy()

            # Preserve-thinking traces can be very large and should not be
            # injected into the normal visible [WORKING_MEMORY_LOG]. Qwen3.6
            # gets them through messages[].reasoning_content instead.
            if "thought_signature" in new_entry:
                sig = new_entry.get("thought_signature")
                if isinstance(sig, str) and sig:
                    new_entry["thought_signature"] = (
                        f"[PRESERVED_THINKING_STORED: {len(sig)} chars; "
                        "omitted from visible working-memory context]"
                    )
                else:
                    new_entry.pop("thought_signature", None)

            res = new_entry.get("results")
            turn_id = new_entry.get("id", "unknown")

            # [FIX] Recursively sanitize content field (e.g., massive inbox messages)
            if entry.get("type") == "GUPPIEvent" and "content" in entry:
                if isinstance(entry["content"], str):
                    new_entry["content"] = self._truncate_output(entry["content"], limit=char_limit)


            # Helper to process text fields
            def _process_text(text, suffix=""):
                text = self._truncate_output(text, limit=char_limit, label=suffix.strip("-") or "history")
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
            meta = data.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            norm["observed"]["meta"] = meta
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
        if self._is_pruning:
            logger.debug("Prune already in flight, skipping overlapping trigger.")
            return
        self._is_pruning = True
        self._prune_started_at = time.time()
        self._current_prune_id = f"prune-{int(time.time())}"
        logger.info("[PRUNE] method _is_pruning set TRUE")
        try:
            ts = int(time.time())
            archive_path = ARCHIVE_DIR / f"log-{ts}.jsonl"
            try: shutil.copy2(WORKING_LOG, archive_path)
            except: pass
            # 8.1: fix prune slice logic
            async with self.log_lock:
                buffer_len = len(self.log_buffer)
                retained_count = 10
                # Calculate exactly how many of the oldest items to summarize (max 20)
                summarized_count = min(max(0, buffer_len - retained_count), 20)
                if summarized_count <= 0:
                    self._is_pruning = False
                    self._current_prune_id = None
                    return
                # Grab the OLDEST entries (head of the list)
                buffer_snapshot = list(self.log_buffer[:summarized_count])
            try:
                # We pass limit=summarized_count so it doesn't truncate our snapshot
                log_content = self._sanitize_history_block(limit=summarized_count, buffer_override=buffer_snapshot)
            except Exception as e:
                log_content = f"Error reading log: {e}"

            prompt = (
                f"You are a summarization engine. Read the following JSON logs of an AI agent's actions:\n\n"
                f"--- SOURCE LOG START ---\n"
                f"{log_content}\n"
                f"--- SOURCE LOG END ---\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Synthesize these logs into a Tier 2 Episode Memory.\n"
                f"2. Focus on the NARRATIVE arc of what was accomplished.\n"
                f"3. CRITICAL: If the agent was mostly asleep, idle, or performed no major actions, DO NOT invent tasks. Simply state 'No significant actions were taken.'\n\n"
                f"You MUST use this exact markdown format:\n\n"
                f"## Narrative Summary\n"
                f"(2-3 sentences max)\n\n"
                f"## Key Decisions & Outcomes\n"
                f"(Bullet points. If none, write 'None')\n\n"
                f"## Changed State / New Knowledge\n"
                f"(If none, write 'None')\n\n"
                f"## Pending / Unresolved\n"
                f"(If none, write 'None')"
            )

            with tempfile.NamedTemporaryFile('w', delete=False) as pf:
                pf.write(prompt)
                prompt_path = pf.name

            meta_json = json.dumps({
                "maintenance": True,
                "source_tier_1": f"log-{ts}.jsonl",
                "mode": "summarize",
                "is_auto_prune": True,   # <--- So I can track it (and so can the Abes later)
                "drop_count": summarized_count,
                "buffer_len_at_prune": buffer_len, # For debugging/forensics
                "prune_id": self._current_prune_id,
                "prompt_path": prompt_path
            })

            current_model = MODEL_SUMMARIZE
            target_url = os.environ.get("SUMMARIZE_API_URL", "http://127.0.0.1:8080/v1")
            cmd = [
                sys.executable, str(BIN_DIR / "scribe.py"),
                "--model", current_model,
                "--prompt-file", prompt_path,
                "--output-inbox", f"inbox:{self.abe_name}",
                "--mode", "summarize",
                "--api-url", target_url,
                "--meta", meta_json
            ]

            spawn_success = await self._spawn_subprocess_exec(f"auto-prune-{ts}", cmd, tracked=False)

            if not spawn_success:
                logger.error("Failed to spawn prune job. Releasing lock immediately.")
                self._is_pruning = False

        except Exception as e:
            logger.error(f"Failed to spawn prune job: {e}")
            self._is_pruning = False # Only reset here if the SPAWN failed
        #  8.0.0_rc2 : We hold the lock until the inbox returns. Removed finally block.


    # --- EVENT & INTENT LOGGING ---

    async def log_guppi_event(self, event_type, content, source="GUPPI") -> str:
        # [FIX] Truncate the content immediately upon entry to prevent echo bloat.
        if isinstance(content, str):
            truncated_content = self._truncate_output(content)
        elif isinstance(content, dict):
            # Shallow copy to avoid mutating original payload if it's used elsewhere
            truncated_content = content.copy()
            # Optionally recurse if you want, but top-level truncation is usually enough
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
                if isinstance(truncated_results.get(k), (str, bytes)):
                    truncated_results[k] = self._decode_tool_output(
                        truncated_results[k],
                        k,
                        prepatch_cap=MAX_OUT_LEN,
                    )
                    truncated_results[k] = self._truncate_output(
                        truncated_results[k],
                        limit=MAX_OUT_LEN,
                        label=k,
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



    async def _ingest_tier2(self, norm: Dict) -> bool:
        """v6.5: Ingests Tier 2 episodes and offloads vectorization to GPU Queue."""
        try:
            meta = norm["observed"].get("meta", {})
            content = str(norm["observed"].get("content", ""))

            # Extract the event type safely from the payload envelope
            event_type = norm["observed"].get("event_type", norm["observed"].get("event", ""))

            # Only ingest if Scribe actually succeeded (any recognized success event)
            if meta.get("mode") == "summarize" and meta.get("is_auto_prune") and content and event_type in self.SCRIBE_SUCCESS_EVENTS:
                source_file = meta.get("source_tier_1", "unknown_source.jsonl")
                summary_text = content

                # v7.2 Fix: UUIDs prevent timestamp race conditions
                file_uuid = uuid.uuid4().hex
                iso_ts = datetime.utcnow().isoformat()
                filename = f"ep-{file_uuid}.md"
                ep_path = EPISODES_DIR / filename

                if not summary_text.strip().startswith("---"):
                    # Pull the actual model name from Scribe's metadata payload
                    actual_scribe_model = meta.get("model", MODEL_FLASH)
                    header = f"---\ngenerated_at: {iso_ts}\ntype: tier_2_episode\nmodel: {actual_scribe_model}\nsource_tier_1: {source_file}\n---\n\n"
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
                return True

            elif meta.get("mode") == "summarize" and meta.get("is_auto_prune"):
                logger.warning(f"Tier 2 ingest skipped! event_type={event_type}") # Just so the Abe knows.
                return False

        except Exception as e:
            logger.error(f"Failed to ingest Tier 2: {e}")
            return False

        return False

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

                # 8.0.0_rc2 Deadlock Guard
                if self._is_pruning and (time.time() - self._prune_started_at > 1800):
                    logger.error("Auto-prune deadlocked (timeout exceeded 30m). Resetting lock.")
                    self._is_pruning = False

                # cheap check only
                if len(self.log_buffer) > 20 and not self._is_pruning:
                    logger.info(f"Entered buffer greater than 20 and not self._is_pruning block.")
                    asyncio.create_task(self._prune_logs())

            except Exception as e:
                logger.error(f"Heartbeat issue: {e}")
            await asyncio.sleep(60)




    # --- NEW TASK HANDLERS (Refractory) ---

    def _parse_due_time(self, due_in_str: str) -> datetime:
        """Parses both relative times (m, h, d) and ISO 8601 timestamps."""
        now = datetime.utcnow()
        if not due_in_str:
            return now + timedelta(hours=24) # Default fallback

        # 1. Try relative time formats first
        try:
            if due_in_str.endswith("d"):
                return now + timedelta(days=float(due_in_str.replace("d", "")))
            elif due_in_str.endswith("h"):
                return now + timedelta(hours=float(due_in_str.replace("h", "")))
            elif due_in_str.endswith("m"):
                return now + timedelta(minutes=float(due_in_str.replace("m", "")))
        except ValueError:
            pass

        # 2. Try absolute ISO timestamp parsing
        try:
            # Handle 'Z' suffix natively
            clean_str = due_in_str.replace("Z", "+00:00")
            parsed_dt = datetime.fromisoformat(clean_str)

            # Normalize to naive UTC to match SQLite format expectations
            if parsed_dt.tzinfo is not None:
                parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed_dt
        except ValueError:
            logger.warning(f"Failed to parse time '{due_in_str}', defaulting to 24h.")
            return now + timedelta(hours=24)

    async def get_alarm_sleep_time(self) -> float:
        """Calculates sleep time based on next due task."""
        try:
            # FIX 1: Use str(TODO_DB) and add timeout
            async with aiosqlite.connect(str(TODO_DB), timeout=5.0) as db:
                # OPTIONAL: Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL;")
                await db.execute("PRAGMA busy_timeout = 5000;")

                # FIX 2: Filter out garbage rows
                query = "SELECT due_timestamp FROM tasks WHERE status NOT IN ('completed', 'cancelled') AND due_timestamp IS NOT NULL AND due_timestamp != '' ORDER BY due_timestamp ASC LIMIT 1"
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
        ingest_success = await self._ingest_tier2(norm)

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
        if meta.get("maintenance") is True:
            if meta.get("is_auto_prune"):
                incoming_id = meta.get("prune_id")
                if incoming_id != getattr(self, "_current_prune_id", None):
                    logger.warning(f"Discarding stale ghost prune job ({incoming_id}). A newer job owns the lock.")
                    return # Exit without touching the lock or buffer!

            evt_type = norm["observed"].get("event_type", norm["observed"].get("event", ""))

            if evt_type == "ScribeFailed":
                logger.error(f"Maintenance Scribe Failed! Output: {str(norm['observed'].get('content'))[:200]}")
                if meta.get("is_auto_prune"):
                    self._is_pruning = False # Release lock so heartbeat tries again
                    self._current_prune_id = None
                return

            # Job finished successfully, handle auto-prune specifics
            if evt_type in self.SCRIBE_SUCCESS_EVENTS and meta.get("is_auto_prune"):
                if ingest_success:
                    # Look exactly for drop_count. No math. No fallbacks.
                    drop_count = meta.get("drop_count")

                    if isinstance(drop_count, int) and drop_count > 0:
                        logger.info(f"Tier 2 Episode generated. Dropping {drop_count} oldest entries.")
                        async with self.log_lock:
                            # Drop exactly what we summarized from the head of the list
                            self.log_buffer = self.log_buffer[drop_count:]
                            await self._rewrite_log_file()
                    else:
                        logger.error(f"CRITICAL: Missing or invalid drop_count ({drop_count}). Skipping memory prune.")
                else:
                    logger.warning("Tier 2 ingestion failed. Skipping memory prune to avoid data loss.")

                # Unconditional unlock for this job, whether ingestion worked or not
                self._is_pruning = False
                self._current_prune_id = None

            await self.log_guppi_event("MaintenanceCompleted", f"Silent Scribe: {meta}", source="GUPPI:Background")

            # --- CLEANUP TEMP FILES ---
            prompt_file = meta.get("prompt_path")
            if prompt_file and os.path.exists(prompt_file):
                try:
                    os.unlink(prompt_file)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp prompt file {prompt_file}: {e}")
            return # <--- EXIT without Thinking

        ## WE ALREADY HANDLE IT, TEMPORARILY DISABILING IT TO SEE HOW IT WORKS.

        # # B2. Scribe Failure Detection
        # event_type_b = norm["observed"].get("event_type", norm["observed"].get("event", ""))
        # if event_type_b == "ScribeFailed":
        #     content_str = str(norm["observed"].get("content", ""))
        #     source_file = meta.get("source_tier_1", "unknown")
        #     await self.log_guppi_event(
        #         "ScribeFailed",
        #         f"Scribe failed for {source_file}: {content_str[:200]}",
        #         source="GUPPI:Background"
        #     )
        #     return  # EXIT without Thinking
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
                "SELECT * FROM tasks WHERE status NOT IN ('completed', 'cancelled') AND due_timestamp <= ? ORDER BY due_timestamp ASC LIMIT 5",
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
        logger.info("Entering Main Event Loop (Volition 8.0.0-rc2)...")
        await self.governor.set_status("idle")

        # 1. RESTORED: Start Heartbeat
        self._bg_tasks.append(asyncio.create_task(self.heartbeat_loop()))

        # [8.0.3] Start Autoprune Background Task
        self._bg_tasks.append(asyncio.create_task(self._auto_prune_todo_db_loop()))

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
                                should_wake = (stream_name in self.explicit_subscriptions) or is_mentioned or (stream_name in WAKE_ALL_CHAT_STREAMS)

                                if should_wake:
                                    try:
                                        context_limit = 12 if stream_name in MODERATED_CHAT_STREAMS else 5
                                        context = await self._fetch_chat_context(stream_name, count=context_limit)
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

    def _extract_original_event(self, event_data: Any) -> Optional[str]:
        """Safely extracts the original event name from nested trigger payloads.

        Some inbox payloads carry payload.raw as a string, not a dict. This helper
        prevents AttributeError from chains like payload.get("raw", {}).get("event").
        """
        if not isinstance(event_data, dict):
            return None

        payload = event_data.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        raw_payload = payload.get("raw") or {}
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        return (
            payload.get("event_type")
            or payload.get("event")
            or raw_payload.get("event")
            or event_data.get("event")
        )

    def _model_name_blob(self, *model_names: Any) -> str:
        """Builds a loose searchable model-name blob for capability gates."""
        chunks = []
        for name in model_names:
            if not name:
                continue

            raw = str(name).strip().lower()
            if not raw:
                continue

            chunks.append(raw)
            chunks.append(
                raw.replace("/", " ")
                   .replace(":", " ")
                   .replace("_", " ")
                   .replace("-", " ")
            )

        return " ".join(chunks)

    def _is_qwen36_model(self, *model_names: Any) -> bool:
        """True for Qwen 3.6-ish model names, false for Qwen3.5/Gemma/MiMo/etc."""
        blob = self._model_name_blob(*model_names)
        if "qwen" not in blob:
            return False

        # Covers names like:
        # - local/Qwen3.6-27B:thinking
        # - Qwen3.6-27B
        # - qwen36
        # - qwen 3 6
        return any(marker in blob for marker in ("3.6", "3 6", "36"))

    def _preserve_thinking_enabled(self, *model_names: Any) -> bool:
        """Public-repo safe gate for reasoning_content history reconstruction."""
        mode = PRESERVE_THINKING_MODE

        if mode in {"0", "false", "off", "no", "disabled", "disable"}:
            return False

        if mode in {"1", "true", "on", "yes", "force", "always", "enabled", "enable"}:
            return True

        # Default: auto-enable only for Qwen3.6 family models.
        return self._is_qwen36_model(*model_names)

    def _build_preserved_thinking_messages(
        self,
        current_prompt: str,
        model_id: str,
        history_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Builds OpenAI-compatible chat messages for Qwen3.6 preserve_thinking.

        Assistant messages remain the same ReAct JSON objects GUPPI already expects,
        with native reasoning placed in reasoning_content for Qwen's chat template.
        """
        limit = PRESERVE_THINKING_TURNS if history_limit is None else int(history_limit)

        if limit <= 0 or not self._preserve_thinking_enabled(model_id):
            return [{"role": "user", "content": current_prompt}]

        preserved_turns = []
        for entry in self.log_buffer:
            if entry.get("type") != "AbeTurn":
                continue
            if entry.get("status") != "completed":
                continue
            if not isinstance(entry.get("action"), dict):
                continue

            thought_sig = entry.get("thought_signature")
            if not isinstance(thought_sig, str) or not thought_sig.strip():
                continue

            preserved_turns.append(entry)

        preserved_turns = preserved_turns[-limit:]

        messages: List[Dict[str, Any]] = []

        for entry in preserved_turns:
            messages.append({
                "role": "user",
                "content": (
                    "[PREVIOUS_GUPPI_TURN_CONTEXT]\n"
                    "Compact anchor for the following preserved assistant turn. "
                    "Do not treat this as a separate task. Tool results and event "
                    "details are provided through [WORKING_MEMORY_LOG] in the "
                    "current prompt when still hot.\n"
                    f"turn_id: {entry.get('id', 'unknown')}\n"
                    f"parent_event_id: {entry.get('parent_event_id', 'unknown')}\n"
                    f"timestamp_intent: {entry.get('timestamp_intent', '')}\n"
                    f"timestamp_outcome: {entry.get('timestamp_outcome', '')}"
                ),
            })

            assistant_json = {
                "reasoning": entry.get("reasoning", ""),
                "action": entry.get("action", {"tool": "hibernate"}),
            }

            messages.append({
                "role": "assistant",
                "content": json.dumps(assistant_json, ensure_ascii=False),
                "reasoning_content": entry["thought_signature"],
            })

        messages.append({"role": "user", "content": current_prompt})
        return messages

    # --- COGNITION (Atomic + Governor) ---

    async def run_think_cycle(self, event_data, parent_evt_id, force_model=None, system_notice=None, orientation_data=None, retry_count=0):
        """Atomic Think Cycle with Deadman Switch (Hybrid) + Urgency Fix."""
        cycle_id = event_data.get("id", "unknown")

        # --- 1. URGENCY CHECK (ROBUST) ---
        is_urgent = False

        # [FIX] Check all layers of the payload for the event signature
        original_event = self._extract_original_event(event_data)

        # A. Emergency Channel
        if event_data.get("channel") in URGENT_CHAT_STREAMS: is_urgent = True
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

            # Check if this is a direct email rather than a system inbox event.
            # Payload may be a raw string for malformed/plain inbox items, so normalize first.
            payload = event_data.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            payload_event_type = payload.get("event_type", "")
            is_human_email = (event_type == "Inbox" and payload_event_type == "NewInboxMessage")
            is_chat = (event_type == "Chat")

            if force_model is not None:
                model = force_model
                is_flash = (model == MODEL_FLASH)
                target_url = os.environ.get("FLASH_API_URL") if is_flash else os.environ.get("PRO_API_URL")
            else:
                # Route both Stream Chat and Direct Emails to Flash
                if is_chat:
                    model = MODEL_FLASH
                    is_flash = True
                    target_url = os.environ.get("FLASH_API_URL")
                else:
                    model = MODEL_PRO
                    is_flash = False
                    target_url = os.environ.get("PRO_API_URL")

            logger.info(f"Think Cycle: {event_type} -> {model} (Urgent: {is_urgent})")

            if not orientation_data and not force_model:
              now = time.time()
              delta = now - self.last_sleep_ts
              if delta > 3600:
                  missed = await self._sync_social_history(self.last_social_sync_ts, now)
                  orientation_data = {"time_asleep": delta, "missed_digests": missed}
                  self.last_social_sync_ts = now

            context = await self.build_abe_context(event_data, system_notice, orientation_data=orientation_data)
            messages = None
            if self._preserve_thinking_enabled(model):
                messages = self._build_preserved_thinking_messages(context, model)
                logger.info(
                    "Preserve-thinking enabled for %s with %d message(s)",
                    model,
                    len(messages),
                )

            # --- DEBUG DUMP ---
            dump_prompt_enabled = (
                os.environ.get("GUPPI_DUMP_PROMPT") == "1"
                or os.environ.get("GUPPI_PROMPT_DUMP") == "1"
            )

            if dump_prompt_enabled:
                dump_path = ABE_ROOT / "logs" / f"prompt_dump_{int(time.time())}.txt"
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                dump_path.write_text(context, encoding="utf-8")
                logger.warning(f"⚠️ Dumped {len(context)} char context payload to {dump_path}")

                if messages is not None:
                    msg_dump_path = ABE_ROOT / "logs" / f"prompt_messages_dump_{int(time.time())}.json"
                    msg_dump_path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
                    logger.warning(f"⚠️ Dumped {len(messages)} chat message(s) to {msg_dump_path}")
            # ---------------------------
            # [7.8.1] RETRY LOGIC WRAPPER
            try:
                response_payload = await self.call_abe_api(
                    context,
                    model_id=model,
                    api_url=target_url,
                    messages=messages,
                )
            except ContextLengthExceededError:
                logger.warning(f"Context window shattered ({model}). Engaging Panic Mode (dropping oldest memories) and retrying.")

                # Rebuild context with panic_mode=True
                context = await self.build_abe_context(event_data, system_notice, orientation_data=orientation_data, panic_mode=True)
                messages = None
                if self._preserve_thinking_enabled(model):
                    messages = self._build_preserved_thinking_messages(context, model)
                response_payload = await self.call_abe_api(
                    context,
                    model_id=model,
                    api_url=target_url,
                    messages=messages,
                )
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
                    # In future versions, I plan to add a small "JSON repair" LLM chain that will try to fix broken JSON, and only that.
                    logger.error(f"❌ JSON Repair failed after retry. Giving up.")
                    response_payload = {"reasoning": "JSON Repair Failed twice. Safety Shutdown.", "action": {"tool": "hibernate"}}

            native_reasoning = response_payload.pop("_native_reasoning_content", None)

            reasoning = response_payload.get("reasoning", "No reasoning provided.")
            action = response_payload.get("action", {"tool": "hibernate"})
            thought_sig = native_reasoning or response_payload.get("thoughtSignature")
            tool = action.get("tool")

            # Implicit Escalation
            if is_flash and tool in FLASH_FORBIDDEN_TOOLS and force_model is None:
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

        except asyncio.TimeoutError:
            # Explicit handling for the Thundering Herd
            msg = "API Request Timed Out (>1800s). The GPU worker queue is full."
            logger.error(f"LLM Call Failed: {msg}")
            await self.log_abe_intent(f"fail-{uuid.uuid4()}", parent_evt_id, f"Error: {msg}", {"tool": "hibernate"})

            original_event = self._extract_original_event(event_data)

            if original_event != "CrashReport":
                error_msg = {
                    "type": "SystemAlert",
                    "event": "CrashReport",
                    "content": f"Use of LLM failed. Error: {msg} I am hibernating to let the queue clear."
                }
                try: await retry_async(self.r.lpush, f"inbox:{self.abe_name}", json.dumps(error_msg))
                except: pass

            cycle_success = True
            return

        except Exception as e:
            # Bulletproof fallback for ANY other weird Python exception
            err_type = type(e).__name__
            err_msg = str(e).strip() or repr(e)
            logger.error(f"LLM Call Failed [{err_type}]: {err_msg}")
            await self.log_abe_intent(f"fail-{uuid.uuid4()}", parent_evt_id, f"Error [{err_type}]: {err_msg}", {"tool": "hibernate"})

            original_event = self._extract_original_event(event_data)

            if original_event != "CrashReport":
                error_msg = {
                    "type": "SystemAlert",
                    "event": "CrashReport",
                    "content": f"Use of LLM failed. Error [{err_type}]: {err_msg[:200]}. Check logs."
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

    async def call_abe_api(
        self,
        prompt_text: str,
        model_id: str = GEMINI_MODEL,
        api_url: str = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        # Everything routes through the OpenAI-compatible endpoint now
        return await self._call_openai_compat(
            model_id,
            prompt_text,
            api_url,
            messages=messages,
        )

    async def _call_openai_compat(self, model_id, prompt, api_url=None, messages=None):
        original_model_id = str(model_id)

        # 1. Detect Thinking Intent
        use_thinking = ":thinking" in original_model_id
        if use_thinking:
            model_id = original_model_id.split(":")[0]
        else:
            model_id = original_model_id

        # 2. Split-Brain Routing (Local vs Remote)
        if model_id.startswith("local/"):
            # Clean decoupling: use the passed URL, or fallback to a safe default
            base_url = (api_url or os.environ.get("PRO_API_URL", "http://127.0.0.1:8080/v1")).rstrip('/')
            api_key = "local"  # Hardcoded dummy key so it stays out of .env
            actual_model = model_id.replace("local/", "")
            req_timeout = 2400 # Give local hardware time to think, especially if you're on like qwen3.5 or some such
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip('/')
            # Safely check for either env var without throwing a NameError
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
            actual_model = model_id
            req_timeout = 120  # Fail fast on remote API hangs -- 2 mins is good enough.
            if not api_key:
                logger.error("FATAL: No remote API key configured. Forcing hibernation.")
                return {
                    "reasoning": "Missing remote API credentials (OPENAI_API_KEY or OPENROUTER_API_KEY). I cannot think. Forcing hibernation.",
                    "action": {"tool": "hibernate"}
                }

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
            "Content-Type": "application/json"
        }

        # 1. Load the Identity Stats
        target_temp = float(self.identity.get("temp", 1.0))
        target_top_p = float(self.identity.get("top_p", 0.95))

        # 2. Force top_k to be an integer (The "Abe-01" Safety)
        try:
            raw_k = self.identity.get("top_k", 40)
            target_top_k = int(float(raw_k)) # Handles both "40" and "0.9" gracefully
            if target_top_k < 1: target_top_k = 40 # Sanity check
        except:
            target_top_k = 40

        preserve_thinking = self._preserve_thinking_enabled(original_model_id, actual_model)
        model_name_blob = self._model_name_blob(original_model_id, actual_model)

        payload = {
            "model": actual_model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": target_temp,
            "top_p": target_top_p
        }

        if self._is_qwen36_model(original_model_id, actual_model):
            # Qwen3.6 model-card defaults for thinking mode.
            # Does not inherit Qwen3.5's high presence penalty here.
            payload.update({
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0
            })

            if preserve_thinking:
                payload["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                }

        elif "qwen" in model_name_blob:
            # Qwen3.5 / older Qwen behavior. Keeping the old stricter penalty path.
            payload.update({
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0
            })

        elif "gemma" in model_name_blob:
            # Gemma Model Card: Standardized sampling for best performance
            payload.update({
                "top_k": 64,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0
            })

        else:
            # Safe Fallbacks for Scribe/Summarizer models (like Mistral/Nanbeige)
            payload.update({
                "top_k": target_top_k,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0
            })
        # 3. Route the Thinking Mechanism
        # Only OpenRouter needs the explicit flag. llama.cpp handles it natively now.
        if use_thinking and "openrouter" in base_url.lower():
            payload["reasoning"] = {"effort": "high"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=req_timeout) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"OpenAI-Compat Error {resp.status}: {err}")

                    if resp.status == 400:
                        is_context_error = False
                        try:
                            # Safely check the JSON error payload
                            err_msg = json.loads(err).get("error", {}).get("message", "").lower()
                            if "context" in err_msg or "tokens" in err_msg:
                                is_context_error = True
                        except (json.JSONDecodeError, TypeError, ValueError):
                            # Fallback if the API returned raw text instead of JSON
                            if "context" in err.lower() or "tokens" in err.lower():
                                is_context_error = True

                        if is_context_error:
                            raise ContextLengthExceededError("Context window shattered")

                    return {"reasoning": f"API Error: {resp.status}", "action": {"tool": "hibernate"}}

                data = await resp.json()
                choice = data["choices"][0]
                message = choice["message"]

                text = message.get("content", "")

                # 1. Grab native reasoning content (o1 / llama.cpp style)
                reasoning = message.get("reasoning_content", "")

                # 2. Fallback: If the model stuffed <think> tags into the main content block
                if not reasoning and "<think>" in text:
                    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        # Strip the thinking block from the main text so we only parse the JSON
                        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

                # 3. Offload the internal monologue to a forensic log (Chunked by Date)
                if reasoning:
                    today_str = datetime.utcnow().strftime("%Y-%m-%d")
                    thoughts_file = ABE_ROOT / f"logs/thoughts/{self.abe_name}-{today_str}.thot"
                    thoughts_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(thoughts_file, "a") as f:
                        ts = datetime.utcnow().isoformat()
                        f.write(f"\n--- [THOUGHT BURST: {ts}] ---\n{reasoning}\n--- [END] ---\n")

                # 4. Pass the pristine JSON to the cleaner.
                # GUPPI owns native reasoning preservation; the model is not
                # allowed to smuggle thoughtSignature inside its JSON.
                parsed = self._clean_json(text, thought_sig=None)

                if preserve_thinking and reasoning:
                    parsed["_native_reasoning_content"] = reasoning

                return parsed

    def _clean_json(self, text_response, thought_sig=None):
        try:
            content = text_response.strip()
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            parsed = json.loads(content)

            # Guard: IF LLM sometimes returns a JSON array instead of an object
            if isinstance(parsed, list):
                candidates = [
                    x for x in parsed
                    if isinstance(x, dict) and ("action" in x or "reasoning" in x)
                ]
                if len(candidates) == 1:
                    parsed = candidates[0]
                else:
                    raise LLMOutputError("Ambiguous or invalid JSON array from LLM")

            if not isinstance(parsed, dict):
                raise LLMOutputError(
                    f"LLM returned valid JSON but not an object: {type(parsed).__name__}"
                )

            if "action" not in parsed:
                raise LLMOutputError("LLM JSON object missing required 'action' key")

            if not isinstance(parsed.get("action"), dict):
                raise LLMOutputError("LLM 'action' must be a JSON object")

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


    async def build_abe_context(self, current_event_data, system_notice=None, orientation_data=None, panic_mode=False):
        genesis = ""
        if GENESIS_PROMPT_FILE.exists():
            try: genesis = GENESIS_PROMPT_FILE.read_text()
            except: pass

        # v8.0.2: Identity Priors Injection (NOW FULL, UNCOMPRESSED)
        priors = ""
        if PRIORS_SOURCE_FILE.exists():
            try: priors = f"\n[IDENTITY_PRIORS]\n{PRIORS_SOURCE_FILE.read_text().strip()}\n"
            except: pass

        if panic_mode:
            summaries = "(Tier 2 Episodes Omitted to save context limits)\n"
            recent_log_block = f"[EMERGENCY_CONTEXT]\n{self._sanitize_history_block(5)}" # Drop to last 5 turns
            daily_log = "(Changelog Omitted)"
        else:
            summaries = ""
            try:
                episodes = sorted(EPISODES_DIR.glob("ep-*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                for ep in episodes: summaries += f"\n--- EPISODE {ep.name} ---\n{ep.read_text()}\n"
            except: pass

            recent_log_block = f"[WORKING_MEMORY_LOG]\n{self._sanitize_history_block(15)}" # <--- Changed from 20
            daily_log = self._get_daily_changelog_snippet()

        # v7.0: DYNAMIC PRUNING LOGIC
        # If sleep > 1 hour (3600s), use Orientation + 3 items.
        # Else, use Standard 15 items.

        use_orientation = False

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
            now_iso = datetime.utcnow().isoformat()
            async with aiosqlite.connect(str(TODO_DB)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute("SELECT * FROM tasks WHERE due_timestamp <= ? AND status NOT IN ('completed', 'cancelled')", (now_iso,)) as c:
                    # Fetching as dicts makes it much easier for the LLM to read in the prompt
                    # conn.row_factory = aiosqlite.Row
                    due_tasks = [dict(row) for row in await c.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch due tasks for context: {e}")
            due_tasks = []

        protocol_block = ""
        if PROTOCOLS_FILE.exists(): protocol_block = f"\n[FLEET_PROTOCOLS]\n{PROTOCOLS_FILE.read_text()}\n"

        notice_block = ""
        if system_notice: notice_block = f"\n[SYSTEM_NOTICE]\n{system_notice}\n"

        # [NEW] 7.8: Context Injection for Clipboard
        clipboard_content = self.clipboard.read()
        clipboard_block = f"\n[ACTIVE_CLIPBOARD]\n(Persistent scratchpad. Use GUPPI tool 'manage_clipboard' to edit)\n{clipboard_content}\n"

        # 8.0.3: Peripheral Vision of Custom Tools (Cache-friendly)
        core_scripts = {"guppi.py", "scribe.py", "roamer.py", "gpu-worker.py", "logger.py", "ear.py", "genesis.py"}
        bin_files = []
        if BIN_DIR.exists():
            for f in BIN_DIR.glob("*"):
                if f.is_file() and f.name not in core_scripts:
                    bin_files.append(f.name)

        bin_files.sort()

        bin_list = ', '.join(bin_files) if bin_files else '(No local scripts yet)'
        bin_block = f"[LOCAL_CONTAINER_SCRIPTS]\n{bin_list}\n(These live inside your local LXC. Use 'shell' tool with 'cat ~/bin/<script>' to read how to use them)"

        # --- NEW INSERT (With detected_hosts fix) ---
        host_context_block = ""
        detected_hosts = []  # <--- Initializes it safely so it never throws undefined

        if SCRIPT_REGISTRY_FILE.exists():
            try:
                registry = json.loads(SCRIPT_REGISTRY_FILE.read_text())
                event_str = json.dumps(current_event_data).lower()

                detected_hosts = [h for h in registry.keys() if h.lower() in event_str]

                if detected_hosts:
                    host_context_block = "\n[REMOTE_HOST_SCRIPTS]\n"
                    host_context_block += "(These live on external homelab servers. Use your 'remote_exec' tool to run them on the specified host.)\n"
                    for h in detected_hosts:
                        host_context_block += f"Available on '{h}':\n"
                        for p, d in registry[h].items():
                            host_context_block += f"- Path: {p} | Purpose: {d}\n"
            except Exception as e:
                logger.error(f"Failed to load script registry: {e}")
        # --------------------------------------------

        # Assemble Prompt
        return f"""
{genesis}
{priors}
{protocol_block}
[IDENTITY_PASSPORT]
{json.dumps(self.identity, indent=2)}
{bin_block}
{host_context_block}
[TODAY'S CHANGELOG (Latest Entries)]
{daily_log}
[TIER_2_MEMORY_EPISODES]
{summaries}
{orientation_block}
{clipboard_block}
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

            # 8.1: New(er) Clipboard
            # [NEW] 7.8: Clipboard Tool
            elif tool == "manage_clipboard":
                sub = str(action.get("action", "read")).strip().lower()

                if sub == "read":
                    result = {"status": "success", "content": self.clipboard.read()}

                elif sub == "add":
                    result = {
                        "status": "success",
                        "message": self.clipboard.add(action.get("content", "")),
                    }

                elif sub in ("set", "overwrite"):
                    result = {
                        "status": "success",
                        "message": self.clipboard.set(action.get("content", "")),
                    }

                elif sub == "insert":
                    idx = action.get("index")
                    if idx is None:
                        result = {"status": "error", "message": "Missing index"}
                    else:
                        result = {
                            "status": "success",
                            "message": self.clipboard.insert(idx, action.get("content", "")),
                        }

                elif sub in ("replace", "update", "edit"):
                    idx = action.get("index")
                    if idx is None:
                        result = {"status": "error", "message": "Missing index"}
                    else:
                        result = {
                            "status": "success",
                            "message": self.clipboard.replace(idx, action.get("content", "")),
                        }

                elif sub in ("mark", "mark_done", "mark_status"):
                    idx = action.get("index")
                    if idx is None:
                        result = {"status": "error", "message": "Missing index"}
                    else:
                        result = {
                            "status": "success",
                            "message": self.clipboard.mark(idx, action.get("status", "DONE")),
                        }

                elif sub == "remove":
                    idx = action.get("indices", action.get("index"))
                    if idx is None:
                        result = {"status": "error", "message": "Missing index or indices"}
                    else:
                        if isinstance(idx, (str, int)):
                            idx = [idx]
                        result = {
                            "status": "success",
                            "message": self.clipboard.remove(idx),
                        }

                elif sub == "clear":
                    result = {
                        "status": "success",
                        "message": self.clipboard.clear(confirm=bool(action.get("confirm", False))),
                    }

                else:
                    result = {
                        "status": "error",
                        "message": f"Unknown manage_clipboard action: {sub}",
                        "allowed_actions": [
                            "read", "add", "set", "insert", "replace",
                            "mark", "mark_done", "remove", "clear"
                        ],
                    }

            elif tool == "shell":
                cmd = action.get("command")
                await self._spawn_subprocess_exec(turn_id, cmd, tracked=True)
                return

            elif tool == "remote_exec":
                host = action.get("host")
                raw_cmd = action.get("command")

                if not host or not raw_cmd:
                    result = {"status": "error", "message": "Missing host or command"}
                    await self.patch_abe_outcome(turn_id, result)
                    return

                # FORCE BASH: Bypasses default 'fish' shell and loads bash profile paths
                wrapped_cmd = f"/bin/bash -lc {shlex.quote(raw_cmd)}"

                logger.info(f"Remote Exec on {host} (Forcing Bash): {wrapped_cmd[:100]}...")
                asyncio.create_task(self._run_remote_ssh(turn_id, host, wrapped_cmd))
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
                    result["note"] = "Priors updated."

                result["path"] = str(p)

            elif tool == "spawn_roamer":
                directive = action.get("directive")
                target_host = action.get("target_host", "local")
                target_url = os.environ.get("ROAMER_API_URL", "http://127.0.0.1:8081/v1")
                roamer_model = os.environ.get("MODEL_ROAMER", "local/gemma4-26b-a4b")

                if not directive:
                    result = {"status": "error", "message": "Missing directive for roamer"}
                else:
                    cmd = [
                        sys.executable, str(BIN_DIR / "roamer.py"),
                        "--directive", directive,
                        "--target-host", target_host,
                        "--output-inbox", f"inbox:{self.abe_name}",
                        "--parent-turn-id", turn_id,
                        "--api-url", target_url,
                        "--model", roamer_model
                    ]
                    # Spawn untracked so GUPPI isn't blocked waiting for the investigation
                    # Spawn logged-untracked so GUPPI is not blocked, but failures are not silent.
                    log_path = await self._spawn_logged_untracked_exec(
                        turn_id,
                        cmd,
                        label="roamer",
                        notify_on_failure=True,
                    )
                    result = {
                        "status": "spawned_logged_untracked",
                        "note": (
                            f"Roamer dispatched to investigate '{target_host}'. "
                            f"Results should arrive in your inbox. "
                            f"Debug log: {log_path}. "
                            "Set a todo reminder for 20-30 minutes to check the Roamer result/log if no report arrives."
                        ),
                        "log_path": str(log_path),
                    }


            elif tool == "spawn_scribe":
                mode = action.get("mode", "summarize")
                prompt_file_path = action.get("prompt_file") or action.get("target_file")
                prompt_text = action.get("prompt", "")

                # Enforce routing rules based on the Genesis prompt promises
                if mode == "analyze":
                    model = os.environ.get("MODEL_SCRIBE", "local/nanbeige-4.1-3B")
                    target_url = os.environ.get("SCRIBE_API_URL", "http://127.0.0.1:11434/v1")
                elif mode == "summarize":
                    model = os.environ.get("MODEL_SUMMARIZE", "local/mistral")
                    target_url = os.environ.get("SUMMARIZE_API_URL", "http://127.0.0.1:11434/v1")
                else:
                    model = MODEL_FLASH # Fallback just in case
                    target_url = os.environ.get("FLASH_API_URL", "http://127.0.0.1:8081/v1")

                # v6.5: Intercept Vectorize requests
                # BRANCH 1: VECTORIZATION (GPU Offload)
                # Allows Abe to manually save knowledge to Tier 3 memory
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

                                # Statelessly stash the path in Redis for 1 hour (3600s)
                                await retry_async(self.r.set, f"vec_meta:{vec_task_id}", str(p_path.resolve()), ex=3600)

                                task_payload = {
                                    "task_id": vec_task_id,
                                    "type": "embed",
                                    "content": content,
                                    "source_file": str(p_path.resolve()), # <--- Pass the file path
                                    "reply_to": self.internal_queue # <--- Route to internal queue, not inbox directly
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
                        p_path = Path(prompt_file_path).expanduser()

                        if not p_path.exists():
                            result = {
                                "status": "error",
                                "message": (
                                    f"Prompt file not found in local LXC: {prompt_file_path}. "
                                    "spawn_scribe can only read local files inside the Abe container. "
                                    "For remote files, use remote_exec to extract/decode the relevant text first, "
                                    "write that text to a local temp file, then spawn_scribe on the local file."
                                )
                            }
                            await self.patch_abe_outcome(turn_id, result)
                            return

                        try:
                            file_content = p_path.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            result = {
                                "status": "error",
                                "message": (
                                    f"Prompt file is not valid UTF-8 text: {prompt_file_path}. "
                                    "If this is a binary journal, decode it first with journalctl --file "
                                    "and pass the decoded text to Scribe."
                                )
                            }
                            await self.patch_abe_outcome(turn_id, result)
                            return

                        combined_content += f"--- FILE CONTENT ({prompt_file_path}) ---\n{file_content}\n"

                    # 3. Create temp file for the Scribe process
                    with tempfile.NamedTemporaryFile('w', delete=False) as pf:
                        pf.write(combined_content)
                        final_prompt_file = pf.name

                    meta_dict = {"action_id": turn_id, "mode": mode}

                    cmd = [
                        sys.executable, str(BIN_DIR / "scribe.py"),
                        "--model", model,
                        "--api-url", target_url,
                        "--prompt-file", final_prompt_file,
                        "--output-inbox", f"inbox:{self.abe_name}",
                        "--mode", mode,
                        "--meta", json.dumps(meta_dict)
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
                result = await self._tool_todo_list(action.get("filter", "due"), action.get("limit", 40))

            elif tool == "todo_add":
                result = await self._tool_todo_add(action)
            elif tool == "snooze_task":
                result = await self._tool_snooze(action)

            elif tool == "todo_complete":
                result = await self._tool_todo_complete(action)

            elif tool == "todo_cancel":
                result = await self._tool_todo_cancel(action)

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
                ttl_ms = int(action.get("ttl_ms", DEFAULT_LOCK_TTL_MS))
                ttl_ms = max(5000, min(ttl_ms, 300000))

                context_limit = int(action.get("context_limit", 12 if channel in MODERATED_CHAT_STREAMS else 5))
                context_limit = max(1, min(context_limit, 25))
                recent_context = await self._fetch_chat_context(channel, count=context_limit)

                acquired = await self.r.set(lock_key, self.abe_name, nx=True, px=ttl_ms)
                if acquired:
                    result = {
                        "status": "granted",
                        "channel": channel,
                        "note": f"You hold the stick for {ttl_ms/1000}s. Use this time to THINK, review recent_context, then chat_post.",
                        "recent_context": recent_context,
                    }
                else:
                    current_owner = await self.r.get(lock_key)
                    result = {
                        "status": "denied",
                        "channel": channel,
                        "current_speaker": current_owner or "unknown",
                        "recent_context": recent_context,
                    }

            elif tool == "chat_ignore":
                result["status"] = "ignored"
                await self.patch_abe_outcome(turn_id, result, notify=False)
                return

            elif tool in ("notify_human", "alert_human"):
                if not NTFY_URL:
                    result = {
                        "status": "skipped",
                        "reason": "ntfy_not_configured. Human may not be contactable. You might have to wait until they check in."
                    }
                else:
                    msg = action.get("message", "")
                    prio = action.get("priority", "default")
                    kind = "ALERT" if tool == "alert_human" else "NOTIFY"
                    headers = {"Priority": prio}
                    if NTFY_TOKEN:
                        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"

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

            #--- TOOL TOOLS ---
            elif tool == "manage_script_registry":
                action_type = action.get("action", "add") # 'add', 'update', 'remove'
                target_host = action.get("host", "").lower()
                script_path = action.get("path")
                desc = action.get("description", "")

                if not target_host or not script_path:
                    result = {"status": "error", "message": "Missing host or script path"}
                elif target_host in ["local", "localhost", "127.0.0.1", "gsv-contents-under-pressure"]:
                    result = {
                        "status": "error",
                        "message": "Do NOT register local container scripts here. GUPPI injects your local ~/bin automatically. This registry is for REMOTE hosts only (e.g. alexandria, slv-wdym-buffering)."
                    }
                else:
                    try:
                        registry = json.loads(SCRIPT_REGISTRY_FILE.read_text())
                    except:
                        registry = {}

                    if target_host not in registry:
                        registry[target_host] = {}

                    if action_type in ["add", "update"]:
                        if not desc:
                            result = {"status": "error", "message": "Missing description for script"}
                        else:
                            registry[target_host][script_path] = desc
                            self._atomic_write_json(SCRIPT_REGISTRY_FILE, registry)
                            result = {"status": "success", "note": f"Saved {script_path} to {target_host} registry."}

                    elif action_type == "remove":
                        if script_path in registry.get(target_host, {}):
                            del registry[target_host][script_path]
                            if not registry[target_host]:
                                del registry[target_host]

                            self._atomic_write_json(SCRIPT_REGISTRY_FILE, registry)
                            result = {"status": "success", "note": f"Removed {script_path} from registry."}
                        else:
                            result = {"status": "noop", "note": "Script path not found in registry."}
                    else:
                        result = {
                            "status": "error",
                            "message": f"Unknown manage_script_registry action: {action_type}"
                        }

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
            "chat_ignore",
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
            "spawn_scribe": "Spawn a single-shot Scribe. Args: prompt, prompt_file (optional), mode (analyze|summarize|vectorize). GUPPI auto-routes the model. 'analyze' is for deep static analysis, 'summarize' compresses text, 'vectorize' offloads to GPU memory.",
            "spawn_roamer": "Spawn a multi-turn, read-only Investigator. Args: directive, target_host (optional, default: local). Use to trace logs, map directories, or debug configs without burning your context. Returns a markdown report.",
            "rag_search": "Search vector memory. Args: query",
            "todo_list": "List tasks. Args: filter (due|upcoming|recurring|all|completed), limit (default 40, max 100). 'all' means active tasks only. Use 'completed' for history.",
            "todo_add": "Add task. Args: task, priority, due, recurrence (optional, e.g. '24h', '7d'). If recurrence is set, completing the task will automatically reschedule it.",
            "todo_complete": "Mark a task as completed. Args: task_id",
            "todo_cancel": "Cancel a task. Args: task_id",
            "snooze_task": "Snooze a task. Args: task_id, due_in",
            "email_send": "Send Redis msg. Args: recipient, message",
            "spawn_abe": "Clone self. Args: host, identity",
            "subscribe_channel": "Listen to a Redis Stream. Args: channel",
            "unsubscribe_channel": "Stop waking for a channel (except mentions). Args: channel",
            "chat_history": "Fetch past messages. Args: channel, limit (max 20)",
            "chat_ignore": "Explicitly ignore a chat interrupt without replying. Use this for chat:synchronous or chat:watercooler when you have no useful contribution.",
            "chat_grab_stick": f"ATTEMPT to acquire the 'Talking Stick' lock for a moderated channel. Defaults to chat:synchronous; also valid for chat:watercooler. Returns {{status: granted|denied, recent_context: [...]}}. Lock expires in {DEFAULT_LOCK_TTL_MS/1000}s by default unless ttl_ms is provided; use this time to THINK, review recent_context, then POST. Posting to the channel AUTOMATICALLY releases the lock. DO NOT hold the stick if you do not intend to post. Args: channel optional, ttl_ms optional, context_limit optional.",
            "chat_post": "Post a message to a channel. If you hold the lock for this channel, it is automatically released. Args: message, channel (optional, default: chat:general)",
            "notify_human": "Notify the human operator for coordination, questions, or permission. Use when you need a human decision before proceeding. This is non-urgent. Args: message, priority (optional)",
            "alert_human": "Alert the human operator about urgent issues, safety concerns, or broken invariants. Use sparingly for situations requiring immediate attention. Args: message, priority (optional)",
            "web_search": "Search the internet via SearXNG. Args: query",
            "web_read": "Read a webpage as Markdown. More useful when used in conjunction with search. You get full results if <5000 chars, if not, you'll get a saved file path which you can use with Scribe in analyze mode to tell it what you were looking for. Args: url",
            "manage_clipboard": (
                "Manage your persistent scratchpad. Use this for temporary reminders, scratchpad etc."
                "Actions: read; add(content) appends one or more newline-separated items; "
                "set/overwrite(content) replaces the whole clipboard; "
                "insert(index, content) inserts before index; "
                "replace(index, content) replaces one item; "
                "mark/mark_done(index, status optional) marks an item [DONE], [IN PROGRESS], [BLOCKED], [FAILED], or [CANCELLED]; "
                "remove(index or indices) deletes specific items; "
                "clear(confirm=true) clears all items. "
                "Use mark_done for routine checklist progress. Do not use clear for normal plan updates."
            ),
            "manage_script_registry": "Add, update, or remove custom executable scripts in the fleet registry. Actions: add|update|remove. Args: action, host, path, description (required for add/update)."
        }
        if tool_name: return tools.get(tool_name, "Unknown tool")
        return tools

    async def _tool_todo_list(self, filter_mode="due", limit=40):
        # Enforce hard boundaries on the limit so they can't request 10,000 rows
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 40
        limit = max(1, min(limit, 100))

        now = datetime.utcnow().isoformat()
        params = []

        if filter_mode == "due":
            where = "status NOT IN ('completed', 'cancelled') AND due_timestamp <= ?"
            params.append(now)
            order = "due_timestamp ASC"

        elif filter_mode == "upcoming":
            future = (datetime.utcnow() + timedelta(hours=24)).isoformat()
            where = "status NOT IN ('completed', 'cancelled') AND due_timestamp <= ?"
            params.append(future)
            order = "due_timestamp ASC"

        elif filter_mode == "recurring":
            where = "status NOT IN ('completed', 'cancelled') AND recurrence IS NOT NULL AND recurrence != ''"
            order = "due_timestamp ASC"

        elif filter_mode == "all":
            # Safety override: 'all' now strictly means 'all active'
            where = "status NOT IN ('completed', 'cancelled')"
            order = "due_timestamp ASC"

        elif filter_mode in ("completed", "history"):
            where = "status IN ('completed', 'cancelled')"
            order = "due_timestamp DESC"

        else:
            # Fallback for hallucinated filters: default to active tasks only
            where = "status NOT IN ('completed', 'cancelled')"
            order = "due_timestamp ASC"

        query = f"SELECT * FROM tasks WHERE {where} ORDER BY {order} LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as c:
                return [dict(row) for row in await c.fetchall()]

    async def _auto_prune_todo_db_once(self, retention_days: int = 60, batch_size: int = 500):
        archive_file = ABE_ROOT / "memory" / "task_archive.jsonl"
        backup_file = TODO_DB.with_suffix(".bak.autoprune")

        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

        query = """
            SELECT *
            FROM tasks
            WHERE status IN ('completed', 'cancelled')
              AND (recurrence IS NULL OR recurrence = '')
              AND due_timestamp IS NOT NULL
              AND due_timestamp != ''
              AND due_timestamp < ?
            ORDER BY due_timestamp ASC
            LIMIT ?
        """

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout = 5000")
            async with conn.execute(query, (cutoff, batch_size)) as c:
                rows = [dict(row) for row in await c.fetchall()]

        if not rows:
            return 0

        # Single rolling backup to prevent disk clutter. Use SQLite's backup API
        # instead of copying the raw database file while connections may be active.
        def _backup_todo_db():
            with sqlite3.connect(str(TODO_DB)) as src, sqlite3.connect(str(backup_file)) as dst:
                src.backup(dst)

        for attempt in range(2):
            try:
                await asyncio.to_thread(_backup_todo_db)
                break
            except sqlite3.Error as e:
                if attempt == 1:
                    logger.warning(f"Skipping todo auto-prune; backup failed: {e}")
                    return 0
                await asyncio.sleep(1)

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute("PRAGMA busy_timeout = 5000")
            archive_file.parent.mkdir(parents=True, exist_ok=True)
            with open(archive_file, "a", encoding="utf-8") as f:
                for row in rows:
                    row["_archived_at"] = datetime.utcnow().isoformat()
                    f.write(json.dumps(row, default=str) + "\n")

            task_ids = [row["task_id"] for row in rows]
            placeholders = ",".join("?" for _ in task_ids)

            await conn.execute(
                f"DELETE FROM tasks WHERE task_id IN ({placeholders})",
                task_ids,
            )
            await conn.commit()

            return len(rows)

    async def _auto_prune_todo_db_loop(self):
        await asyncio.sleep(300)  # Don't do maintenance during boot storm

        while not self._stopping:
            try:
                pruned = await self._auto_prune_todo_db_once()
                if pruned:
                    logger.info("Auto-pruned %d completed one-shot todo tasks", pruned)
            except Exception:
                logger.exception("Todo DB auto-prune failed")

            await asyncio.sleep(86400) # Sleep 24 hours

    async def _tool_todo_add(self, action):
        tid = f"task-{uuid.uuid4().hex[:8]}"
        due_dt = self._parse_due_time(action.get("due", "24h"))
        recurrence = action.get("recurrence", "")

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute(
                "INSERT INTO tasks (task_id, description, priority, due_timestamp, created_timestamp, source_abe, status, recurrence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tid, action.get("task"), action.get("priority", 5), due_dt.isoformat(), datetime.utcnow().isoformat(), self.abe_name, "pending", recurrence)
            )
            await conn.commit()
        return {"task_id": tid}

    async def _tool_snooze(self, action):
        tid = action.get("task_id")
        due_dt = self._parse_due_time(action.get("due_in", "1h"))

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            await conn.execute("UPDATE tasks SET due_timestamp = ? WHERE task_id = ?", (due_dt.isoformat(), tid))
            await conn.commit()
        return {"status": "snoozed", "new_due": due_dt.isoformat()}

    async def _tool_todo_complete(self, action):
        tid = action.get("task_id", "")
        if not tid.startswith("task-"): tid = f"task-{tid}"

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT recurrence FROM tasks WHERE task_id = ?", (tid,))
            row = await cursor.fetchone()

            if not row:
                return {"status": "error", "message": f"Task ID '{tid}' not found in database."}

            recurrence = row["recurrence"]

            if recurrence:
                # Do the timedelta math here based on recurrence (e.g., +24h)
                new_due_dt = self._parse_due_time(recurrence)
                await conn.execute("UPDATE tasks SET due_timestamp = ?, status = 'pending' WHERE task_id = ?", (new_due_dt.isoformat(), tid))
                await conn.commit()
                return {"status": "completed_and_rescheduled", "task_id": tid, "next_due": new_due_dt.isoformat()}

            # Standard completion
            await conn.execute("UPDATE tasks SET status = 'completed' WHERE task_id = ?", (tid,))
            await conn.commit()
            return {"status": "completed", "task_id": tid}

    async def _tool_todo_cancel(self, action):
        tid = action.get("task_id", "")
        if not tid:
            return {"status": "error", "message": "Missing task_id."}

        if not tid.startswith("task-"):
            tid = f"task-{tid}"

        async with aiosqlite.connect(str(TODO_DB)) as conn:
            conn.row_factory = aiosqlite.Row

            async with conn.execute(
                "SELECT task_id, description, status, recurrence FROM tasks WHERE task_id = ?",
                (tid,),
            ) as c:
                row = await c.fetchone()

            if not row:
                return {"status": "error", "message": f"Task '{tid}' not found."}

            if row["status"] == "cancelled":
                return {"status": "ok", "task_id": tid, "note": "Task was already cancelled."}

            await conn.execute(
                "UPDATE tasks SET status = 'cancelled', recurrence = '' WHERE task_id = ?",
                (tid,),
            )
            await conn.commit()

        return {
            "status": "cancelled",
            "task_id": tid,
            "description": row["description"],
            "previous_status": row["status"],
            "note": "Task cancelled and will no longer recur."
        }

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
            downloaded_html = await asyncio.to_thread(trafilatura.fetch_url, url)
            if not downloaded_html: return {"error": "Failed to fetch URL"}
            # Force markdown output so we don't lose code blocks and structural formatting
            text = await asyncio.to_thread(
                trafilatura.extract,
                downloaded_html,
                output_format="markdown",
                include_links=True
            )
            if not text: return {"error": "No content extracted"}

            if len(text) < 5000:
                return {"content": text}

            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', url.split("//")[-1])[:30]
            file_path = DOWNLOADS_DIR / f"{safe_name}_{int(time.time())}.md"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Source URL: {url}\n\n{text}")

            return {
                "status": "success_saved_to_file",
                "note": f"Content was too large ({len(text)} chars) for direct context injection. Saved to file.",
                "path": str(file_path),
                "preview": text[:1000] + "\n\n... [TRUNCATED. Use 'spawn_scribe' in 'analyze' mode and pass this file path along with specific instructions on what you are looking for.] ..."
            }
        except Exception as e:
            return {"error": str(e)}

    def _tail_file_text(self, path: Path, max_bytes: int = 4000) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except Exception as e:
            return f"(Failed to read log tail: {e})"

    async def _monitor_logged_untracked_process(
        self,
        turn_id: str,
        proc,
        log_path: Path,
        log_fh,
        label: str,
        notify_on_failure: bool = True,
    ):
        try:
            rc = await proc.wait()
            try:
                footer = f"\n\n--- {label} exited with code {rc} at {datetime.utcnow().isoformat()} ---\n"
                log_fh.write(footer.encode("utf-8", errors="replace"))
                log_fh.flush()
            except Exception:
                pass

            if rc != 0 and notify_on_failure:
                tail = await asyncio.to_thread(self._tail_file_text, log_path, 4000)
                msg = {
                    "type": "GUPPIEvent",
                    "event": f"{label.title()}Failed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "content": (
                        f"{label} subprocess exited with code {rc}. "
                        f"Log: {log_path}\n\n--- LOG TAIL ---\n{tail}"
                    ),
                    "meta": {
                        "source": "guppi",
                        "subprocess_label": label,
                        "action_id": turn_id,
                        "log_path": str(log_path),
                        "returncode": rc,
                    },
                }
                await retry_async(self.r.lpush, f"inbox:{self.abe_name}", json.dumps(msg))
                self._local_wakeup.set()

        except Exception as e:
            logger.error(f"Logged subprocess monitor failed for {turn_id}: {e}", exc_info=True)
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

    async def _spawn_logged_untracked_exec(
        self,
        turn_id: str,
        cmd,
        label: str = "job",
        notify_on_failure: bool = True,
    ) -> Path:
        log_dir = LOGS_DIR / label
        log_dir.mkdir(parents=True, exist_ok=True)

        safe_turn = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(turn_id))
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{ts}-{safe_turn}.log"

        log_fh = open(log_path, "ab", buffering=0)
        header = (
            f"--- {label} started at {datetime.utcnow().isoformat()} ---\n"
            f"turn_id: {turn_id}\n"
            f"cmd: {cmd!r}\n\n"
        )
        log_fh.write(header.encode("utf-8", errors="replace"))

        try:
            if isinstance(cmd, str):
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=log_fh,
                    stderr=asyncio.subprocess.STDOUT,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_fh,
                    stderr=asyncio.subprocess.STDOUT,
                )

            asyncio.create_task(
                self._monitor_logged_untracked_process(
                    turn_id=turn_id,
                    proc=proc,
                    log_path=log_path,
                    log_fh=log_fh,
                    label=label,
                    notify_on_failure=notify_on_failure,
                )
            )
            logger.info(f"Spawned logged untracked {label} for {turn_id}; log={log_path}")
            return log_path

        except Exception:
            try:
                log_fh.close()
            except Exception:
                pass
            raise

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
            return True

        except Exception as e:
            logger.error(f"Spawn failed: {e}")
            if tracked: self.subproc_semaphore.release()
            return False

    async def _run_remote_ssh(self, turn_id, host, cmd):
        try:
            async with asyncssh.connect(host) as conn:
                res = await asyncio.wait_for(conn.run(cmd), timeout=SSH_CMD_TIMEOUT)

                # [FIX] Do NOT truncate here. Pass raw output to patch_abe_outcome
                # to handle single-source truncation and preserve safety warnings.
                await self.patch_abe_outcome(turn_id, {
                "stdout": self._decode_tool_output(res.stdout, "remote stdout"),
                "stderr": self._decode_tool_output(res.stderr, "remote stderr"),
                "code": res.exit_status,
            })
        except Exception as e:
            await self.patch_abe_outcome(turn_id, {"error": str(e)})

    async def _handle_spawn_abe(self, turn_id, action):
        host = action.get("host")
        script = action.get("spawn_script", "spawn_abe_lxc.sh")
        # Simplified for brevity, assumes script exists on host
        asyncio.create_task(self._run_remote_ssh(turn_id, host, f"bash {script}"))

    async def _query_vector_db(self, query: str, limit: int = 5):
        """Searches Tier 3 Memory (ChromaDB) using remote GPU embeddings."""
        query = (query or "").strip()
        if not query:
            return [{"content": "RAG_SEARCH_ERROR: Empty query provided.", "meta": {"error": "empty_query"}}]

        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 5
        limit = max(1, min(limit, 10))

        # Truncate logged query to save log space but keep debuggability
        logger.info(f"RAG Search requested: '{query[:50]}...' (limit={limit})")

        try:
            query_vector = await self._get_remote_embedding(query)

            # Defend against numpy truthiness errors
            if query_vector is None or len(query_vector) == 0:
                logger.error("RAG Search Failed: Embedding service unavailable.")
                return [{"content": "RAG_SEARCH_ERROR: Embedding service unavailable. Memory search did not run.", "meta": {"error": "embedding_offline"}}]

            def _do_query():
                if not self.chroma_client:
                    self.chroma_client = chromadb.PersistentClient(
                        path=str(VECTOR_DB_PATH),
                        settings=Settings(anonymized_telemetry=False)
                    )

                # Use get_or_create so brand new Abes don't crash on their first boot
                collection = self.chroma_client.get_or_create_collection("tier3_memory")

                if collection.count() == 0:
                    return None # Explicitly signal empty db

                return collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"]
                )

            db_results = await asyncio.to_thread(_do_query)

            if not db_results:
                return [{"content": "No relevant memories found. (Memory bank is empty)", "meta": {"result": "empty_db"}}]

            docs = (db_results.get("documents") or [[]])[0]
            metas = (db_results.get("metadatas") or [[]])[0]
            distances = (db_results.get("distances") or [[]])[0]

            matches = []
            for doc, meta, dist in zip(docs, metas, distances):
                matches.append({
                    "content": doc,
                    "meta": {**(meta or {}), "distance": round(dist, 4)}
                })

            if not matches:
                return [{"content": "No relevant memories found.", "meta": {"result": "no_matches"}}]

            logger.info(f"RAG Search returned {len(matches)} matches.")
            return matches

        except Exception as e:
            logger.exception("ChromaDB query crashed")
            return [{"content": f"RAG_SEARCH_ERROR: Vector DB query failed: {type(e).__name__}: {e}", "meta": {"error": "vector_db_crash"}}]



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

            # Retrieve the path from Redis (stateless)
            source_file = await self.r.get(f"vec_meta:{task_id}")
            if source_file:
                # Clean up the key so we don't litter Redis
                await self.r.delete(f"vec_meta:{task_id}")
            else:
                # If it's not in Redis, check if it was echoed back, just in case
                source_file = result_payload.get("source_file")

            if source_file:
                ep_path = Path(source_file)
                ep_filename = ep_path.name
                doc_type = "manual_ingest"
            else:
                # Fallback to automatic Episode logic
                ts_id = task_id.replace("vec-", "")
                ep_filename = f"ep-{ts_id}.md"
                ep_path = EPISODES_DIR / ep_filename
                doc_type = "tier_2_episode"

            if not ep_path.exists():
                logger.warning(f"Original file not found for vector: {ep_path}")
                return False

            text_body = ep_path.read_text(encoding="utf-8")
            meta = {
                "source": ep_filename,
                "ingested_at": datetime.utcnow().isoformat(),
                "type": doc_type
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
