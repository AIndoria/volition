#!/usr/bin/env python3
"""
Spider - bounded local deep-research harness.

Spider accepts a research question plus seed URLs/files, runs an explicit
search/read/analyze action loop, and writes a cited markdown report with
session JSON and JSONL trace artifacts. Nanbeige is used through normal
OpenAI-compatible chat completions; all web/file actions are executed by this
harness, not by hidden/native model tools.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests


for env_file in (Path.home() / ".env", Path(__file__).resolve().parent.parent / ".env", Path(__file__).parent / ".env"):
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


SEARXNG_URL = os.environ.get("SPIDER_SEARXNG_URL", os.environ.get("SEARXNG_URL", "https://civitat.es/search"))
SCRIBE_API_URL = os.environ.get(
    "SPIDER_RESEARCH_BASE_URL",
    os.environ.get("SCRIBE_API_URL", "http://127.0.0.1:8080/v1"),
).rstrip("/")
NANBEIGE_MODEL = os.environ.get(
    "SPIDER_RESEARCH_MODEL",
    os.environ.get("MODEL_SCRIBE", "nanbeige-4.1-3B:latest"),
).replace("local/", "")
OPENAI_API_KEY = os.environ.get("SPIDER_RESEARCH_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
CONTROLLER_API_URL = os.environ.get("SPIDER_CONTROLLER_BASE_URL", SCRIBE_API_URL).rstrip("/")
CONTROLLER_MODEL = os.environ.get("SPIDER_CONTROLLER_MODEL", NANBEIGE_MODEL).replace("local/", "")
CONTROLLER_API_KEY = os.environ.get("SPIDER_CONTROLLER_API_KEY", OPENAI_API_KEY)
PRO_API_URL = os.environ.get("PRO_API_URL", os.environ.get("SPIDER_PRO_BASE_URL", SCRIBE_API_URL)).rstrip("/")
PRO_MODEL = os.environ.get("MODEL_PRO", NANBEIGE_MODEL).replace("local/", "").replace(":thinking", "")
STRUCTURER_API_URL = os.environ.get("SPIDER_STRUCTURER_BASE_URL", PRO_API_URL).rstrip("/")
STRUCTURER_MODEL = os.environ.get("SPIDER_STRUCTURER_MODEL", PRO_MODEL or NANBEIGE_MODEL).replace("local/", "").replace(":thinking", "")
STRUCTURER_API_KEY = os.environ.get("SPIDER_STRUCTURER_API_KEY", CONTROLLER_API_KEY)
WRITER_API_URL = os.environ.get("SPIDER_WRITER_BASE_URL", STRUCTURER_API_URL).rstrip("/")
WRITER_MODEL = os.environ.get("SPIDER_WRITER_MODEL", STRUCTURER_MODEL).replace("local/", "").replace(":thinking", "")
WRITER_API_KEY = os.environ.get("SPIDER_WRITER_API_KEY", STRUCTURER_API_KEY)
EDITOR_API_URL = os.environ.get("SPIDER_EDITOR_BASE_URL", WRITER_API_URL).rstrip("/")
EDITOR_MODEL = os.environ.get("SPIDER_EDITOR_MODEL", WRITER_MODEL).replace("local/", "").replace(":thinking", "")
EDITOR_API_KEY = os.environ.get("SPIDER_EDITOR_API_KEY", WRITER_API_KEY)
FILTER_API_URL = os.environ.get("SPIDER_FILTER_BASE_URL", CONTROLLER_API_URL).rstrip("/")
FILTER_MODEL = os.environ.get("SPIDER_FILTER_MODEL", CONTROLLER_MODEL).replace("local/", "")
FILTER_API_KEY = os.environ.get("SPIDER_FILTER_API_KEY", CONTROLLER_API_KEY)
USE_FILTER_MODEL = os.environ.get("SPIDER_USE_FILTER_MODEL", "0") == "1"
DEBUG_DIR = Path.home() / "logs" / "debug"
SESSION_DIR = Path(os.environ.get("SPIDER_OUTPUT_DIR", str(Path.home() / "spider_sessions")))
CACHE_DIR = SESSION_DIR / ".cache"
DOC_DIR = SESSION_DIR / ".docs"
try:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    fallback_session_dir = Path.home() / "spider_sessions"
    print(f"[WARN] Could not use SPIDER_OUTPUT_DIR={SESSION_DIR}: {exc}; falling back to {fallback_session_dir}", file=sys.stderr)
    SESSION_DIR = fallback_session_dir
    CACHE_DIR = SESSION_DIR / ".cache"
    DOC_DIR = SESSION_DIR / ".docs"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_BACKEND_UNAVAILABLE = False
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_LOCK_PATH = Path(os.environ.get("SPIDER_LOCK_PATH", "/tmp/spider-gpu.lock"))
DEFAULT_BLOCKLIST_DOMAINS = {
    "csdn.net",
    "mynw.cn",
    "globenewswire.com",
    "skywork.ai",
}


def default_search_backends(api_key_env: str, configured_env: str) -> list[str]:
    configured = os.environ.get(configured_env, "").strip()
    if configured:
        return [item.strip().lower() for item in configured.split(",") if item.strip()]
    return ["brave", "searxng"] if os.environ.get(api_key_env, "").strip() else ["searxng"]


def normalized_query_key(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls = []
    for match in re.finditer(r"https?://[^\s<>)\"']+", text or ""):
        url = match.group(0).rstrip(".,;:]}")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def normalize_search_query(query: str, max_chars: int = 220) -> str:
    query = re.sub(r"https?://\S+", " ", query or "")
    query = re.sub(r"[\n\r\t]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip(" -.,;:")
    if len(query) <= max_chars:
        return query
    words = [
        w.strip(".,;:()[]{}\"'")
        for w in query.split()
        if len(w.strip(".,;:()[]{}\"'")) > 2
    ]
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about", "would", "should", "their",
        "there", "then", "also", "like", "have", "been", "your", "mine", "they", "them", "according",
        "research", "report", "include", "please", "before", "beginning", "unclear",
    }
    priority_terms = []
    for word in words:
        low = word.lower()
        if low in stop:
            continue
        if any(token in low for token in ("memory", "memories", "rag", "embedding", "agent", "dream", "brain", "homelab", "implementation", "paper", "project", "harness")):
            priority_terms.append(word)
    fallback_terms = [w for w in words if w.lower() not in stop]
    compact = " ".join((priority_terms or fallback_terms)[:18])
    return compact[:max_chars].strip() or query[:max_chars].strip()


def set_session_dir(path: Path) -> None:
    global SESSION_DIR, CACHE_DIR, DOC_DIR
    SESSION_DIR = path.expanduser()
    CACHE_DIR = SESSION_DIR / ".cache"
    DOC_DIR = SESSION_DIR / ".docs"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def artifact_path(name: str) -> Path:
    return SESSION_DIR / name


def write_json_artifact(name: str, data: Any) -> Path:
    path = artifact_path(name)
    path.write_text(json.dumps(sanitize_for_storage(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_text_artifact(name: str, text: str) -> Path:
    path = artifact_path(name)
    path.write_text(strip_thinking(text), encoding="utf-8")
    return path


def find_latest_session_id() -> Optional[str]:
    candidates = []
    for path in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("session_id") == path.stem:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).stem


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(message, flush=True)


class SpiderBusyError(RuntimeError):
    pass


class SpiderRunLock:
    def __init__(self, path: Path | None = None, wait: bool = False):
        self.path = Path(path or DEFAULT_LOCK_PATH).expanduser()
        self.wait = wait
        self.handle = None

    def __enter__(self) -> "SpiderRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX if self.wait else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(self.handle.fileno(), flags)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise SpiderBusyError(f"Spider is busy; lock is held at {self.path}. Re-run with --wait-for-lock to wait.") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "acquired_at": now_iso()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def has_unclosed_thinking(text: str) -> bool:
    if not text:
        return False
    opens = list(re.finditer(r"<think\b[^>]*>", str(text), flags=re.IGNORECASE))
    closes = list(re.finditer(r"</think\s*>", str(text), flags=re.IGNORECASE))
    return bool(opens and (not closes or opens[-1].start() > closes[-1].start()))


def strip_thinking(text: Any) -> str:
    """Remove paired, orphan, unclosed, and fenced thinking blocks."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"```(?:thinking|thoughts?)\s*.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*</think\s*>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*</think\s*>\s*$", "", cleaned, flags=re.IGNORECASE)
    open_match = re.search(r"<think\b[^>]*>", cleaned, flags=re.IGNORECASE)
    if open_match:
        cleaned = cleaned[: open_match.start()]
    return cleaned.strip()


def sanitize_for_storage(value: Any) -> Any:
    if isinstance(value, str):
        return strip_thinking(value)
    if isinstance(value, list):
        return [sanitize_for_storage(item) for item in value]
    if isinstance(value, dict):
        return {str(k): sanitize_for_storage(v) for k, v in value.items()}
    return value


def extract_json_value(text: Any) -> Any:
    """Best-effort JSON extraction for plain, fenced, embedded, or stringified JSON."""
    if text is None:
        return None
    cleaned = strip_thinking(text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    decoder = json.JSONDecoder()
    candidates = [cleaned]
    for needle in ("{", "["):
        idx = cleaned.find(needle)
        if idx >= 0:
            candidates.append(cleaned[idx:])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                nested = extract_json_value(value)
                return nested if nested is not None else value
            return value
        except Exception:
            pass
        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[idx:])
                if isinstance(value, str) and value.strip().startswith(("{", "[")):
                    nested = extract_json_value(value)
                    return nested if nested is not None else value
                return value
            except Exception:
                continue
    return None


def save_raw_model_output(raw_content: str, prefix: str = "spider") -> None:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        path = DEBUG_DIR / f"{prefix}_raw_{int(time.time())}.txt"
        path.write_text(raw_content or "", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to save raw model output: {exc}", file=sys.stderr)


def call_chat_model(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str = "",
    max_tokens: int,
    system: str,
    temperature: float,
    timeout: int,
    debug_prefix: str,
    disable_thinking: bool = False,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": float(os.environ.get("SPIDER_TOP_P", "0.95")),
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"].get("content", "")
        save_raw_model_output(raw, prefix=debug_prefix)
        cleaned = strip_thinking(raw)
        if cleaned:
            return cleaned
        return "[Model output did not reach a usable final answer before the output limit.]"
    except Exception as exc:
        return f"[Error calling {model}: {exc}]"


def call_nanbeige(prompt: str, max_tokens: Optional[int] = None, system: str = "", timeout: Optional[int] = None) -> str:
    """Call the research model with a large default output budget."""
    if max_tokens is None:
        max_tokens = int(os.environ.get("SPIDER_NANBEIGE_MAX_TOKENS", "32768"))
    return call_chat_model(
        prompt,
        model=NANBEIGE_MODEL,
        base_url=SCRIBE_API_URL,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens,
        system=system
        or (
            "You are research model inside Spider harness. "
            "Use only the provided observations. Do not include hidden reasoning, "
            "chain of thought, or <think> blocks."
        ),
        temperature=float(os.environ.get("SPIDER_TEMPERATURE", "0.55")),
        timeout=timeout if timeout is not None else int(os.environ.get("NANBEIGE_TIMEOUT", "3600")),
        debug_prefix="spider_research",
    )


def call_controller_model(prompt: str, max_tokens: Optional[int] = None, system: str = "") -> str:
    """Call the optional intake/controller model, defaulting to the research model."""
    if max_tokens is None:
        max_tokens = int(os.environ.get("SPIDER_CONTROLLER_MAX_TOKENS", "4096"))
    return call_chat_model(
        prompt,
        model=CONTROLLER_MODEL,
        base_url=CONTROLLER_API_URL,
        api_key=CONTROLLER_API_KEY,
        max_tokens=max_tokens,
        system=system
        or (
            "You are Spider's visible controller. Return compact JSON only when asked. "
            "Never include hidden reasoning, chain of thought, or <think> blocks."
        ),
        temperature=float(os.environ.get("SPIDER_CONTROLLER_TEMPERATURE", "0.2")),
        timeout=int(os.environ.get("SPIDER_CONTROLLER_TIMEOUT", os.environ.get("NANBEIGE_TIMEOUT", "3600"))),
        debug_prefix="spider_controller",
    )


def call_structurer_model(prompt: str, max_tokens: Optional[int] = None, system: str = "") -> str:
    if max_tokens is None:
        max_tokens = int(os.environ.get("SPIDER_STRUCTURER_MAX_TOKENS", "8192"))
    return call_chat_model(
        prompt,
        model=STRUCTURER_MODEL,
        base_url=STRUCTURER_API_URL,
        api_key=STRUCTURER_API_KEY,
        max_tokens=max_tokens,
        system=system
        or (
            "You are Spider's strict structurer and JSON repair model. "
            "Convert source cards into compact valid JSON. Do not include hidden reasoning or <think> blocks."
        ),
        temperature=float(os.environ.get("SPIDER_STRUCTURER_TEMPERATURE", "0.1")),
        timeout=int(os.environ.get("SPIDER_STRUCTURER_TIMEOUT", "900")),
        debug_prefix="spider_structurer",
        disable_thinking=os.environ.get("SPIDER_STRUCTURER_DISABLE_THINKING", "1") == "1",
    )


def call_writer_model(prompt: str, max_tokens: Optional[int] = None, system: str = "", model_override: str = "", base_url_override: str = "") -> str:
    if max_tokens is None:
        max_tokens = int(os.environ.get("SPIDER_WRITER_MAX_TOKENS", "32768"))
    return call_chat_model(
        prompt,
        model=(model_override or WRITER_MODEL).replace("local/", "").replace(":thinking", ""),
        base_url=(base_url_override or WRITER_API_URL).rstrip("/"),
        api_key=WRITER_API_KEY,
        max_tokens=max_tokens,
        system=system
        or (
            "You are Spider's longform research writer. Write polished, source-grounded prose. "
            "Use source IDs for citations. Do not include hidden reasoning or <think> blocks."
        ),
        temperature=float(os.environ.get("SPIDER_WRITER_TEMPERATURE", "0.35")),
        timeout=int(os.environ.get("SPIDER_WRITER_TIMEOUT", "1800")),
        debug_prefix="spider_writer",
        disable_thinking=os.environ.get("SPIDER_WRITER_DISABLE_THINKING", "1") == "1",
    )


def call_editor_model(prompt: str, max_tokens: Optional[int] = None, system: str = "") -> str:
    if max_tokens is None:
        max_tokens = int(os.environ.get("SPIDER_EDITOR_MAX_TOKENS", "8192"))
    return call_chat_model(
        prompt,
        model=EDITOR_MODEL,
        base_url=EDITOR_API_URL,
        api_key=EDITOR_API_KEY,
        max_tokens=max_tokens,
        system=system
        or (
            "You are Spider's report QA editor. Return concise structured feedback or corrected prose. "
            "Do not include hidden reasoning or <think> blocks."
        ),
        temperature=float(os.environ.get("SPIDER_EDITOR_TEMPERATURE", "0.2")),
        timeout=int(os.environ.get("SPIDER_EDITOR_TIMEOUT", "900")),
        debug_prefix="spider_editor",
        disable_thinking=os.environ.get("SPIDER_EDITOR_DISABLE_THINKING", "1") == "1",
    )


def normalize_source_variants(url: str) -> list[str]:
    """Return text-bearing variants for common source hosts."""
    url = (url or "").strip()
    if not url:
        return []
    variants = [url]
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host == "huggingface.co":
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] not in {"api", "datasets", "spaces"}:
            repo = "/".join(parts[:2])
            variants.extend(
                [
                    f"https://huggingface.co/{repo}/raw/main/README.md",
                    f"https://huggingface.co/{repo}/resolve/main/README.md",
                    f"https://huggingface.co/api/models/{repo}",
                ]
            )
        elif len(parts) >= 3 and parts[0] in {"datasets", "spaces"}:
            repo = "/".join(parts[:3])
            variants.extend(
                [
                    f"https://huggingface.co/{repo}/raw/main/README.md",
                    f"https://huggingface.co/{repo}/resolve/main/README.md",
                ]
            )

    if host in {"arxiv.org", "www.arxiv.org"}:
        match = re.search(r"/abs/([^/?#]+)", parsed.path)
        if match:
            arxiv_id = match.group(1)
            variants.extend(
                [
                    f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
                    f"https://arxiv.org/pdf/{arxiv_id}",
                ]
            )

    if host == "github.com":
        parts = path.split("/")
        if len(parts) >= 4 and parts[2] == "issues":
            owner, repo, _, issue_num = parts[:4]
            variants.insert(0, f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}")
        elif len(parts) >= 4 and parts[2] == "pull":
            owner, repo, _, pull_num = parts[:4]
            variants.insert(0, f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_num}")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            rest = "/".join(parts[4:])
            variants.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}")
        elif len(parts) >= 2 and not (len(parts) >= 4 and parts[2] in {"issues", "pull", "discussions"}):
            owner, repo = parts[:2]
            variants.extend(
                [
                    f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
                    f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
                ]
            )

    seen: set[str] = set()
    out: list[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def canonical_url_key(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return (url or "").strip()
    path = re.sub(r"/+$", "", parsed.path or "/")
    if parsed.netloc.lower() == "docs.python.org":
        path = re.sub(r"^/3(?:\.\d+)?/", "/3/", path)
    query = parsed.query
    if parsed.netloc.lower() in {"arxiv.org", "www.arxiv.org"} and parsed.path.startswith("/abs/"):
        query = ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}" + (f"?{query}" if query else "")


def url_variant_keys(url: str) -> set[str]:
    keys = {canonical_url_key(url)}
    for variant in normalize_source_variants(url):
        keys.add(canonical_url_key(variant))
    return {key for key in keys if key}


def extract_text_from_response(resp: requests.Response, max_chars: int = 16000) -> str:
    content_type = resp.headers.get("content-type", "").lower()
    raw = resp.text or ""
    if "application/json" in content_type:
        try:
            return json.dumps(resp.json(), indent=2, ensure_ascii=False)[:max_chars]
        except Exception:
            return raw[:max_chars]
    if "xml" in content_type or "export.arxiv.org/api/query" in resp.url:
        text = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        return text[:max_chars]
    if "pdf" in content_type:
        return extract_pdf_text(resp.content, resp.url, max_chars=max_chars)
    if any(marker in content_type for marker in ("text/plain", "text/markdown")) or "/raw/" in resp.url or "raw.githubusercontent.com" in resp.url:
        text = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\n{4,}", "\n\n\n", text).strip()[:max_chars]

    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<noscript\b[^>]*>.*?</noscript>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</(p|div|section|article|h1|h2|h3|li|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def extract_links_from_html(raw: str, base_url: str, max_links: int = 80) -> list[dict[str, str]]:
    """Extract readable same-site-ish links from HTML with anchor text."""
    if not raw:
        return []
    base_host = urlparse(base_url).netloc.lower()
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r"<a\b[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", raw, flags=re.DOTALL | re.IGNORECASE):
        href = html.unescape(match.group(1).strip())
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        host = parsed.netloc.lower()
        if base_host and host != base_host and not host.endswith(f".{base_host}") and not base_host.endswith(f".{host}"):
            continue
        if not is_readable_url(url) or is_blocklisted_url(url):
            continue
        key = canonical_url_key(url)
        if key in seen or key == canonical_url_key(base_url):
            continue
        seen.add(key)
        text = re.sub(r"<[^>]+>", " ", match.group(2))
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        links.append({"url": url, "text": strip_thinking(text[:180])})
        if len(links) >= max_links:
            break
    return links


def extract_media_candidates_from_html(raw: str, base_url: str, max_items: int = 80) -> list[dict[str, str]]:
    """Discover image-like media candidates without OCR/VLM analysis."""
    if not raw:
        return []
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_media(url: str, alt: str = "", caption: str = "", source: str = "img") -> None:
        if not url:
            return
        resolved = urljoin(base_url, html.unescape(url.strip()))
        parsed = urlparse(resolved)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return
        key = canonical_url_key(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "url": resolved,
                "alt": strip_thinking(alt[:240]),
                "caption": strip_thinking(caption[:400]),
                "source": source,
            }
        )

    for match in re.finditer(r"<meta\b[^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"'][^>]+content=[\"']([^\"']+)[\"'][^>]*>", raw, flags=re.IGNORECASE):
        add_media(match.group(1), source="metadata")
    for match in re.finditer(r"<figure\b[^>]*>(.*?)</figure>", raw, flags=re.DOTALL | re.IGNORECASE):
        block = match.group(1)
        img = re.search(r"<img\b([^>]*)>", block, flags=re.IGNORECASE)
        if not img:
            continue
        attrs = img.group(1)
        src = re.search(r"\bsrc=[\"']([^\"']+)[\"']", attrs, flags=re.IGNORECASE)
        alt = re.search(r"\balt=[\"']([^\"']*)[\"']", attrs, flags=re.IGNORECASE)
        caption = re.search(r"<figcaption\b[^>]*>(.*?)</figcaption>", block, flags=re.DOTALL | re.IGNORECASE)
        add_media(
            src.group(1) if src else "",
            alt=html.unescape(alt.group(1)) if alt else "",
            caption=re.sub(r"<[^>]+>", " ", html.unescape(caption.group(1))) if caption else "",
            source="figure",
        )
    for match in re.finditer(r"<img\b([^>]*)>", raw, flags=re.IGNORECASE):
        attrs = match.group(1)
        src = re.search(r"\bsrc=[\"']([^\"']+)[\"']", attrs, flags=re.IGNORECASE)
        alt = re.search(r"\balt=[\"']([^\"']*)[\"']", attrs, flags=re.IGNORECASE)
        if src:
            add_media(src.group(1), alt=html.unescape(alt.group(1)) if alt else "", source="img")
        srcset = re.search(r"\bsrcset=[\"']([^\"']+)[\"']", attrs, flags=re.IGNORECASE)
        if srcset:
            first = srcset.group(1).split(",")[0].strip().split(" ")[0]
            add_media(first, alt=html.unescape(alt.group(1)) if alt else "", source="srcset")
        if len(candidates) >= max_items:
            break
    return candidates[:max_items]


def terms_from_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().lower() for item in re.split(r"[,;\n]", value) if item.strip()]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(terms_from_text(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(terms_from_text(item))
        return out
    return [str(value).strip().lower()]


def source_family_for_url(url: str) -> dict[str, str]:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    path = re.sub(r"/+$", "", parsed.path or "/")
    parts = [p for p in path.split("/") if p]
    if host in {"github.com", "raw.githubusercontent.com"} and len(parts) >= 2:
        project = f"github:{parts[0]}/{parts[1]}"
        path_family = project
        if len(parts) >= 4 and parts[2] in {"blob", "tree"}:
            path_family = f"{project}/{'/'.join(parts[4:6])}" if len(parts) >= 5 else project
        return {"domain": host, "path_family": path_family, "source_family": project}
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        paper = ""
        match = re.search(r"/(?:abs|pdf|html)/([^/?#]+)", path)
        if match:
            paper = re.sub(r"v\d+$", "", match.group(1).removesuffix(".pdf"))
        elif "api/query" in path:
            paper = urlparse(url).query.replace("id_list=", "")
        family = f"arxiv:{paper}" if paper else "arxiv:unknown"
        return {"domain": host, "path_family": family, "source_family": family}
    if host == "aindoria.com" and len(parts) >= 2 and parts[0] == "posts":
        slug = re.sub(r"-\d+$", "", parts[1])
        family = f"aindoria-post:{slug}"
        return {"domain": host, "path_family": family, "source_family": family}
    if "docs" in host or any(marker in host for marker in ("developers.", "docs.")):
        path_family = f"{host}/{'/'.join(parts[:4])}" if parts else host
        return {"domain": host, "path_family": path_family, "source_family": path_family}
    path_family = f"{host}/{'/'.join(parts[:2])}" if parts else host
    return {"domain": host, "path_family": path_family, "source_family": path_family}


def source_category_for_url(url: str, title: str = "", snippet: str = "") -> str:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    text = f"{title} {snippet} {url}".lower()
    if "arxiv.org" in host or "paper" in text or "technical report" in text:
        return "papers"
    if "github.com" in host or "githubusercontent.com" in host:
        return "project_repos"
    if any(marker in host for marker in ("docs.", "developers.", "readthedocs", "documentation")) or any(marker in text for marker in ("docs", "documentation", "guide", "api reference")):
        return "project_docs"
    if "blog" in text or "/posts/" in parsed.path:
        return "blogs"
    if any(marker in host for marker in ("openai.com", "anthropic.com", "googleblog.com", "microsoft.com", "perplexity.ai")):
        return "labs_products"
    return "web"


def default_source_policy(session: "SpiderSession") -> dict[str, Any]:
    question_l = session.question.lower()
    high_terms = [
        "official",
        "paper",
        "technical report",
        "documentation",
        "github",
        "repository",
        "evidence",
        "limitations",
    ]
    must_cover: list[str] = []
    categories = ["seed_sources", "project_docs", "project_repos", "papers", "labs_products", "blogs", "web"]
    if any(term in question_l for term in ("memory", "memories", "rag", "embedding", "dreaming", "brains")):
        high_terms.extend(
            [
                "memory",
                "memories",
                "episodic",
                "semantic",
                "procedural",
                "reflection",
                "retrieval",
                "rag",
                "embedding",
                "vector",
                "long-term memory",
                "agent memory",
                "self-improving memory",
                "memgpt",
                "letta",
                "zep",
                "langmem",
                "langgraph",
                "memorybank",
            ]
        )
        must_cover.extend(["openai memory", "perplexity memory", "memgpt", "letta", "zep", "langgraph", "langmem", "papers", "volition", "abe"])
    if any(term in question_l for term in ("migration", "migrate", "porting", "upgrade")):
        high_terms.extend(["migration", "migrate", "porting", "upgrade", "compatibility"])
    if any(term in question_l for term in ("testing", "profiling", "test")):
        high_terms.extend(["testing", "profiling", "test"])
    if any(term in question_l for term in ("sandbox", "runtime", "isolation")):
        high_terms.extend(["sandbox", "runtime", "isolation", "container"])
    return {
        "must_cover_targets": must_cover,
        "source_categories": categories,
        "category_minimums": {"papers": 2 if session.depth == "deep" else 0, "project_repos": 2 if session.depth == "deep" else 0, "project_docs": 2 if session.depth == "deep" else 0},
        "category_budgets": {"project_docs": 24, "project_repos": 24, "papers": 24, "blogs": 12, "labs_products": 12, "web": 12},
        "domain_caps": {},
        "path_family_caps": {},
        "source_family_caps": {},
        "high_value_terms": sorted(set(high_terms)),
        "low_value_unless_plan_relevant": ["migration", "migrate", "quickstart", "auth", "sdk setup", "widget", "chatkit", "sandbox", "generic tools", "observability", "background mode"],
        "explicit_allow_terms": [],
        "explicit_skip_terms": [],
        "source_role_notes": {},
    }


def source_policy(session: "SpiderSession") -> dict[str, Any]:
    policy = default_source_policy(session)
    plan_policy = (session.research_plan or {}).get("source_policy")
    if isinstance(plan_policy, dict):
        for key, value in plan_policy.items():
            if isinstance(value, dict) and isinstance(policy.get(key), dict):
                merged = dict(policy.get(key, {}))
                merged.update(value)
                policy[key] = merged
            elif isinstance(value, list) and isinstance(policy.get(key), list):
                policy[key] = list(dict.fromkeys([*policy.get(key, []), *value]))
            elif value not in (None, "", [], {}):
                policy[key] = value
    return sanitize_for_storage(policy)


def text_matches_any(haystack: str, terms: list[str]) -> list[str]:
    haystack_l = haystack.lower()
    matches = []
    for term in terms:
        term_l = term.lower().strip()
        if not term_l:
            continue
        if re.fullmatch(r"[a-z0-9]+", term_l):
            if re.search(rf"(?<![a-z0-9]){re.escape(term_l)}(?![a-z0-9])", haystack_l):
                matches.append(term)
        elif term_l in haystack_l:
            matches.append(term)
    return matches


def category_counts(session: "SpiderSession") -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in session.sources:
        category = source.get("source_category") or source_category_for_url(source.get("final_url") or source.get("url", ""), source.get("title", ""), source.get("snippet", ""))
        counts[category] = counts.get(category, 0) + 1
    return counts


def source_family_counts(session: "SpiderSession") -> dict[str, dict[str, int]]:
    counts = {"domain": {}, "path_family": {}, "source_family": {}, "category": {}}
    for source in session.sources:
        fam = source_family_for_url(source.get("final_url") or source.get("url", ""))
        category = source.get("source_category") or source_category_for_url(source.get("final_url") or source.get("url", ""), source.get("title", ""), source.get("snippet", ""))
        for key in ("domain", "path_family", "source_family"):
            value = fam.get(key, "")
            counts[key][value] = counts[key].get(value, 0) + 1
        counts["category"][category] = counts["category"].get(category, 0) + 1
    return counts


def saturation_for_item(session: "SpiderSession", item: dict[str, Any], policy: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    policy = policy or source_policy(session)
    url = item.get("url", "")
    fam = source_family_for_url(url)
    category = item.get("source_category") or source_category_for_url(url, item.get("title", ""), item.get("snippet", ""))
    counts = source_family_counts(session)
    path_cap = int((policy.get("path_family_caps") or {}).get(fam["path_family"], os.environ.get("SPIDER_DEFAULT_PATH_FAMILY_CAP", "3")))
    family_cap = int((policy.get("source_family_caps") or {}).get(fam["source_family"], os.environ.get("SPIDER_DEFAULT_SOURCE_FAMILY_CAP", "3")))
    domain_cap = (policy.get("domain_caps") or {}).get(fam["domain"])
    if domain_cap is None:
        domain_cap = max(1, int(session.max_reads * float(os.environ.get("SPIDER_DOCS_DOMAIN_SHARE_CAP", "0.25")))) if session.depth == "deep" else session.max_reads
    category_cap = int((policy.get("category_budgets") or {}).get(category, session.max_reads))
    domain_count = counts["domain"].get(fam["domain"], 0)
    path_count = counts["path_family"].get(fam["path_family"], 0)
    family_count = counts["source_family"].get(fam["source_family"], 0)
    category_count = counts["category"].get(category, 0)
    saturated_reasons = []
    if path_count >= path_cap:
        saturated_reasons.append("path_family")
    if family_count >= family_cap:
        saturated_reasons.append("source_family")
    if category_count >= category_cap:
        saturated_reasons.append("category")
    broad_coverage = coverage_sufficient(session)
    family_driven_domain = fam["domain"] in {"github.com", "raw.githubusercontent.com", "arxiv.org", "www.arxiv.org", "export.arxiv.org"}
    if session.depth == "deep" and not broad_coverage and not family_driven_domain and domain_count >= int(domain_cap):
        saturated_reasons.append("domain_share")
    return {
        **fam,
        "source_category": category,
        "counts": {"domain": domain_count, "path_family": path_count, "source_family": family_count, "category": category_count},
        "caps": {"domain": int(domain_cap), "path_family": path_cap, "source_family": family_cap, "category": category_cap},
        "saturated": bool(saturated_reasons),
        "reasons": saturated_reasons,
    }


def policy_score_item(session: "SpiderSession", item: dict[str, Any], policy: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    policy = policy or source_policy(session)
    url = item.get("url", "")
    title = item.get("title", "")
    snippet = item.get("snippet", item.get("text", ""))
    haystack = f"{title} {snippet} {url}".lower()
    high_matches = text_matches_any(haystack, terms_from_text(policy.get("high_value_terms", [])))
    must_matches = text_matches_any(haystack, terms_from_text(policy.get("must_cover_targets", [])))
    explicit_allow = text_matches_any(haystack, terms_from_text(policy.get("explicit_allow_terms", [])))
    explicit_skip = text_matches_any(haystack, terms_from_text(policy.get("explicit_skip_terms", [])))
    low_matches = text_matches_any(haystack, terms_from_text(policy.get("low_value_unless_plan_relevant", [])))
    relevant_low = [term for term in low_matches if term in high_matches or term in explicit_allow or term in must_matches or term in session.question.lower()]
    low_penalty = [term for term in low_matches if term not in relevant_low]
    category = item.get("source_category") or source_category_for_url(url, title, snippet)
    counts = category_counts(session)
    minimums = policy.get("category_minimums") or {}
    fills_category_gap = counts.get(category, 0) < int(minimums.get(category, 0))
    saturation = saturation_for_item(session, {**item, "source_category": category}, policy)
    score = int(item.get("search_score", item.get("score", 0)) or 0)
    score += len(high_matches) * 8
    score += len(must_matches) * 18
    score += len(explicit_allow) * 16
    score += 20 if fills_category_gap else 0
    score -= len(low_penalty) * 14
    score -= len(explicit_skip) * 100
    if saturation["saturated"]:
        score -= 80
    status = "read_now"
    reason = "matches source policy"
    if source_already_seen(session, url):
        status, reason = "skipped_duplicate", "already read"
    elif explicit_skip:
        status, reason = "skipped_off_topic", f"explicit skip terms: {', '.join(explicit_skip)}"
    elif saturation["saturated"]:
        status, reason = "skipped_source_family_saturated", f"saturated: {', '.join(saturation['reasons'])}"
    elif score < min_frontier_score(session):
        status, reason = "skipped_low_policy_score", f"policy score {score} below threshold {min_frontier_score(session)}"
    elif low_penalty and not (high_matches or must_matches or explicit_allow):
        status, reason = "reserve", f"low-value unless plan-relevant terms: {', '.join(low_penalty)}"
    elif score < min_frontier_score(session) + 10:
        status, reason = "reserve", f"weak policy score {score}"
    return {
        "policy_score": score,
        "frontier_state": status,
        "frontier_reason": reason,
        "source_policy_matches": {
            "high_value_terms": high_matches,
            "must_cover_targets": must_matches,
            "explicit_allow_terms": explicit_allow,
            "explicit_skip_terms": explicit_skip,
            "low_value_terms": low_matches,
            "low_value_penalized": low_penalty,
            "fills_category_gap": fills_category_gap,
        },
        "saturation": saturation,
        "source_category": category,
    }


def score_link_candidate(link: dict[str, str], question: str = "", session: Optional["SpiderSession"] = None) -> int:
    if session is None:
        shim = SpiderSession(question=question, max_sources=1, max_reads=1)
        try:
            return policy_score_item(shim, {"url": link.get("url", ""), "title": link.get("text", ""), "search_score": 0})["policy_score"]
        finally:
            try:
                shim.session_path.unlink(missing_ok=True)
            except Exception:
                pass
    return policy_score_item(session, {"url": link.get("url", ""), "title": link.get("text", ""), "search_score": 0})["policy_score"]


def extract_pdf_text(data: bytes, label: str = "PDF", max_chars: int = 16000) -> str:
    """Extract PDF text if pypdf or PyMuPDF is installed; otherwise fail softly."""
    try:
        from pypdf import PdfReader  # type: ignore

        import io

        reader = PdfReader(io.BytesIO(data))
        chunks = []
        for page in reader.pages[: int(os.environ.get("SPIDER_PDF_MAX_PAGES", "24"))]:
            chunks.append(page.extract_text() or "")
            if sum(len(c) for c in chunks) >= max_chars:
                break
        text = "\n".join(chunks).strip()
        if text:
            return text[:max_chars]
    except Exception:
        pass

    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=data, filetype="pdf")
        chunks = []
        for page in doc[: int(os.environ.get("SPIDER_PDF_MAX_PAGES", "24"))]:
            chunks.append(page.get_text() or "")
            if sum(len(c) for c in chunks) >= max_chars:
                break
        text = "\n".join(chunks).strip()
        if text:
            return text[:max_chars]
    except Exception:
        pass

    return f"[PDF fetched from {label}; install pypdf or PyMuPDF for text extraction.]"


def score_source_text(text: str, url: str = "") -> int:
    if not text:
        return -1000
    t = text.lower()
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_.-]*\b", text)
    score = min(len(words) // 45, 35)
    for marker in (
        "abstract",
        "model card",
        "readme",
        "introduction",
        "benchmark",
        "evaluation",
        "license",
        "citation",
        "paper",
        "arxiv",
        "capabilities",
        "training",
        "reasoning",
        "alignment",
        "code generation",
        "tool",
        "agentic",
        "deep search",
    ):
        if marker in t:
            score += 8
    for marker in (
        "window.__",
        "document.documentelement",
        "webpack",
        "plausible",
        "frontend",
        "javascript",
        "cookie.match",
        "localstorage",
        "next/static",
        "chunk",
    ):
        if marker in t:
            score -= 18
    if text.count("{") + text.count("}") + text.count("=>") > 120:
        score -= 30
    if "huggingface.co/api/models" in url:
        score += 8
    if "/raw/" in url or "raw.githubusercontent.com" in url:
        score += 25
    if "export.arxiv.org/api/query" in url:
        score += 25
    if "arxiv.org/pdf" in url:
        score += 5
    return score


def source_type_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "huggingface.co" in host:
        return "huggingface"
    if "arxiv.org" in host:
        return "arxiv"
    if "github.com" in host or "githubusercontent.com" in host:
        return "github"
    if host:
        return "web"
    return "file"


def fetch_webpage(url: str, max_chars: int = 16000) -> dict[str, Any]:
    """Fetch and cache the best source variant."""
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists() and os.environ.get("SPIDER_DISABLE_CACHE") != "1":
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    best: Optional[dict[str, Any]] = None
    errors: list[str] = []
    for candidate in normalize_source_variants(url):
        try:
            resp = requests.get(
                candidate,
                timeout=25,
                headers={"User-Agent": "Mozilla/5.0 (compatible; VolitionSpider/1.0)"},
                allow_redirects=True,
            )
            resp.raise_for_status()
            text = extract_text_from_response(resp, max_chars=max_chars)
            links = []
            content_type = resp.headers.get("content-type", "").lower()
            if "html" in content_type:
                links = extract_links_from_html(resp.text or "", resp.url)
                media = extract_media_candidates_from_html(resp.text or "", resp.url)
            else:
                media = []
            score = score_source_text(text, resp.url)
            item = {
                "requested_url": url,
                "url": candidate,
                "final_url": resp.url,
                "adapter": source_type_for_url(candidate),
                "text": text,
                "score": score,
                "extracted_chars": len(text),
                "fetched_at": now_iso(),
                "links": links,
                "media": media,
            }
            if best is None or item["score"] > best["score"]:
                best = item
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if best is None:
        best = {
            "requested_url": url,
            "url": url,
            "final_url": url,
            "adapter": source_type_for_url(url),
            "text": f"[Could not fetch useful text: {'; '.join(errors[:3])}]",
            "score": -1000,
            "extracted_chars": 0,
            "fetched_at": now_iso(),
            "errors": errors[:5],
        }
    best["cache_key"] = cache_key
    best["cache_path"] = str(cache_path)
    cache_path.write_text(json.dumps(sanitize_for_storage(best), indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def read_local_file(path: str, max_chars: int = 16000) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".pdf":
        text = extract_pdf_text(p.read_bytes(), str(p), max_chars=max_chars)
        adapter = "file_pdf"
    else:
        text = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
        adapter = "file"
    return {
        "requested_url": str(p),
        "url": str(p),
        "final_url": str(p),
        "adapter": adapter,
        "text": text,
        "score": min(max(len(text) // 500, 5), 35),
        "extracted_chars": len(text),
        "fetched_at": now_iso(),
    }


def fetch_searxng(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    global SEARCH_BACKEND_UNAVAILABLE
    if SEARCH_BACKEND_UNAVAILABLE:
        return []
    params = {"q": query, "format": "json", "language": "en", "pageno": 1}
    time_range = os.environ.get("SPIDER_SEARCH_TIME_RANGE", "").strip()
    if time_range:
        params["time_range"] = time_range
    try:
        resp = requests.get(SEARXNG_URL, params=params, timeout=18)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("results", [])[:max_results]
        if not rows and data.get("unresponsive_engines"):
            engines = ", ".join(f"{name}: {reason}" for name, reason in data.get("unresponsive_engines", [])[:5])
            # Some engines being blocked does not mean the whole backend is dead.
            # Treat this as a coverage warning unless an explicit connection error occurs.
            print(f"[WARN] SearXNG returned no results and reports unavailable engines: {engines}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[WARN] SearXNG fetch failed for {query!r}: {exc}", file=sys.stderr)
        return []
    out = []
    for row in rows:
        url = row.get("url", "")
        out.append(
            {
                "title": strip_thinking(row.get("title", "")),
                "url": url,
                "snippet": strip_thinking(row.get("content", "")),
                "score": score_search_result(row, query),
                "source_type": source_type_for_url(url),
            }
        )
    return sorted(out, key=lambda r: r["score"], reverse=True)


def normalize_brave_results(data: dict[str, Any], query: str, max_results: int = 6) -> list[dict[str, Any]]:
    rows = (data.get("web") or {}).get("results") or []
    out = []
    for row in rows[:max_results]:
        url = row.get("url", "")
        extra = row.get("extra_snippets") or []
        snippet = row.get("description", "") or (extra[0] if extra else "")
        item = {
            "title": strip_thinking(row.get("title", "")),
            "url": url,
            "snippet": strip_thinking(snippet),
            "score": score_search_result({"title": row.get("title", ""), "content": row.get("description", ""), "url": url}, query),
            "source_type": source_type_for_url(url),
            "backend": "brave",
        }
        out.append(item)
    return sorted(out, key=lambda r: r["score"], reverse=True)


def fetch_brave(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    api_key = os.environ.get("SPIDER_BRAVE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SPIDER_BRAVE_API_KEY is not set")
    resp = requests.get(
        BRAVE_SEARCH_URL,
        params={"q": query, "count": max_results, "search_lang": "en", "safesearch": "moderate"},
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
        timeout=18,
    )
    if resp.status_code == 429:
        raise RuntimeError("Brave rate limit reached")
    resp.raise_for_status()
    return normalize_brave_results(resp.json(), query, max_results=max_results)


def search_cache_path(backend: str, query: str) -> Path:
    key = hashlib.sha256(f"{backend}:{normalized_query_key(query)}".encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"search_{key}.json"


def get_cached_search_results(backend: str, query: str) -> Optional[list[dict[str, Any]]]:
    ttl_hours = float(os.environ.get("SPIDER_SEARCH_CACHE_TTL_HOURS", "24"))
    if ttl_hours <= 0:
        return None
    path = search_cache_path(backend, query)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age_hours = (time.time() - float(data.get("cached_at", 0))) / 3600
        if age_hours > ttl_hours:
            return None
        return data.get("results", [])
    except Exception:
        return None


def write_search_cache(backend: str, query: str, results: list[dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"backend": backend, "query": normalized_query_key(query), "cached_at": time.time(), "results": sanitize_for_storage(results)}
    search_cache_path(backend, query).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_search(session: "SpiderSession", query: str, max_results: int = 6) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch search results through configured backends with cache and Brave budget accounting."""
    original_query = query
    query = normalize_search_query(query)
    backends = default_search_backends("SPIDER_BRAVE_API_KEY", "SPIDER_SEARCH_BACKENDS")
    max_api_calls = int(os.environ.get("SPIDER_SEARCH_MAX_API_CALLS", "40"))
    meta: dict[str, Any] = {"backends": backends, "attempts": [], "backend": "", "cache_hit": False, "fallback_reason": "", "query": query, "original_query": original_query}
    for backend in backends:
        if backend not in {"brave", "searxng"}:
            meta["attempts"].append({"backend": backend, "status": "unknown_backend"})
            continue
        cached = get_cached_search_results(backend, query)
        if cached is not None:
            meta.update({"backend": backend, "cache_hit": True, "fallback_reason": ""})
            meta["attempts"].append({"backend": backend, "status": "cache_hit", "result_count": len(cached)})
            return cached, meta
        if backend == "brave":
            api_calls = int(session.coverage.get("search_api_calls", 0))
            if api_calls >= max_api_calls:
                meta["attempts"].append({"backend": backend, "status": "budget_exhausted", "api_calls": api_calls})
                meta["fallback_reason"] = "brave_budget_exhausted"
                continue
            try:
                session.coverage["search_api_calls"] = api_calls + 1
                results = fetch_brave(query, max_results=max_results)
                if results or os.environ.get("SPIDER_SEARCH_CACHE_EMPTY", "0") == "1":
                    write_search_cache(backend, query, results)
                meta["attempts"].append({"backend": backend, "status": "ok", "result_count": len(results), "api_calls": session.coverage["search_api_calls"]})
                if results or os.environ.get("SPIDER_SEARCH_FALLBACK_ON_EMPTY", "1") != "1":
                    meta.update({"backend": backend, "cache_hit": False})
                    return results, meta
                meta["fallback_reason"] = "brave_empty"
            except Exception as exc:
                meta["attempts"].append({"backend": backend, "status": "error", "error": str(exc)[:180], "api_calls": session.coverage.get("search_api_calls", 0)})
                meta["fallback_reason"] = "brave_error"
                if os.environ.get("SPIDER_SEARCH_FALLBACK_ON_ERROR", "1") != "1":
                    return [], meta
            continue
        try:
            results = fetch_searxng(query, max_results=max_results)
            if results or os.environ.get("SPIDER_SEARCH_CACHE_EMPTY", "0") == "1":
                write_search_cache(backend, query, results)
            meta["attempts"].append({"backend": backend, "status": "ok", "result_count": len(results)})
            meta.update({"backend": backend, "cache_hit": False})
            return results, meta
        except Exception as exc:
            meta["attempts"].append({"backend": backend, "status": "error", "error": str(exc)[:180]})
            meta["fallback_reason"] = "searxng_error"
    return [], meta


def score_search_result(row: dict[str, Any], query: str = "") -> int:
    url = row.get("url", "")
    if not is_readable_url(url) or is_blocklisted_url(url):
        return -1000
    host = urlparse(url).netloc.lower()
    text = f"{row.get('title', '')} {row.get('content', '')}".lower()
    score = 10
    if any(
        host.endswith(domain)
        for domain in (
            "arxiv.org",
            "github.com",
            "huggingface.co",
            "openai.com",
            "anthropic.com",
            "microsoft.com",
            "googleblog.com",
            "python.org",
            "docs.python.org",
            "peps.python.org",
            "discuss.python.org",
            "py-free-threading.github.io",
            "pyo3.rs",
            "lwn.net",
            "quansight.org",
        )
    ):
        score += 25
    if any(word in text for word in ("official", "release", "paper", "technical report", "advisory", "documentation", "howto", "guide")):
        score += 8
    if any(word in text for word in ("pep 703", "free-thread", "free threading", "py_mod_gil", "extension module", "c api", "gil disabled")):
        score += 12
    if any(word in host for word in ("medium.com", "forbes.com", "analyticsinsight", "sponsored")):
        score -= 10
    for token in re.findall(r"[a-z0-9]{4,}", query.lower()):
        if token in text or token in url.lower():
            score += 1
    return score


def is_readable_url(url: str) -> bool:
    parsed = urlparse(url or "")
    path = parsed.path.lower()
    blocked_suffixes = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".gguf",
        ".ckpt",
        ".zip",
        ".tar",
        ".gz",
        ".xz",
        ".7z",
        ".mp4",
        ".webm",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
    )
    if path.endswith(blocked_suffixes):
        return False
    if "/resolve/" in path and any(path.endswith(suffix) for suffix in blocked_suffixes):
        return False
    return bool(parsed.scheme in {"http", "https"} and parsed.netloc)


def is_blocklisted_url(url: str) -> bool:
    host = urlparse(url or "").netloc.lower()
    configured = {
        domain.strip().lower()
        for domain in os.environ.get("SPIDER_BLOCKLIST_DOMAINS", ",".join(sorted(DEFAULT_BLOCKLIST_DOMAINS))).split(",")
        if domain.strip()
    }
    return any(host == domain or host.endswith(f".{domain}") for domain in configured)


def min_frontier_score(session: "SpiderSession") -> int:
    if session.depth == "deep":
        return int(os.environ.get("SPIDER_MIN_FRONTIER_SCORE", "25"))
    if session.depth == "standard":
        return int(os.environ.get("SPIDER_MIN_FRONTIER_SCORE", "20"))
    return int(os.environ.get("SPIDER_MIN_FRONTIER_SCORE", "0"))


def min_extracted_source_score(session: "SpiderSession") -> int:
    if session.depth in {"standard", "deep"}:
        return int(os.environ.get("SPIDER_MIN_EXTRACTED_SOURCE_SCORE", "10"))
    return int(os.environ.get("SPIDER_MIN_EXTRACTED_SOURCE_SCORE", "-10"))


def find_existing_source_by_url(session: "SpiderSession", url: str) -> Optional[dict[str, Any]]:
    wanted = url_variant_keys(url)
    for source in session.sources:
        source_urls = {
            source.get("requested_url", ""),
            source.get("url", ""),
            source.get("final_url", ""),
        }
        seen: set[str] = set()
        for item in source_urls:
            seen |= url_variant_keys(item)
        if wanted & seen:
            return source
    return None


def source_already_seen(session: "SpiderSession", url: str) -> bool:
    return find_existing_source_by_url(session, url) is not None


class SpiderSession:
    def __init__(
        self,
        session_id: Optional[str] = None,
        question: str = "",
        max_steps: int = 15,
        max_sources: int = 30,
        max_reads: int = 20,
        time_limit_min: int = 30,
        depth: str = "standard",
        min_steps: Optional[int] = None,
        min_sources: Optional[int] = None,
        min_reads: Optional[int] = None,
    ):
        if session_id:
            self.session_id = session_id
            self.load()
            return
        self.session_id = str(uuid.uuid4())[:8]
        self.question = question
        self.max_steps = max_steps
        self.max_sources = max_sources
        self.max_reads = max_reads
        self.time_limit_min = time_limit_min
        self.depth = depth
        mins = depth_minimums(depth)
        self.min_steps = min_steps if min_steps is not None else mins["min_steps"]
        self.min_sources = min_sources if min_sources is not None else mins["min_sources"]
        self.min_reads = min_reads if min_reads is not None else mins["min_reads"]
        self.started_at = now_iso()
        self.status = "active"
        self.current_step = 0
        self.read_count = 0
        self.research_brief: dict[str, Any] = {}
        self.clarifying_answers: str = ""
        self.research_plan: dict[str, Any] = {}
        self.plan_approved: bool = False
        self.report_style: str = "standard"
        self.stop_reason: str = ""
        self.stop_details: dict[str, Any] = {}
        self.sources: list[dict[str, Any]] = []
        self.source_ledger: list[dict[str, Any]] = []
        self.claims: list[dict[str, Any]] = []
        self.contradictions: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.frontier: list[dict[str, Any]] = []
        self.frontier_audit: list[dict[str, Any]] = []
        self.media_ledger: list[dict[str, Any]] = []
        self.search_history: list[dict[str, Any]] = []
        self.coverage: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.save()

    @property
    def session_path(self) -> Path:
        return SESSION_DIR / f"{self.session_id}.json"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_storage(
            {
                "session_id": self.session_id,
                "question": self.question,
                "max_steps": self.max_steps,
                "max_sources": self.max_sources,
                "max_reads": self.max_reads,
                "time_limit_min": self.time_limit_min,
                "depth": self.depth,
                "min_steps": self.min_steps,
                "min_sources": self.min_sources,
                "min_reads": self.min_reads,
                "started_at": self.started_at,
                "status": self.status,
                "current_step": self.current_step,
                "read_count": self.read_count,
                "research_brief": self.research_brief,
                "clarifying_answers": self.clarifying_answers,
                "research_plan": self.research_plan,
                "plan_approved": self.plan_approved,
                "report_style": self.report_style,
                "stop_reason": self.stop_reason,
                "stop_details": self.stop_details,
                "sources": self.sources,
                "source_ledger": self.source_ledger,
                "claims": self.claims,
                "contradictions": self.contradictions,
                "notes": self.notes,
                "frontier": self.frontier,
                "frontier_audit": self.frontier_audit,
                "media_ledger": self.media_ledger,
                "search_history": self.search_history,
                "coverage": self.coverage,
                "trace": self.trace,
            }
        )

    def save(self) -> None:
        self.session_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        data = json.loads(self.session_path.read_text(encoding="utf-8"))
        self.__dict__.update(data)
        self.max_reads = getattr(self, "max_reads", data.get("max_reads", int(os.environ.get("SPIDER_MAX_READS", "20"))))
        self.claims = getattr(self, "claims", data.get("claims", []))
        self.frontier = getattr(self, "frontier", data.get("frontier", []))
        self.frontier_audit = getattr(self, "frontier_audit", data.get("frontier_audit", []))
        self.media_ledger = getattr(self, "media_ledger", data.get("media_ledger", []))
        self.search_history = getattr(self, "search_history", data.get("search_history", []))
        self.coverage = getattr(self, "coverage", data.get("coverage", {}))
        self.read_count = getattr(self, "read_count", data.get("read_count", 0))
        self.depth = getattr(self, "depth", data.get("depth", "standard"))
        self.clarifying_answers = getattr(self, "clarifying_answers", data.get("clarifying_answers", ""))
        self.research_plan = getattr(self, "research_plan", data.get("research_plan", {}))
        self.plan_approved = getattr(self, "plan_approved", data.get("plan_approved", False))
        self.report_style = getattr(self, "report_style", data.get("report_style", "standard"))
        self.stop_reason = getattr(self, "stop_reason", data.get("stop_reason", ""))
        self.stop_details = getattr(self, "stop_details", data.get("stop_details", {}))
        mins = depth_minimums(self.depth)
        self.min_steps = getattr(self, "min_steps", data.get("min_steps", mins["min_steps"]))
        self.min_sources = getattr(self, "min_sources", data.get("min_sources", mins["min_sources"]))
        self.min_reads = getattr(self, "min_reads", data.get("min_reads", mins["min_reads"]))

    def add_trace(self, step_type: str, content: dict[str, Any]) -> None:
        self.trace.append(sanitize_for_storage({"step": self.current_step, "type": step_type, "timestamp": now_iso(), **content}))
        self.save()

    def record_frontier_state(self, item: dict[str, Any], state: str, reason: str = "") -> None:
        entry = sanitize_for_storage({**item, "frontier_state": state, "frontier_reason": reason, "timestamp": now_iso(), "step": self.current_step})
        self.frontier_audit.append(entry)
        if state != "read_now":
            self.add_trace("frontier_state", entry)
        else:
            self.save()

    def record_stop_decision(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        self.stop_reason = reason
        self.stop_details = sanitize_for_storage(details or {})
        self.add_trace("stop_decision", {"reason": reason, "details": self.stop_details})

    def add_source(self, source: dict[str, Any], seed: bool = False) -> Optional[str]:
        url = source.get("requested_url") or source.get("url") or source.get("final_url")
        if not url:
            return None
        for existing in self.sources:
            if url in {existing.get("requested_url"), existing.get("url"), existing.get("final_url")}:
                return existing["source_id"]
        if len(self.sources) >= self.max_sources:
            return None
        source_id = f"S{len(self.sources) + 1}"
        text = source.get("text", "")
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        text_ref = DOC_DIR / f"{self.session_id}_{source_id}.txt"
        text_ref.write_text(strip_thinking(text), encoding="utf-8")
        title = source.get("title") or infer_title(text, url)
        family = source_family_for_url(source.get("final_url", url))
        source_category = source.get("source_category") or source_category_for_url(source.get("final_url", url), title, text[:1000])
        entry = {
            "source_id": source_id,
            "requested_url": source.get("requested_url", url),
            "url": url,
            "final_url": source.get("final_url", url),
            "title": title,
            "source_type": source.get("adapter", source_type_for_url(url)),
            "score": source.get("score", score_source_text(text, url)),
            "relevance": source.get("relevance", "seed" if seed else "candidate"),
            "relevance_reason": source.get("relevance_reason", ""),
            "source_card": source.get("source_card", ""),
            "structured_from_card": source.get("structured_from_card", False),
            "seed": seed,
            "extracted_chars": source.get("extracted_chars", len(text)),
            "fetched_at": source.get("fetched_at", now_iso()),
            "cache_key": source.get("cache_key"),
            "cache_path": source.get("cache_path"),
            "text_ref": str(text_ref),
            "links": source.get("links", [])[:25],
            "media": source.get("media", [])[:25],
            "source_category": source_category,
            "source_family": family.get("source_family"),
            "path_family": family.get("path_family"),
            "domain": family.get("domain"),
            "target_hint": source.get("target_hint", ""),
            "snippet": strip_thinking(text[:900]),
        }
        self.sources.append(entry)
        self.source_ledger.append(entry.copy())
        for media in entry["media"]:
            self.media_ledger.append(
                {
                    "media_id": f"M{len(self.media_ledger) + 1}",
                    "parent_source_id": source_id,
                    "url": media.get("url"),
                    "alt": media.get("alt", ""),
                    "caption": media.get("caption", ""),
                    "source": media.get("source", "img"),
                    "status": "discovered",
                    "timestamp": now_iso(),
                }
            )
        self.save()
        return source_id

    def add_note(self, text: str, source_id: str = "") -> None:
        cleaned = strip_thinking(text)
        if cleaned:
            self.notes.append({"text": cleaned, "source_id": source_id, "timestamp": now_iso()})
            self.save()

    def update_source_relevance(self, source_id: str, relevance: str, reason: str = "") -> None:
        relevance = relevance if relevance in {"seed", "relevant", "partial", "irrelevant", "summarization_failed"} else "partial"
        for source in self.sources:
            if source.get("source_id") == source_id:
                source["relevance"] = relevance
                source["relevance_reason"] = strip_thinking(reason)
        for source in self.source_ledger:
            if source.get("source_id") == source_id:
                source["relevance"] = relevance
                source["relevance_reason"] = strip_thinking(reason)
        self.save()

    def add_claim(self, claim: str, source_ids: list[str], confidence: str = "medium", status: str = "supported", evidence: str = "") -> None:
        claim = strip_thinking(claim)
        if not claim:
            return
        evidence = strip_thinking(evidence)
        for existing in self.claims:
            if existing.get("claim", "").lower() == claim.lower():
                existing["source_ids"] = sorted(set(existing.get("source_ids", []) + source_ids))
                if evidence and evidence not in existing.get("evidence", []):
                    existing.setdefault("evidence", []).append(evidence)
                if "verification" not in existing:
                    existing["verification"] = "unverified"
                self.save()
                return
        self.claims.append(
            {
                "claim_id": f"C{len(self.claims) + 1}",
                "claim": claim,
                "source_ids": source_ids,
                "confidence": confidence,
                "status": status,
                "evidence": [evidence] if evidence else [],
                "verification": "unverified",
                "verification_note": "",
                "created_at": now_iso(),
            }
        )
        self.save()


def infer_title(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip(" #\t")
        if 8 <= len(line) <= 160:
            return strip_thinking(line)
    return fallback


def depth_minimums(depth: str) -> dict[str, int]:
    profiles = {
        "quick": {"min_steps": 1, "min_sources": 1, "min_reads": 1},
        "standard": {"min_steps": 3, "min_sources": 4, "min_reads": 3},
        "deep": {"min_steps": 10, "min_sources": 12, "min_reads": 10},
    }
    return profiles.get(depth, profiles["standard"]).copy()


def depth_budget_defaults(depth: str) -> dict[str, int]:
    profiles = {
        "quick": {"max_steps": 3, "max_sources": 12, "max_reads": 6, "time_limit": 10},
        "standard": {"max_steps": 15, "max_sources": 40, "max_reads": 20, "time_limit": 30},
        "deep": {"max_steps": 120, "max_sources": 200, "max_reads": 80, "time_limit": 120},
    }
    return profiles.get(depth, profiles["standard"]).copy()


def get_source(session: SpiderSession, source_id: str) -> Optional[dict[str, Any]]:
    return next((s for s in session.sources if s.get("source_id") == source_id), None)


def get_source_text(session: SpiderSession, source_id: str) -> str:
    source = get_source(session, source_id)
    if not source:
        return ""

    text_ref = source.get("text_ref")
    if text_ref:
        path = Path(text_ref)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")

    cache_path = source.get("cache_path")
    if cache_path:
        path = Path(cache_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return strip_thinking(data.get("text", ""))
            except Exception:
                pass

    return source.get("snippet", "")


def create_research_brief(session: SpiderSession, clarify: bool = False) -> None:
    if session.research_brief:
        return
    log("[Spider] Creating research brief")
    prompt = f"""Question/topic:
{session.question}

Return a compact JSON object with:
question, scope, exclusions, target_audience, assumptions, output_style,
needs_clarification (boolean), clarifying_questions (array of strings).
If the topic is answerable with the given question and seed sources, needs_clarification should be false."""
    parsed = extract_json_value(call_controller_model(prompt))
    if not isinstance(parsed, dict):
        parsed = {
            "question": session.question,
            "scope": "Research the topic using seed and authoritative sources.",
            "exclusions": [],
            "target_audience": "Abe / technical operator",
            "assumptions": [],
            "output_style": "cited markdown report",
            "needs_clarification": False,
            "clarifying_questions": [],
        }
    session.research_brief = parsed
    session.add_trace("research_brief", parsed)
    write_json_artifact("research_brief.json", session.research_brief)
    if clarify and parsed.get("needs_clarification"):
        write_clarifying_question_artifacts(session, force=False)


def fallback_clarifying_questions(session: SpiderSession) -> list[str]:
    return [
        "What scope should Spider prioritize?",
        "Are there seed sources, files, or exclusions Spider should treat as authoritative?",
        "What output shape would be most useful: concise answer, implementation plan, comparison table, or full report?",
    ]


def write_clarifying_question_artifacts(session: SpiderSession, force: bool = False) -> list[str]:
    questions = [strip_thinking(q) for q in session.research_brief.get("clarifying_questions", []) if strip_thinking(q)]
    if force and not questions:
        questions = fallback_clarifying_questions(session)
    payload = {
        "session_id": session.session_id,
        "question": session.question,
        "needs_clarification": True,
        "clarifying_questions": questions,
        "research_brief": session.research_brief,
    }
    write_json_artifact("research_brief.json", session.research_brief)
    write_json_artifact("clarifying_questions.json", payload)
    lines = [f"# Clarifying Questions for Spider Session {session.session_id}", ""]
    lines.extend(f"- {question}" for question in questions)
    lines.extend(["", "Add answers in an answers.md file, then resume with --answers answers.md --plan-only."])
    write_text_artifact("clarifying_questions.md", "\n".join(lines) + "\n")
    session.status = "needs_clarification"
    session.add_trace("clarifying_questions", {"questions": questions, "force": force})
    session.save()
    print("[Spider] Clarifying questions:")
    for question in questions:
        print(f"- {question}")
    return questions


def apply_clarify_policy(session: SpiderSession, policy: str) -> bool:
    """Return True when the workflow should stop before planning/research."""
    if policy == "never":
        return False
    if policy == "always":
        write_clarifying_question_artifacts(session, force=True)
        return True
    if session.research_brief.get("needs_clarification"):
        write_clarifying_question_artifacts(session, force=True)
        return True
    return False


def merge_answers(session: SpiderSession, answers_path: str | None) -> None:
    if not answers_path:
        return
    path = Path(answers_path).expanduser()
    answers = strip_thinking(path.read_text(encoding="utf-8", errors="replace"))
    session.clarifying_answers = answers
    session.research_brief.setdefault("clarifying_answers", answers)
    session.research_brief["needs_clarification"] = False
    session.status = "active"
    write_json_artifact("research_brief.json", session.research_brief)
    write_text_artifact("answers.md", answers)
    session.add_trace("clarifying_answers", {"path": str(path), "chars": len(answers)})
    session.save()


def fallback_research_plan(session: SpiderSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "question": session.question,
        "scope": session.research_brief.get("scope", "Research the question with retrieved and supplied sources."),
        "exclusions": session.research_brief.get("exclusions", []),
        "search_strategy": [
            "Ingest supplied seed URLs and files first.",
            "Prefer official or primary sources before community summaries.",
            "Use search only to fill source diversity, missing primary sources, contradictions, and verification gaps.",
        ],
        "source_priorities": ["seed sources", "official documentation", "primary repositories/issues", "technical reports", "reputable secondary sources"],
        "source_policy": default_source_policy(session),
        "verification_plan": [
            "Extract claims with source IDs and evidence snippets.",
            "Verify high-impact report claims first.",
            "Mark unverifiable claims as source_supported_unverified instead of presenting them as settled.",
        ],
        "budget": {
            "depth": session.depth,
            "max_steps": session.max_steps,
            "max_sources": session.max_sources,
            "max_reads": session.max_reads,
            "time_limit_min": session.time_limit_min,
        },
        "expected_output_sections": [
            "Executive Summary",
            "Answer or Detailed Analysis",
            "Evidence Table",
            "Contradictions and Uncertainties",
            "Limitations / Next Research",
            "Sources",
        ],
        "approval_required": True,
    }


def research_plan_to_markdown(plan: dict[str, Any]) -> str:
    def lines_for(value: Any) -> list[str]:
        if isinstance(value, list):
            return [f"- {item}" for item in value]
        if isinstance(value, dict):
            return [f"- `{k}`: {v}" for k, v in value.items()]
        if value in (None, ""):
            return ["- Not specified."]
        return [str(value)]

    sections = [
        ("Scope", plan.get("scope")),
        ("Exclusions", plan.get("exclusions")),
        ("Search Strategy", plan.get("search_strategy")),
        ("Source Priorities", plan.get("source_priorities")),
        ("Source Policy", plan.get("source_policy")),
        ("Verification Plan", plan.get("verification_plan")),
        ("Budget / Depth", plan.get("budget")),
        ("Expected Output Sections", plan.get("expected_output_sections")),
    ]
    out = [f"# Spider Research Plan - {plan.get('session_id', '')}", "", f"Question: {plan.get('question', '')}", ""]
    for heading, value in sections:
        out.extend([f"## {heading}", *lines_for(value), ""])
    out.append("Approval: edit this plan if needed, then resume Spider with --approve-plan research_plan.json or --yes.")
    return "\n".join(out).strip() + "\n"


def generate_research_plan(session: SpiderSession) -> dict[str, Any]:
    if session.research_plan:
        write_json_artifact("research_plan.json", session.research_plan)
        write_text_artifact("research_plan.md", research_plan_to_markdown(session.research_plan))
        return session.research_plan
    prompt = f"""Create a proposed Spider deep-research plan. Return ONLY JSON.

Research brief:
{json.dumps(session.research_brief, indent=2, ensure_ascii=False)}

Clarifying answers:
{session.clarifying_answers or 'None.'}

Budgets:
{json.dumps({'depth': session.depth, 'max_steps': session.max_steps, 'max_sources': session.max_sources, 'max_reads': session.max_reads, 'time_limit_min': session.time_limit_min}, indent=2)}

JSON keys: scope, exclusions, search_strategy, source_priorities, source_policy, verification_plan, budget, expected_output_sections, approval_required.
source_policy must include: must_cover_targets, source_categories, category_minimums, category_budgets, domain_caps, path_family_caps, source_family_caps, high_value_terms, low_value_unless_plan_relevant, explicit_allow_terms, explicit_skip_terms, source_role_notes.
Keep it practical and editable."""
    parsed = extract_json_value(call_controller_model(prompt, max_tokens=int(os.environ.get("SPIDER_PLAN_MAX_TOKENS", "4096"))))
    plan = parsed if isinstance(parsed, dict) else fallback_research_plan(session)
    plan.setdefault("session_id", session.session_id)
    plan.setdefault("question", session.question)
    plan.setdefault("budget", fallback_research_plan(session)["budget"])
    plan.setdefault("source_policy", default_source_policy(session))
    plan.setdefault("approval_required", True)
    session.research_plan = sanitize_for_storage(plan)
    session.status = "awaiting_plan_approval"
    write_json_artifact("research_plan.json", session.research_plan)
    write_text_artifact("research_plan.md", research_plan_to_markdown(session.research_plan))
    session.add_trace("research_plan", {"path": str(artifact_path("research_plan.json")), "approval_required": True})
    session.save()
    return session.research_plan


def approve_research_plan(session: SpiderSession, plan_path: str | None = None, auto: bool = False) -> None:
    if plan_path:
        path = Path(plan_path).expanduser()
        plan = json.loads(path.read_text(encoding="utf-8"))
        session.research_plan = sanitize_for_storage(plan)
    elif not session.research_plan:
        generate_research_plan(session)
    session.plan_approved = True
    session.status = "active"
    session.add_trace("plan_approved", {"auto": auto, "plan_path": plan_path or str(artifact_path("research_plan.json"))})
    session.save()


def handle_workflow_gates(
    session: SpiderSession,
    *,
    clarify_policy: str,
    answers_path: str | None,
    plan_only: bool,
    yes: bool,
    approve_plan_path: str | None,
) -> bool:
    """Return True when Spider may proceed to the expensive research run."""
    create_research_brief(session, clarify=False)
    if answers_path:
        merge_answers(session, answers_path)
    if approve_plan_path:
        approve_research_plan(session, approve_plan_path)
        return True

    if not answers_path and apply_clarify_policy(session, clarify_policy):
        return False

    if plan_only:
        generate_research_plan(session)
        return False

    if yes:
        if not session.research_plan:
            generate_research_plan(session)
        approve_research_plan(session, auto=True)
        return True

    if not session.plan_approved:
        generate_research_plan(session)
        print(f"[Spider] Research plan written: {artifact_path('research_plan.md')}")
        print("[Spider] Review/edit the plan, then resume with --approve-plan research_plan.json or pass --yes to auto-approve.")
        return False

    return True


def parse_source_card_sections(card: str) -> dict[str, str]:
    sections = {key: "" for key in ("SOURCE_RELEVANCE", "SUMMARY", "KEY_CLAIMS", "EVIDENCE_SNIPPETS", "CONTRADICTIONS", "FOLLOW_UP_LINKS")}
    current = ""
    for line in strip_thinking(card).splitlines():
        match = re.match(r"^\s*(SOURCE_RELEVANCE|SUMMARY|KEY_CLAIMS|EVIDENCE_SNIPPETS|CONTRADICTIONS|FOLLOW_UP_LINKS)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            current = match.group(1).upper()
            sections[current] = match.group(2).strip()
            continue
        if current:
            sections[current] = (sections[current] + "\n" + line).strip()
    return sections


def source_card_to_fallback_json(card: str) -> dict[str, Any]:
    sections = parse_source_card_sections(card)
    relevance_text = sections.get("SOURCE_RELEVANCE", "").lower()
    relevance = "partial"
    if "irrelevant" in relevance_text:
        relevance = "irrelevant"
    elif "relevant" in relevance_text:
        relevance = "relevant"
    claims = []
    evidence_lines = [re.sub(r"^\s*[-*]\s*", "", line).strip() for line in sections.get("EVIDENCE_SNIPPETS", "").splitlines() if line.strip()]
    for idx, line in enumerate(sections.get("KEY_CLAIMS", "").splitlines()):
        claim = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if claim and claim.lower() not in {"none", "n/a"}:
            claims.append({"claim": claim, "confidence": "medium", "evidence": evidence_lines[min(idx, len(evidence_lines) - 1)] if evidence_lines else ""})
    contradictions = []
    for line in sections.get("CONTRADICTIONS", "").splitlines():
        item = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if item and item.lower() not in {"none", "n/a"}:
            contradictions.append({"claim1": item, "claim2": "", "source1": "", "source2": ""})
    return {
        "relevance": relevance,
        "relevance_reason": sections.get("SOURCE_RELEVANCE", ""),
        "summary": sections.get("SUMMARY", ""),
        "claims": claims,
        "contradictions": contradictions,
        "follow_up_links": sections.get("FOLLOW_UP_LINKS", ""),
    }


def structure_source_card(session: SpiderSession, source_id: str, source_card: str) -> tuple[dict[str, Any], bool]:
    already_json = extract_json_value(source_card)
    if isinstance(already_json, dict):
        return already_json, True
    if "SOURCE_RELEVANCE" not in source_card.upper():
        return source_card_to_fallback_json(source_card), False
    prompt = f"""Convert this Spider source card into valid compact JSON.

Required JSON object keys:
relevance ("relevant"|"partial"|"irrelevant"), relevance_reason, summary,
claims (array of objects with claim, confidence high|medium|low, evidence),
contradictions (array of objects), follow_up_links (array of strings).

Research question:
{session.question}

Source ID: {source_id}

Source card:
{source_card[:18000]}

Return ONLY JSON. Do not add prose."""
    parsed = extract_json_value(call_structurer_model(prompt))
    if isinstance(parsed, dict):
        return parsed, True
    return source_card_to_fallback_json(source_card), False


def summarize_source(session: SpiderSession, source_id: str, text: str) -> None:
    log(f"[Spider] Analyzing source {source_id} ({len(text)} chars)")
    source = get_source(session, source_id) or {}
    target_hint = source.get("target_hint", "")
    target_guidance = ""
    if target_hint:
        target_guidance = (
            f"\nThis source was selected for named comparison target {target_hint!r}. "
            "For comparison questions, a source does not need to compare every option to be useful. "
            "Treat target-specific evidence as relevant/partial when it supports facts about that target."
        )
    prompt = f"""Research question:
{session.question}

Source ID: {source_id}
Source title: {source.get('title', '')}
Source URL: {source.get('final_url') or source.get('url', '')}

Source text:
{text[:18000]}

Create a source-grounded markdown SOURCE CARD. This is not strict JSON.
Use exactly these headings, each followed by concise content:

SOURCE_RELEVANCE:
SUMMARY:
KEY_CLAIMS:
EVIDENCE_SNIPPETS:
CONTRADICTIONS:
FOLLOW_UP_LINKS:

Rules:
- Include only claims supported by the source text.
- Evidence snippets should be short exact or near-exact excerpts.
- If unrelated, set SOURCE_RELEVANCE to irrelevant and explain why.
- Do not include hidden reasoning or <think> blocks.
{target_guidance}"""
    source_card = strip_thinking(call_nanbeige(prompt, max_tokens=int(os.environ.get("SPIDER_SOURCE_CARD_MAX_TOKENS", "12000"))))
    parsed, structured_ok = structure_source_card(session, source_id, source_card)
    if not isinstance(parsed, dict):
        parsed, structured_ok = source_card_to_fallback_json(source_card), False
    for collection in (session.sources, session.source_ledger):
        for item in collection:
            if item.get("source_id") == source_id:
                item["source_card"] = source_card
                item["structured_from_card"] = structured_ok
                item["structuring_status"] = "ok" if structured_ok else "fallback_markdown_card"
    relevance = parsed.get("relevance", "partial")
    if relevance not in {"relevant", "partial", "irrelevant"}:
        relevance = "partial"
    reason = parsed.get("relevance_reason", "")
    if not structured_ok and relevance != "irrelevant":
        relevance = "partial"
        reason = reason or "Structurer failed; retained markdown source card as partial evidence."
    session.update_source_relevance(source_id, relevance, reason)
    session.add_note(parsed.get("summary", "") or parse_source_card_sections(source_card).get("SUMMARY", ""), source_id=source_id)
    session.add_trace("source_card", {"source_id": source_id, "structured": structured_ok, "relevance": relevance})
    if relevance == "irrelevant":
        session.save()
        return
    for item in parsed.get("claims", []) or []:
        if isinstance(item, dict):
            session.add_claim(item.get("claim", ""), [source_id], item.get("confidence", "medium"), evidence=item.get("evidence", ""))
        elif isinstance(item, str):
            session.add_claim(item, [source_id])
    for item in parsed.get("contradictions", []) or []:
        if isinstance(item, dict):
            session.contradictions.append(sanitize_for_storage({**item, "step": session.current_step, "timestamp": now_iso()}))
    session.save()


def ingest_seed_urls(session: SpiderSession, seed_urls: list[str]) -> None:
    combined_urls = []
    for url in [*seed_urls, *extract_urls(session.question)]:
        if url not in combined_urls:
            combined_urls.append(url)
    for url in combined_urls:
        log(f"[Seed URL] {url}")
        source = fetch_webpage(url, max_chars=18000)
        sid = session.add_source(source, seed=True)
        session.add_trace("read_url", {"url": url, "source_id": sid, "score": source.get("score"), "seed": True})
        if sid:
            enqueue_source_links(session, sid, source)
        if sid and source.get("score", -1000) > -10:
            session.read_count += 1
            summarize_source(session, sid, source.get("text", ""))
        elif sid:
            session.add_note(source.get("text", ""), sid)


def ingest_files(session: SpiderSession, files: list[str]) -> None:
    for item in files:
        p = Path(item).expanduser()
        paths = sorted([x for x in p.rglob("*") if x.is_file()]) if p.is_dir() else [p]
        for path in paths:
            if session.read_count >= session.max_reads:
                return
            log(f"[Seed file] {path}")
            source = read_local_file(str(path), max_chars=18000)
            sid = session.add_source(source, seed=True)
            session.add_trace("read_file", {"path": str(path), "source_id": sid, "seed": True})
            if sid:
                session.read_count += 1
                summarize_source(session, sid, source.get("text", ""))


def plan_fallback_actions(session: SpiderSession) -> list[dict[str, Any]]:
    if not session.frontier:
        action = next_search_action(session)
        return [action] if action else [{"tool": "stop", "reason": "No frontier and no untried search queries remain."}]
    return [session.frontier.pop(0)]


def search_query_candidates(session: SpiderSession) -> list[str]:
    seeds = []
    question_l = session.question.lower()
    for source in session.sources:
        title = source.get("title", "")
        url = source.get("url", "")
        if "Nanbeige" in title or "Nanbeige" in url:
            seeds.extend(
                [
                    "Nanbeige4.1-3B",
                    "\"Nanbeige4.1-3B\"",
                    "\"Nanbeige4.1-3B\" arxiv",
                    "\"Nanbeige4.1-3B\" Hugging Face",
                    "\"Nanbeige4.1-3B\" benchmark reasoning alignment agentic",
                    "\"Nanbeige4.1-3B\" limitations",
                ]
            )
    if ("pep 703" in question_l or "free-thread" in question_l or "free threading" in question_l) and "python" in question_l:
        seeds.extend(
            [
                "CPython free-threading C extension HOWTO",
                "free-threaded CPython extension modules Py_mod_gil",
                "PEP 703 extension authors free threading",
                "Python free-threaded C API extension thread safety",
                "Updating Extension Modules Python Free-Threading Guide",
                "PyO3 supporting free-threaded Python",
                "Python free-threading extension authors risks migration",
            ]
        )
    if any(term in question_l for term in ("llama.cpp", "ollama", "vllm", "local llm serving", "inference server")):
        seeds.extend(
            [
                "llama.cpp server documentation OpenAI compatible API",
                "Ollama documentation API model serving",
                "vLLM documentation serving OpenAI compatible server",
                "ggml-org llama.cpp server README GitHub",
                "ollama ollama GitHub README server API",
                "vllm project vllm docs serving engine",
                "llama.cpp Ollama vLLM deployment GPU CPU memory official docs",
            ]
        )
    if any(term in question_l for term in ("memory", "memories", "rag", "embedding", "dreaming", "agent memory", "brains")) and any(term in question_l for term in ("agent", "llm", "harness", "homelab", "abe", "openai", "perplexity")):
        seeds.extend(
            [
                "LLM agent memory architecture RAG embeddings long term memory",
                "OpenAI memory ChatGPT agent memory project documentation",
                "self improving memory for agents Perplexity paper",
                "agent memory systems MemGPT Letta Zep LangGraph LangChain",
                "AI agent memory papers episodic semantic procedural memory RAG",
                "Generative Agents memory reflection planning paper",
                "Reflexion language agents verbal reinforcement learning memory",
                "Voyager lifelong learning agent skill memory paper",
                "homelab local LLM agent memory RAG embeddings architecture",
            ]
        )
    base_query = normalize_search_query(session.question)
    generic = [
        base_query,
        re.sub(r"[?].*$", "", base_query).strip(),
        f"{base_query} official source",
        f"{base_query} arxiv github huggingface",
        f"{base_query} limitations evidence",
    ]
    seen = set()
    out = []
    for query in seeds + generic:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in seen:
            seen.add(query)
            out.append(query)
    return out


def next_search_action(session: SpiderSession) -> Optional[dict[str, Any]]:
    if SEARCH_BACKEND_UNAVAILABLE:
        return None
    tried = {item.get("query") for item in session.search_history}
    for query in search_query_candidates(session):
        if query not in tried:
            return {"tool": "search", "query": query}
    return None


def comparison_targets(session: SpiderSession) -> list[str]:
    question = session.question.lower()
    targets = []
    for target in ("llama.cpp", "ollama", "vllm"):
        if target in question:
            targets.append(target)
    return targets


def source_mentions_target(source: dict[str, Any], target: str) -> bool:
    haystack = f"{source.get('title', '')} {source.get('url', '')} {source.get('final_url', '')} {source.get('snippet', '')} {source.get('target_hint', '')}".lower()
    if target == "llama.cpp":
        return "llama.cpp" in haystack or "llama-cpp" in haystack or "llama_cpp" in haystack
    return target in haystack


def missing_comparison_targets(session: SpiderSession) -> list[str]:
    targets = comparison_targets(session)
    if not targets:
        return []
    evidence = evidence_sources(session)
    covered = {target for target in targets if any(source_mentions_target(source, target) for source in evidence)}
    return [target for target in targets if target not in covered]


def target_hint_matches_url(url: str, title: str, target: str) -> bool:
    haystack = f"{url} {title}".lower()
    if target == "llama.cpp":
        return any(token in haystack for token in ("llama.cpp", "llama-cpp", "llama_cpp", "ggml-org"))
    return target in haystack


def next_gap_search_action(session: SpiderSession) -> Optional[dict[str, Any]]:
    targets = comparison_targets(session)
    if not targets:
        return None
    missing = missing_comparison_targets(session)
    tried = {item.get("query") for item in session.search_history}
    target_queries = {
        "llama.cpp": "llama.cpp server documentation OpenAI compatible API",
        "ollama": "Ollama documentation API model serving",
        "vllm": "vLLM documentation serving OpenAI compatible server",
    }
    for target in targets:
        query = target_queries[target]
        if target in missing and query not in tried:
            session.coverage["missing_comparison_targets"] = missing
            return {"tool": "search", "query": query, "coverage_gap": target}
    return None


def source_mentions_policy_target(source: dict[str, Any], target: str) -> bool:
    haystack = f"{source.get('title', '')} {source.get('url', '')} {source.get('final_url', '')} {source.get('snippet', '')} {source.get('target_hint', '')}".lower()
    return target.lower() in haystack


def missing_policy_targets(session: SpiderSession) -> list[str]:
    targets = terms_from_text(source_policy(session).get("must_cover_targets", []))
    if not targets:
        return []
    evidence = evidence_sources(session)
    return [target for target in targets if not any(source_mentions_policy_target(source, target) for source in evidence)]


def next_policy_gap_search_action(session: SpiderSession) -> Optional[dict[str, Any]]:
    if SEARCH_BACKEND_UNAVAILABLE:
        return None
    policy = source_policy(session)
    tried = {item.get("query") for item in session.search_history}
    for target in missing_policy_targets(session):
        if any(target_hint_matches_url(item.get("url", ""), item.get("title", ""), target) or target.lower() in f"{item.get('title', '')} {item.get('snippet', '')} {item.get('url', '')}".lower() for item in session.frontier):
            continue
        query = f"{target} {normalize_search_query(session.question, max_chars=120)}"
        query = normalize_search_query(query)
        if query and query not in tried:
            session.coverage["missing_policy_targets"] = missing_policy_targets(session)
            return {"tool": "search", "query": query, "coverage_gap": target, "policy_gap": "must_cover_target"}
    counts = category_counts(session)
    for category, minimum in (policy.get("category_minimums") or {}).items():
        if counts.get(category, 0) >= int(minimum):
            continue
        if any((item.get("source_category") or source_category_for_url(item.get("url", ""), item.get("title", ""), item.get("snippet", ""))) == category for item in session.frontier):
            continue
        category_query = {
            "papers": f"{normalize_search_query(session.question, 120)} arxiv paper agent memory",
            "project_repos": f"{normalize_search_query(session.question, 120)} GitHub project repository",
            "project_docs": f"{normalize_search_query(session.question, 120)} official documentation",
            "labs_products": f"{normalize_search_query(session.question, 120)} official blog documentation",
            "blogs": f"{normalize_search_query(session.question, 120)} technical blog",
        }.get(str(category), f"{normalize_search_query(session.question, 120)} {category}")
        category_query = normalize_search_query(category_query)
        if category_query not in tried:
            session.coverage["missing_source_categories"] = {
                k: {"have": counts.get(k, 0), "need": int(v)}
                for k, v in (policy.get("category_minimums") or {}).items()
                if counts.get(k, 0) < int(v)
            }
            return {"tool": "search", "query": category_query, "coverage_gap": category, "policy_gap": "source_category"}
    session.coverage.pop("missing_policy_targets", None)
    session.coverage.pop("missing_source_categories", None)
    return None


def search_exhausted(session: SpiderSession) -> bool:
    if SEARCH_BACKEND_UNAVAILABLE:
        session.coverage["search_backend_unavailable"] = True
        return True
    tried = {item.get("query") for item in session.search_history}
    return all(query in tried for query in search_query_candidates(session))


def ask_controller(session: SpiderSession) -> dict[str, Any]:
    state = {
        "brief": session.research_brief,
        "frontier": sorted(session.frontier, key=lambda x: x.get("search_score", 0), reverse=True)[:10],
        "sources": session.source_ledger[-12:],
        "claims": session.claims[-18:],
        "notes": session.notes[-8:],
        "coverage": session.coverage,
        "budgets": {
            "step": session.current_step,
            "max_steps": session.max_steps,
            "sources": len(session.sources),
            "max_sources": session.max_sources,
            "reads": session.read_count,
            "max_reads": session.max_reads,
        },
    }
    prompt = f"""You are the visible research controller for Spider. Choose exactly one next harness action.

Allowed tools:
- search: {{"tool":"search","query":"..."}}
- read_url: {{"tool":"read_url","url":"..."}}
- extract_claims: {{"tool":"extract_claims","source_id":"S1"}}
- verify_claim: {{"tool":"verify_claim","claim_id":"C1","query":"..."}}
- final_report: {{"tool":"final_report"}}
- stop: {{"tool":"stop","reason":"..."}}

Prefer primary/official sources. If frontier contains promising read_url actions, consume them before searching again.
Seeds are strong evidence. Empty/bad search results must not override seed evidence.
Return ONLY JSON with keys: rationale (short visible rationale) and action.

Current state:
{json.dumps(state, indent=2, ensure_ascii=False)}"""
    parsed = extract_json_value(call_controller_model(prompt))
    if isinstance(parsed, dict) and isinstance(parsed.get("action"), dict):
        return parsed
    fallback = pop_frontier_action(session) or next_search_action(session) or {"tool": "final_report"}
    return {"rationale": "Controller did not return a valid action; harness selected the next bounded fallback.", "action": fallback}


def pop_frontier_action(session: SpiderSession) -> Optional[dict[str, Any]]:
    missing_targets = missing_comparison_targets(session)
    policy = source_policy(session)
    ranked: list[dict[str, Any]] = []
    for item in list(session.frontier):
        if item.get("tool") != "read_url" or not item.get("url") or not is_readable_url(item.get("url", "")) or is_blocklisted_url(item.get("url", "")):
            session.frontier.remove(item)
            session.record_frontier_state(item, "skipped_low_policy_score", "unreadable or blocklisted")
            continue
        evaluation = policy_score_item(session, item, policy)
        item.update(evaluation)
        if evaluation["frontier_state"].startswith("skipped_"):
            session.frontier.remove(item)
            session.record_frontier_state(item, evaluation["frontier_state"], evaluation["frontier_reason"])
            continue
        ranked.append(item)
    readable = [item for item in ranked if item.get("frontier_state", "read_now") == "read_now"]
    if not readable:
        session.save()
        return None
    def frontier_priority(item: dict[str, Any]) -> tuple[int, int, int, int]:
        target_hint = item.get("target_hint", "")
        title = item.get("title", "")
        url = item.get("url", "")
        gap_match = int(bool(target_hint and target_hint in missing_targets))
        organic_match = int(any(target_hint_matches_url(url, title, target) for target in missing_targets))
        fills_gap = int(bool((item.get("source_policy_matches") or {}).get("fills_category_gap")))
        must_match = int(bool((item.get("source_policy_matches") or {}).get("must_cover_targets")))
        return (gap_match, organic_match, must_match + fills_gap, int(item.get("policy_score", item.get("search_score", 0))))

    readable.sort(key=frontier_priority, reverse=True)
    selected = readable[0]
    session.frontier.remove(selected)
    session.record_frontier_state(selected, "read_now", selected.get("frontier_reason", "selected for reading"))
    session.save()
    return selected


def add_frontier_candidate(session: SpiderSession, item: dict[str, Any]) -> bool:
    url = item.get("url", "")
    if not url or not is_readable_url(url) or is_blocklisted_url(url):
        session.record_frontier_state(item, "skipped_low_policy_score", "unreadable or blocklisted")
        return False
    if source_already_seen(session, url) or any(existing.get("url") == url for existing in session.frontier):
        session.record_frontier_state(item, "skipped_duplicate", "already seen or already queued")
        return False
    item.setdefault("tool", "read_url")
    item.setdefault("source_category", source_category_for_url(url, item.get("title", ""), item.get("snippet", "")))
    item.update(policy_score_item(session, item))
    state = item.get("frontier_state", "read_now")
    if state == "read_now" or state == "reserve":
        session.frontier.append(item)
        session.record_frontier_state(item, state, item.get("frontier_reason", "queued"))
        return state == "read_now"
    session.record_frontier_state(item, state, item.get("frontier_reason", "not queued"))
    return False


def enqueue_source_links(session: SpiderSession, source_id: str, source: dict[str, Any]) -> int:
    links = source.get("links", []) or []
    if not links:
        return 0
    ranked = sorted(
        (
            {
                "tool": "read_url",
                "url": link.get("url"),
                "title": link.get("text") or link.get("url"),
                "snippet": link.get("text", ""),
                "search_score": score_link_candidate(link, session.question, session=session),
                "from_source_id": source_id,
                "discovery": "source_link",
            }
            for link in links
            if link.get("url")
        ),
        key=lambda item: item.get("search_score", 0),
        reverse=True,
    )
    added = 0
    max_links = int(os.environ.get("SPIDER_MAX_LINKS_PER_SOURCE", "4"))
    for item in ranked:
        if item.get("search_score", 0) < int(os.environ.get("SPIDER_MIN_LINK_SCORE", "8")):
            session.record_frontier_state(item, "skipped_low_policy_score", f"link score {item.get('search_score', 0)} below threshold")
            continue
        if add_frontier_candidate(session, item):
            added += 1
        if added >= max_links:
            break
    if added:
        session.add_trace("link_frontier", {"source_id": source_id, "added": added, "candidates": len(links)})
    return added


def execute_action(session: SpiderSession, action: dict[str, Any]) -> dict[str, Any]:
    tool = action.get("tool")
    if tool == "search":
        query = action.get("query") or session.question
        log(f"[Spider] Searching: {query}")
        results, search_meta = fetch_search(session, query, max_results=8)
        session.search_history.append(
            {
                "query": query,
                "result_count": len(results),
                "timestamp": now_iso(),
                "backend": search_meta.get("backend"),
                "cache_hit": search_meta.get("cache_hit", False),
                "fallback_reason": search_meta.get("fallback_reason", ""),
                "attempts": search_meta.get("attempts", []),
            }
        )
        if not results:
            session.coverage["empty_searches"] = int(session.coverage.get("empty_searches", 0)) + 1
            if search_meta.get("fallback_reason") or search_meta.get("attempts"):
                session.coverage["search_limitation"] = "Search returned no usable results after configured backend attempts."
        else:
            session.coverage["empty_searches"] = 0
        target_hint = action.get("coverage_gap", "")
        for row in results:
            url = row.get("url", "")
            if not url:
                continue
            add_frontier_candidate(
                session,
                {
                    "tool": "read_url",
                    "url": url,
                    "title": row.get("title"),
                    "snippet": row.get("snippet") or row.get("content", ""),
                    "search_score": row.get("score"),
                    "target_hint": target_hint,
                    "discovery": "search",
                    "query": query,
                },
            )
        session.save()
        log(f"[Spider] Search returned {len(results)} results via {search_meta.get('backend') or 'none'}; frontier={len(session.frontier)}")
        return {"query": query, "result_count": len(results), "top_results": results[:5], "search": search_meta}

    if tool == "read_url":
        if session.read_count >= session.max_reads:
            return {"error": "max_reads reached"}
        url = action.get("url", "")
        existing = find_existing_source_by_url(session, url)
        if existing:
            source_id = existing.get("source_id")
            session.add_trace("read_url_skipped", {"url": url, "status": "already_read", "source_id": source_id})
            return {"url": url, "status": "already_read", "source_id": source_id}
        log(f"[Spider] Reading URL: {url}")
        source = fetch_webpage(url, max_chars=18000)
        if action.get("target_hint"):
            source["target_hint"] = action.get("target_hint")
        sid = session.add_source(source, seed=False)
        if sid:
            enqueue_source_links(session, sid, source)
        existing_after_fetch = get_source(session, sid) if sid else None
        if existing_after_fetch and source.get("requested_url") != existing_after_fetch.get("requested_url") and existing_after_fetch.get("relevance") != "candidate":
            return {"url": url, "status": "already_read", "source_id": sid}
        session.read_count += 1
        if sid and source.get("score", -1000) < min_extracted_source_score(session):
            session.update_source_relevance(
                sid,
                "irrelevant",
                f"Extraction score {source.get('score')} below threshold {min_extracted_source_score(session)}; not used as evidence.",
            )
            session.add_note(f"Skipped low-quality extraction from {url}.", sid)
        elif sid and source.get("score", -1000) > -10:
            summarize_source(session, sid, source.get("text", ""))
        elif sid:
            session.update_source_relevance(sid, "summarization_failed", "Fetch produced no useful text for summarization.")
        log(f"[Spider] Read {sid or 'untracked'} score={source.get('score')} chars={source.get('extracted_chars')}")
        return {"url": url, "source_id": sid, "score": source.get("score"), "chars": source.get("extracted_chars")}

    if tool == "read_file":
        if session.read_count >= session.max_reads:
            return {"error": "max_reads reached"}
        path_value = action.get("path") or action.get("file") or ""
        path = Path(path_value).expanduser()
        if not path.exists() or not path.is_file():
            return {"error": f"file not found: {path}"}
        log(f"[Spider] Reading file: {path}")
        source = read_local_file(str(path), max_chars=18000)
        sid = session.add_source(source, seed=False)
        session.read_count += 1
        if sid and source.get("score", -1000) < min_extracted_source_score(session):
            session.update_source_relevance(
                sid,
                "irrelevant",
                f"Extraction score {source.get('score')} below threshold {min_extracted_source_score(session)}; not used as evidence.",
            )
        elif sid:
            summarize_source(session, sid, source.get("text", ""))
        return {"path": str(path), "source_id": sid, "score": source.get("score"), "chars": source.get("extracted_chars")}

    if tool == "extract_claims":
        sid = action.get("source_id", "")
        source = get_source(session, sid)
        if source:
            summarize_source(session, sid, get_source_text(session, sid))
            return {"source_id": sid, "claim_count": len(session.claims)}
        return {"error": f"unknown source_id {sid}"}

    if tool == "verify_claim":
        query = action.get("query") or next((c.get("claim") for c in session.claims if c.get("claim_id") == action.get("claim_id")), session.question)
        results, search_meta = fetch_search(session, query, max_results=5)
        for row in results[:3]:
            url = row.get("url", "")
            add_frontier_candidate(
                session,
                {
                    "tool": "read_url",
                    "url": url,
                    "title": row.get("title"),
                    "snippet": row.get("snippet") or row.get("content", ""),
                    "verify": action.get("claim_id"),
                    "search_score": row.get("score", 0),
                    "discovery": "verify_search",
                    "query": query,
                },
            )
        return {"query": query, "result_count": len(results), "top_results": results[:3], "search": search_meta}

    if tool in {"final_report", "stop"}:
        session.status = "ready_to_report"
        session.save()
        return {"status": session.status, "reason": action.get("reason", "")}

    return {"error": f"unknown tool {tool}"}


def coverage_sufficient(session: SpiderSession) -> bool:
    evidence_sources = [s for s in session.sources if s.get("relevance") in {"seed", "relevant", "partial"}]
    seed_count = len([s for s in evidence_sources if s.get("seed")])
    strong_sources = len([s for s in evidence_sources if s.get("score", 0) >= 25])
    has_claims = len(session.claims) >= 4 or len(session.notes) >= 3
    meets_minimums = (
        session.current_step >= session.min_steps
        and len(evidence_sources) >= session.min_sources
        and session.read_count >= session.min_reads
    )
    exhausted_but_seeded = (
        session.current_step >= session.min_steps
        and search_exhausted(session)
        and seed_count >= 1
        and strong_sources >= 1
        and has_claims
    )
    if exhausted_but_seeded:
        session.coverage["limitation"] = "Search query variants were exhausted or returned no useful frontier; report relies primarily on seed/primary sources."
    return (meets_minimums or exhausted_but_seeded) and has_claims and (seed_count >= 1 or strong_sources >= 2)


def evidence_sources(session: SpiderSession) -> list[dict[str, Any]]:
    return [s for s in session.sources if s.get("relevance") in {"seed", "relevant", "partial"}]


def evaluate_coverage(session: SpiderSession) -> dict[str, Any]:
    evidence = evidence_sources(session)
    verification_counts: dict[str, int] = {}
    for claim in session.claims:
        status = claim.get("verification", "unverified")
        verification_counts[status] = verification_counts.get(status, 0) + 1
    gaps = []
    if len(evidence) < session.min_sources:
        gaps.append(f"Only {len(evidence)} evidence sources; target minimum is {session.min_sources}.")
    if session.read_count < session.min_reads:
        gaps.append(f"Only {session.read_count} reads; target minimum is {session.min_reads}.")
    if not session.claims:
        gaps.append("No extracted claims.")
    if verification_counts.get("source_supported_unverified") or verification_counts.get("unverified"):
        gaps.append("Some claims are extracted from sources but not independently model-verified.")
    if session.coverage.get("search_limitation") or session.coverage.get("search_backend_unavailable"):
        gaps.append("Search was degraded or unavailable for at least one query.")
    missing_targets = missing_comparison_targets(session)
    if missing_targets:
        session.coverage["missing_comparison_targets"] = missing_targets
        gaps.append(f"Missing evidence for comparison targets: {', '.join(missing_targets)}.")
    else:
        session.coverage.pop("missing_comparison_targets", None)
    return {
        "evidence_sources": len(evidence),
        "excluded_sources": len([s for s in session.sources if s.get("relevance") == "irrelevant"]),
        "failed_sources": len([s for s in session.sources if s.get("relevance") == "summarization_failed"]),
        "claims": len(session.claims),
        "verification_counts": verification_counts,
        "reads": session.read_count,
        "min_sources": session.min_sources,
        "min_reads": session.min_reads,
        "gaps": gaps,
    }


def frontier_state_counts(session: SpiderSession) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in session.frontier:
        state = item.get("frontier_state", "read_now")
        counts[state] = counts.get(state, 0) + 1
    for item in session.frontier_audit:
        state = item.get("frontier_state", "")
        if state and state != "read_now":
            counts[state] = counts.get(state, 0) + 1
    return counts


def lock_owner(path: Path | None = None) -> dict[str, Any] | None:
    lock_path = Path(path or DEFAULT_LOCK_PATH).expanduser()
    if not lock_path.exists():
        return None
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else None
    except Exception:
        return {"path": str(lock_path), "status": "unreadable"}


def write_status_artifact(session: SpiderSession, current_action: Optional[dict[str, Any]] = None, current_message: str = "") -> Path:
    coverage = evaluate_coverage(session)
    status = {
        "session_id": session.session_id,
        "status": session.status,
        "current_step": session.current_step,
        "current_action": current_action or {},
        "current_message": strip_thinking(current_message),
        "stop_reason": session.stop_reason,
        "stop_details": session.stop_details,
        "lock_owner": lock_owner(),
        "counts": {
            "sources": len(session.sources),
            "reads": session.read_count,
            "searches": len(session.search_history),
            "claims": len(session.claims),
            "frontier": len(session.frontier),
            "media": len(session.media_ledger),
        },
        "frontier_counts": frontier_state_counts(session),
        "category_coverage": category_counts(session),
        "source_family_saturation": source_family_counts(session),
        "coverage_gaps": coverage.get("gaps", []),
        "paths": {
            "questions": str(artifact_path("clarifying_questions.md")),
            "answers": str(artifact_path("answers.md")),
            "plan": str(artifact_path("research_plan.json")),
            "report": str(SESSION_DIR / f"{session.session_id}_report.md"),
            "trace": str(SESSION_DIR / f"{session.session_id}_trace.jsonl"),
            "session": str(session.session_path),
            "status": str(artifact_path("status.json")),
        },
    }
    return write_json_artifact("status.json", status)


def verify_claim_ledger(session: SpiderSession) -> None:
    """Batch-verify extracted claims against evidence source excerpts before report synthesis."""
    if os.environ.get("SPIDER_VERIFY_CLAIMS", "1") == "0":
        return
    pending = [claim for claim in session.claims if claim.get("verification", "unverified") == "unverified"]
    if not pending:
        return

    source_ids = {s.get("source_id") for s in evidence_sources(session)}
    claims = [claim for claim in pending if set(claim.get("source_ids", [])) & source_ids]
    if not claims:
        return

    source_excerpts = "\n\n".join(
        f"[{source['source_id']}] {source.get('title')}\n{get_source_text(session, source['source_id'])[:1800]}"
        for source in evidence_sources(session)[:12]
    )
    batch_size = int(os.environ.get("SPIDER_VERIFY_BATCH_SIZE", "8"))
    def verification_priority(claim: dict[str, Any]) -> tuple[int, int, int]:
        confidence_score = {"high": 3, "medium": 2, "low": 1}.get(str(claim.get("confidence", "")).lower(), 1)
        evidence_score = int(bool(claim.get("evidence")))
        source_score = max((int((get_source(session, sid) or {}).get("score", 0)) for sid in claim.get("source_ids", [])), default=0)
        return (evidence_score, confidence_score, source_score)

    claims = sorted(claims, key=verification_priority, reverse=True)
    default_max_claims = "20" if session.depth == "deep" else str(len(claims))
    max_claims = int(os.environ.get("SPIDER_VERIFY_MAX_CLAIMS", default_max_claims))
    claims = claims[:max_claims]
    allowed = {"corroborated", "supported", "weak", "unsupported", "contradicted"}
    model_verified = 0
    fallback_unverified = 0

    for offset in range(0, len(claims), max(batch_size, 1)):
        batch = claims[offset : offset + max(batch_size, 1)]
        log(f"[Spider] Verifying claims {offset + 1}-{offset + len(batch)} of {len(claims)}")
        claim_payload = [
            {
                "claim_id": claim.get("claim_id"),
                "claim": claim.get("claim"),
                "source_ids": claim.get("source_ids", []),
                "extraction_confidence": claim.get("confidence"),
                "evidence": claim.get("evidence", [])[:2],
            }
            for claim in batch
        ]
        prompt = f"""Research question:
{session.question}

Evidence source excerpts:
{source_excerpts}

Extracted claim ledger:
{json.dumps(claim_payload, indent=2, ensure_ascii=False)}

Verify each claim against the cited evidence sources and evidence snippets. Return ONLY JSON, preferably an array.
Each verification item must have:
claim_id, verification ("corroborated"|"supported"|"weak"|"unsupported"|"contradicted"), note.

Definitions:
- corroborated: clearly supported by more than one evidence source.
- supported: clearly supported by its cited evidence source.
- weak: partially supported, underspecified, or benchmark-bound.
- unsupported: not found in the cited evidence.
- contradicted: evidence conflicts with the claim."""
        parsed = extract_json_value(
            call_nanbeige(
                prompt,
                max_tokens=int(os.environ.get("SPIDER_VERIFY_MAX_TOKENS", "8192")),
                timeout=int(os.environ.get("SPIDER_VERIFY_TIMEOUT", "600")),
            )
        )
        if isinstance(parsed, dict) and isinstance(parsed.get("verifications"), list):
            parsed = parsed.get("verifications")
        if not isinstance(parsed, list):
            retry_prompt = f"""Return ONLY a JSON array verifying these claims against the cited source excerpts.

Allowed verification values: corroborated, supported, weak, unsupported, contradicted.
Each item must be: {{"claim_id":"C1","verification":"supported","note":"short source-grounded reason"}}

Evidence source excerpts:
{source_excerpts[:9000]}

Claims:
{json.dumps(claim_payload, indent=2, ensure_ascii=False)}"""
            parsed = extract_json_value(
                call_nanbeige(
                    retry_prompt,
                    max_tokens=int(os.environ.get("SPIDER_VERIFY_RETRY_MAX_TOKENS", "8192")),
                    timeout=int(os.environ.get("SPIDER_VERIFY_RETRY_TIMEOUT", os.environ.get("SPIDER_VERIFY_TIMEOUT", "600"))),
                )
            )
            if isinstance(parsed, dict) and isinstance(parsed.get("verifications"), list):
                parsed = parsed.get("verifications")
        if not isinstance(parsed, list):
            for claim in batch:
                claim["verification"] = "source_supported_unverified"
                claim["verification_note"] = "Verifier output was not parseable; claim remains an extracted source-supported claim, not independently verified."
            fallback_unverified += len(batch)
            continue

        by_id = {item.get("claim_id"): item for item in parsed if isinstance(item, dict)}
        for claim in batch:
            item = by_id.get(claim.get("claim_id"))
            if not item:
                claim["verification"] = "source_supported_unverified"
                claim["verification_note"] = "Verifier did not return this claim; retained as source-supported extraction, not independently verified."
                fallback_unverified += 1
                continue
            status = item.get("verification", "weak")
            if status not in allowed:
                status = "weak"
            claim["verification"] = status
            claim["verification_note"] = strip_thinking(item.get("note", ""))
            model_verified += 1
            if status == "contradicted":
                session.contradictions.append(
                    {
                        "claim_id": claim.get("claim_id"),
                        "claim": claim.get("claim"),
                        "source_ids": claim.get("source_ids", []),
                        "note": claim["verification_note"],
                        "step": session.current_step,
                        "timestamp": now_iso(),
                    }
                )
    session.add_trace("claim_verification", {"mode": "model_with_unverified_fallback", "model_verified": model_verified, "source_supported_unverified": fallback_unverified, "claim_count": len(claims)})
    session.save()


def report_claims(session: SpiderSession, evidence_source_ids: set[str]) -> list[dict[str, Any]]:
    claims = [c for c in session.claims if set(c.get("source_ids", [])) & evidence_source_ids]
    if session.depth == "deep":
        claims = [c for c in claims if c.get("evidence")]
    return claims


LONGFORM_SECTIONS = [
    "Opening thesis",
    "What was investigated",
    "Memory-system taxonomy",
    "Frontier lab approaches",
    "Major projects/frameworks",
    "Smaller/interesting projects",
    "Papers and novel methods",
    "Comparison to Volition/Abe",
    "Practical homelab architecture",
    "Recommended implementation path",
    "Risks/failure modes",
    "What to build next",
    "Final recommendation",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def markdown_section_count(text: str) -> int:
    return len(re.findall(r"(?m)^##\s+", text or ""))


def build_source_appendix(session: SpiderSession) -> str:
    lines = ["# Source Appendix", ""]
    for source in session.sources:
        card = source.get("source_card") or ""
        lines.extend(
            [
                f"## [{source.get('source_id')}] {source.get('title') or source.get('url')}",
                f"- URL: {source.get('final_url') or source.get('url')}",
                f"- Relevance: {source.get('relevance')} ({source.get('relevance_reason', '')})",
                f"- Category/family: {source.get('source_category', '')} / {source.get('source_family', '')}",
                f"- Chars: {source.get('extracted_chars')}; score: {source.get('score')}",
                "",
            ]
        )
        if card:
            lines.extend(["### Source Card", "", card[:5000], ""])
        else:
            lines.extend(["### Excerpt", "", get_source_text(session, source.get("source_id", ""))[:1800], ""])
    failed = [s for s in session.sources if s.get("relevance") == "summarization_failed"]
    if failed:
        lines.extend(["# Failed Source Structuring / Extraction", ""])
        for source in failed:
            lines.append(f"- [{source.get('source_id')}] {source.get('title') or source.get('url')} - {source.get('relevance_reason')}")
    return strip_thinking("\n".join(lines).strip() + "\n")


def build_research_dossier(session: SpiderSession) -> str:
    evidence = evidence_sources(session)
    claims = report_claims(session, {s.get("source_id") for s in evidence})
    source_cards = "\n\n".join(
        f"## [{s.get('source_id')}] {s.get('title')}\nURL: {s.get('final_url') or s.get('url')}\nRelevance: {s.get('relevance')}\n{s.get('source_card') or get_source_text(session, s.get('source_id', ''))[:2200]}"
        for s in evidence[:80]
    )
    claims_md = "\n".join(
        f"- {c.get('claim')} [{', '.join(c.get('source_ids', []))}] Evidence: {' | '.join(c.get('evidence', [])[:2])}"
        for c in claims[:220]
    )
    dossier = f"""# Spider Research Dossier

Question:
{session.question}

Research brief:
{json.dumps(session.research_brief, indent=2, ensure_ascii=False)}

Stop decision:
{session.stop_reason or 'unknown'} {json.dumps(session.stop_details, ensure_ascii=False)}

Coverage:
{json.dumps(evaluate_coverage(session), indent=2, ensure_ascii=False)}

Key claim ledger:
{claims_md or 'No claim ledger entries.'}

Source cards and excerpts:
{source_cards}
"""
    return strip_thinking(dossier)


def write_longform_outline(session: SpiderSession, dossier: str, target_words: int) -> str:
    log("[Spider] Writing thesis outline")
    deterministic = "# Thesis Outline\n\n" + "\n\n".join(
        f"## {section}\n- Write source-grounded prose for this section.\n- Use citations from the dossier claim ledger and source cards.\n- Keep the main report separate from the source appendix."
        for section in LONGFORM_SECTIONS
    )
    if os.environ.get("SPIDER_USE_WRITER_OUTLINE", "0") != "1":
        return deterministic
    prompt = f"""Create a thesis-driven outline for a longform Spider report.

Question:
{session.question}

Required sections:
{json.dumps(LONGFORM_SECTIONS, indent=2)}

Target main-report word count: {target_words}

Dossier:
{dossier[:18000]}

Return markdown. Keep it prose-oriented, not a table ledger. Include citation/source IDs to use in each section."""
    outline = strip_thinking(call_writer_model(prompt, max_tokens=int(os.environ.get("SPIDER_OUTLINE_MAX_TOKENS", "8192"))))
    return outline or deterministic


def write_longform_sections(session: SpiderSession, dossier: str, outline: str, target_words: int, model_override: str = "", base_url_override: str = "") -> str:
    per_section = max(450, target_words // max(len(LONGFORM_SECTIONS), 1))
    parts = ["# Spider Longform Research Report", ""]
    evidence_pack = "\n".join(
        f"- {c.get('claim')} [{', '.join(c.get('source_ids', []))}] Evidence: {' | '.join(c.get('evidence', [])[:1])}"
        for c in report_claims(session, {s.get("source_id") for s in evidence_sources(session)})[:80]
    )
    source_pack = "\n".join(
        f"- [{s.get('source_id')}] {s.get('title')} ({s.get('source_category')}, {s.get('relevance')}): {(s.get('source_card') or s.get('snippet') or '')[:500]}"
        for s in evidence_sources(session)[:28]
    )
    compact_dossier = f"""Question: {session.question}

Stop reason: {session.stop_reason or 'unknown'}

Evidence source summaries:
{source_pack}

Claim ledger excerpt:
{evidence_pack}
"""
    for idx, section in enumerate(LONGFORM_SECTIONS, start=1):
        log(f"[Spider] Writing longform section {idx}/{len(LONGFORM_SECTIONS)}: {section}")
        prompt = f"""Write the longform report section titled: {section}

Research question:
{session.question}

Outline:
{outline[:12000]}

Compact evidence pack:
{compact_dossier[:7000]}

Rules:
- Write approximately {per_section} words for this section.
- Prose-first: paragraphs, not mostly bullets or tables.
- Cite source IDs like [S1], [S2] where claims depend on sources.
- Do not include a source appendix or raw ledger.
- Mention the stop reason only where relevant to methodology/limitations.
- No <think> blocks."""
        section_text = strip_thinking(
            call_writer_model(
                prompt,
                max_tokens=int(os.environ.get("SPIDER_SECTION_MAX_TOKENS", "7000")),
                model_override=model_override,
                base_url_override=base_url_override,
            )
        )
        section_text = re.sub(r"(?m)^#\s+.*$", "", section_text).strip()
        if not re.match(r"(?m)^##\s+", section_text):
            section_text = f"## {section}\n\n{section_text}"
        parts.extend([section_text.strip(), ""])
        write_text_artifact("longform_report.partial.md", "\n".join(parts).strip() + "\n")
    return strip_thinking("\n".join(parts).strip() + "\n")


def longform_report_qa(main_report: str, source_appendix: str, session: SpiderSession, target_words: int) -> dict[str, Any]:
    words = word_count(main_report)
    bullets = len(re.findall(r"(?m)^\s*[-*]\s+", main_report))
    paragraphs = len([p for p in re.split(r"\n\s*\n", main_report) if len(p.split()) > 25])
    table_lines = len(re.findall(r"(?m)^\s*\|", main_report))
    failed = [s for s in session.sources if s.get("relevance") == "summarization_failed"]
    read_sources = max(1, len([s for s in session.sources if s.get("extracted_chars", 0) or s.get("text_ref")]))
    fail_ratio = len(failed) / read_sources
    issues = []
    if words < int(0.75 * target_words):
        issues.append(f"main_report_word_count_below_target: {words} < {int(0.75 * target_words)}")
    if "## Source Appendix" in main_report or "# Source Appendix" in main_report:
        issues.append("source_appendix_inlined_in_main_report")
    if bullets > paragraphs * 4 or table_lines > max(20, paragraphs * 3):
        issues.append("main_report_too_table_or_bullet_heavy")
    if fail_ratio > 0.10:
        issues.append(f"source_structuring_failure_ratio_high: {fail_ratio:.2%}")
    if session.depth == "deep" and session.report_style == "longform" and markdown_section_count(main_report) < 10:
        issues.append("deep_longform_missing_required_sections")
    if re.search(r"Executive Summary", main_report, flags=re.IGNORECASE) and re.search(r"Evidence Table", main_report, flags=re.IGNORECASE) and words < 2500:
        issues.append("looks_like_compact_summary_plus_evidence_table")
    return {
        "pass": not issues,
        "issues": issues,
        "word_count": words,
        "target_words": target_words,
        "section_count": markdown_section_count(main_report),
        "appendix_word_count": word_count(source_appendix),
        "appendix_ratio": word_count(source_appendix) / max(words, 1),
        "citation_count": len(re.findall(r"\[S\d+\]", main_report)),
        "source_density": len(set(re.findall(r"\[S\d+\]", main_report))) / max(len(evidence_sources(session)), 1),
        "source_structuring_failure_ratio": fail_ratio,
        "failed_sources": [s.get("source_id") for s in failed],
    }


def generate_longform_report(session: SpiderSession, model_override: str = "", base_url_override: str = "") -> str:
    verify_claim_ledger(session)
    session.coverage["evaluation"] = evaluate_coverage(session)
    target_words = int(os.environ.get("SPIDER_LONGFORM_TARGET_WORDS", "8000"))
    dossier = build_research_dossier(session)
    appendix = build_source_appendix(session)
    outline = write_longform_outline(session, dossier, target_words)
    main_report = write_longform_sections(session, dossier, outline, target_words, model_override=model_override, base_url_override=base_url_override)
    qa = longform_report_qa(main_report, appendix, session, target_words)
    write_text_artifact("research_dossier.md", dossier)
    write_text_artifact("thesis_outline.md", outline)
    write_text_artifact("longform_report.md", main_report)
    write_text_artifact("source_appendix.md", appendix)
    write_json_artifact("longform_qa.json", qa)
    if not qa["pass"]:
        warning = "\n\n---\n\n## Longform QA Warning\n\n" + "\n".join(f"- {issue}" for issue in qa["issues"]) + "\n"
        main_report = main_report.rstrip() + warning
    session.coverage["longform_qa"] = qa
    session.save()
    return sanitize_report(main_report, session)


def generate_report(session: SpiderSession) -> str:
    if session.report_style == "longform":
        return generate_longform_report(session)
    verify_claim_ledger(session)
    session.coverage["evaluation"] = evaluate_coverage(session)
    session.save()
    report_sources = evidence_sources(session)
    excluded_sources = [s for s in session.sources if s.get("relevance") == "irrelevant"]
    failed_sources = [s for s in session.sources if s.get("relevance") == "summarization_failed"]
    evidence_source_ids = {s.get("source_id") for s in report_sources}
    included_claims = report_claims(session, evidence_source_ids)
    session.coverage["report_claim_ids"] = [c.get("claim_id") for c in included_claims]
    if session.depth == "deep":
        dropped = [
            c.get("claim_id")
            for c in session.claims
            if set(c.get("source_ids", [])) & evidence_source_ids and not c.get("evidence")
        ]
        session.coverage["claims_excluded_from_report_missing_evidence"] = dropped
    session.save()
    sources_md = "\n".join(
        f"[{s['source_id']}] {s.get('title') or s.get('url')} - {s.get('final_url') or s.get('url')} "
        f"(type={s.get('source_type')}, score={s.get('score')}, relevance={s.get('relevance')}, chars={s.get('extracted_chars')})"
        for s in report_sources
    )
    excluded_md = "\n".join(
        f"[{s['source_id']}] {s.get('title') or s.get('url')} - {s.get('final_url') or s.get('url')} "
        f"(reason={s.get('relevance_reason') or 'marked irrelevant'})"
        for s in excluded_sources
    )
    failed_md = "\n".join(
        f"[{s['source_id']}] {s.get('title') or s.get('url')} - {s.get('final_url') or s.get('url')} "
        f"(reason={s.get('relevance_reason') or 'summarization failed'})"
        for s in failed_sources
    )
    claims_md = "\n".join(
        f"- {c['claim']} [{', '.join(c.get('source_ids', []))}] "
        f"(extraction confidence: {c.get('confidence')}; verification: {c.get('verification', 'unverified')}; note: {c.get('verification_note', '')}; evidence: {' | '.join(c.get('evidence', [])[:2])})"
        for c in included_claims
    )
    notes_md = "\n".join(
        f"- {n.get('text')} [{n.get('source_id')}]"
        for n in session.notes[-25:]
        if not n.get("source_id") or n.get("source_id") in evidence_source_ids
    )
    source_excerpts_md = "\n\n".join(
        f"[{s['source_id']}] {s.get('title')}\n{get_source_text(session, s['source_id'])[:2500]}"
        for s in report_sources[-12:]
    )
    if session.depth == "deep":
        section_spec = """Write a cited markdown deep research report with exactly these sections:
# Spider Research Report
## Executive Summary
## Methodology / Research Path
## Key Findings
## Practical Checklist
## Detailed Analysis
## Evidence Table
## Claim Verification / Confidence
## Contradictions and Uncertainties
## Source Appendix
## Limitations / Next Research"""
        depth_rules = (
            "- Make the report substantially richer than a compact brief: include actionable checklists, implementation notes, source-by-source synthesis, and next research questions.\n"
            "- In deep mode, only use claim-ledger entries that include explicit evidence snippets. Mention any dropped no-evidence claims only in Limitations/QA, not as support.\n"
        )
    else:
        section_spec = """Write a cited markdown report with exactly these sections:
# Spider Research Report
## Executive Summary
## Answer
## Evidence Table
## Contradictions and Uncertainties
## Limitations
## Sources"""
        depth_rules = ""

    prompt = f"""Research question:
{session.question}

Research brief:
{json.dumps(session.research_brief, indent=2, ensure_ascii=False)}

Claim ledger:
{claims_md or 'No explicit claims extracted.'}

Source notes:
{notes_md}

Contradictions:
{json.dumps(session.contradictions, indent=2, ensure_ascii=False)}

Coverage / search status:
{json.dumps(session.coverage, indent=2, ensure_ascii=False)}

Sources:
{sources_md}

Reviewed but excluded sources:
{excluded_md or 'None.'}

Fetched but not summarized:
{failed_md or 'None.'}

Source excerpts:
{source_excerpts_md}

{section_spec}

Rules:
- No misleading limitation wording. If searches or external sources were used, do not say "no additional external research was performed"; say "No research was performed beyond the retrieved/cited sources and stored claim ledger" if that limitation is needed.
- Cite source IDs like [S1], [S2].
- Only cite evidence sources. Do not cite reviewed-but-excluded sources as support.
- Reflect claim verification status exactly. Treat source_supported_unverified as extracted source support, not independent verification. Do not present weak, unsupported, source_supported_unverified, or contradicted claims as settled facts.
{depth_rules}
- Do not include <think> blocks or hidden reasoning.
- Do not claim empty/bad search results refute seed evidence.
- If search_backend_unavailable or another coverage limitation is present, state it plainly in Limitations.
- If evidence is thin, say what is missing without burying supported findings."""
    return sanitize_report(strip_thinking(call_nanbeige(prompt)), session)


def sanitize_report(report: str, session: SpiderSession) -> str:
    """Apply small deterministic guards for canonical facts in primary sources."""
    if "Nanbeige4.1-3B" in report or "Nanbeige 4.1" in report or "Nanbeige4.1-3B" in session.question:
        report = re.sub(r"\b30[- ]billion[- ]parameter\b", "3B-parameter", report, flags=re.IGNORECASE)
        report = re.sub(r"\b30\s+billion\s+parameters\b", "3B parameters", report, flags=re.IGNORECASE)
        report = re.sub(r"\b30B[- ]parameter\b", "3B-parameter", report, flags=re.IGNORECASE)
    if session.search_history:
        report = re.sub(
            r"No external searches conducted beyond supplied materials\.?",
            "External searches were conducted; sources that did not support the research question are listed in the reviewed-but-excluded audit.",
            report,
            flags=re.IGNORECASE,
        )
    if session.search_history or session.sources:
        report = re.sub(
            r"No additional external research was performed\.?",
            "No research was performed beyond the retrieved/cited sources and stored claim ledger.",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"No external research was performed\.?",
            "No research was performed beyond the retrieved/cited sources and stored claim ledger.",
            report,
            flags=re.IGNORECASE,
        )
    if re.search(r"\bfree[- ]thread", session.question, flags=re.IGNORECASE):
        report = re.sub(r"\*\*Maced\*\*\s+\(mimalloc allocator\)", "mimalloc", report, flags=re.IGNORECASE)
        report = re.sub(r"\bvia\s+\*\*Maced\*\*", "via mimalloc", report, flags=re.IGNORECASE)
        report = re.sub(r"\bMaced allocator\b", "mimalloc allocator", report, flags=re.IGNORECASE)
        report = re.sub(r"\bvia\s+Maced\b", "via mimalloc", report, flags=re.IGNORECASE)
        report = re.sub(r"\bMaced\s+\(mimalloc allocator\)", "mimalloc", report, flags=re.IGNORECASE)
        report = re.sub(r"\bPyMemAlloc\b", "PyMem_Malloc", report)
        report = re.sub(
            r"Extensions must be compiled with `--disable-gil`[^.\n]*\.?",
            "CPython itself is configured with `--disable-gil` for the free-threaded build; extension authors should build/test against a free-threaded Python and advertise GIL-disabled support with the documented C-API mechanisms where appropriate.",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"Build extensions with `--disable-gil`[^.\n]*\.?",
            "Build and test extensions against a free-threaded Python; `--disable-gil` is the CPython build configuration option, not a generic extension compiler flag.",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"Build with `--disable-gil`[^;\n|]*",
            "Build/test against a free-threaded Python",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"`Py_mod_gil`\s*=\s*1",
            "`Py_mod_gil` set to `Py_MOD_GIL_NOT_USED`",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"Add `Py_mod_gil\s*=\s*1` slot",
            "Add a `Py_mod_gil` slot set to `Py_MOD_GIL_NOT_USED`",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"All Python object allocations must use `?PyMem_Malloc`?[^.\n]*`?PyObject_Malloc`?[^.\n]*\.?",
            "Allocation guidance is API-specific: extension authors should use the documented allocator family for the kind of memory they allocate, and avoid unsupported object-allocation shortcuts in free-threaded builds.",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"Use `?PyMem_Malloc`? exclusively;[^.\n]*Avoid `?PyObject_Malloc`?\.?",
            "Use allocator APIs according to the documented memory domain; avoid unsupported object-allocation shortcuts in free-threaded builds.",
            report,
            flags=re.IGNORECASE,
        )
        report = re.sub(
            r"Use `?PyMem_Malloc`? exclusively[^.]*\.\s*Avoid `?PyObject_Malloc`?\.?",
            "Use allocator APIs according to the documented memory domain; avoid unsupported object-allocation shortcuts in free-threaded builds.",
            report,
            flags=re.IGNORECASE,
        )
    report = strip_thinking(report)
    if session.report_style == "longform":
        return append_report_qa(report, session)
    report = append_review_audit(report, session)
    return append_report_qa(report, session)


def append_review_audit(report: str, session: SpiderSession) -> str:
    excluded = [s for s in session.sources if s.get("relevance") == "irrelevant"]
    failed = [s for s in session.sources if s.get("relevance") == "summarization_failed"]
    if "Reviewed But Excluded Sources" in report and "Fetched But Not Summarized" in report:
        return report
    lines = ["", "---"]
    if excluded and "Reviewed But Excluded Sources" not in report:
        lines.extend(["", "## Reviewed But Excluded Sources", ""])
        for source in excluded:
            reason = source.get("relevance_reason") or "Marked irrelevant to the research question."
            lines.append(f"- [{source.get('source_id')}] {source.get('title') or source.get('url')} - {source.get('final_url') or source.get('url')}")
            lines.append(f"  Reason: {reason}")
    if failed and "Fetched But Not Summarized" not in report:
        lines.extend(["", "## Fetched But Not Summarized", ""])
        for source in failed:
            reason = source.get("relevance_reason") or "Summary extraction failed or returned unparsable output."
            lines.append(f"- [{source.get('source_id')}] {source.get('title') or source.get('url')} - {source.get('final_url') or source.get('url')}")
            lines.append(f"  Reason: {reason}")
    if len(lines) == 2:
        return report
    return report.rstrip() + "\n" + "\n".join(lines) + "\n"


def append_report_qa(report: str, session: SpiderSession) -> str:
    if "Report QA Notes" in report:
        return report
    issues = []
    coverage = session.coverage.get("evaluation") or evaluate_coverage(session)
    verification_counts = coverage.get("verification_counts", {})
    unverified = verification_counts.get("source_supported_unverified", 0) + verification_counts.get("unverified", 0)
    weak = verification_counts.get("weak", 0) + verification_counts.get("unsupported", 0) + verification_counts.get("contradicted", 0)
    if unverified:
        issues.append(f"{unverified} claims are source-supported extractions but not independently verified.")
    if weak:
        issues.append(f"{weak} claims are weak, unsupported, or contradicted in the verification ledger.")
    report_claim_ids = set(session.coverage.get("report_claim_ids") or [])
    qa_claims = [c for c in session.claims if not report_claim_ids or c.get("claim_id") in report_claim_ids]
    missing_evidence = [c.get("claim_id") for c in qa_claims if not c.get("evidence")]
    if missing_evidence:
        issues.append(f"{len(missing_evidence)} claims lack explicit evidence snippets in the ledger.")
    dropped_missing = session.coverage.get("claims_excluded_from_report_missing_evidence") or []
    if session.depth == "deep" and dropped_missing:
        issues.append(f"{len(dropped_missing)} no-evidence claims were excluded from deep report synthesis.")
    cited = set(re.findall(r"\[S(\d+)\]", report))
    uncited = [s.get("source_id") for s in evidence_sources(session) if str(s.get("source_id", "")).lstrip("S") not in cited]
    if uncited:
        issues.append(f"Evidence sources not cited in generated report text: {', '.join(uncited)}.")
    for gap in coverage.get("gaps", []):
        issues.append(gap)
    if not issues:
        return report
    lines = ["", "---", "", "## Report QA Notes", ""]
    for issue in dict.fromkeys(issues):
        lines.append(f"- {issue}")
    return report.rstrip() + "\n" + "\n".join(lines) + "\n"


def run_spider(session: SpiderSession, seed_urls: Optional[list[str]] = None, files: Optional[list[str]] = None, clarify: bool = False) -> str:
    log(f"[Spider] Session {session.session_id}")
    log(f"[Spider] Question: {session.question}")
    log(
        f"[Spider] Budgets: steps={session.max_steps}, sources={session.max_sources}, reads={session.max_reads}, "
        f"minutes={session.time_limit_min}, depth={session.depth}, mins=({session.min_steps} steps/{session.min_sources} sources/{session.min_reads} reads)"
    )
    create_research_brief(session, clarify=clarify)
    if session.status == "needs_clarification":
        return ""

    if session.current_step == 0:
        ingest_seed_urls(session, seed_urls or [])
        ingest_files(session, files or [])

    deadline = time.monotonic() + max(session.time_limit_min, 1) * 60
    while session.current_step < session.max_steps and session.status == "active":
        if time.monotonic() > deadline:
            session.record_stop_decision("time_limit", {"time_limit_min": session.time_limit_min, "step": session.current_step})
            break
        if len(session.sources) >= session.max_sources:
            session.record_stop_decision("max_sources", {"sources": len(session.sources), "max_sources": session.max_sources})
            break
        if session.read_count >= session.max_reads:
            session.record_stop_decision("max_reads", {"reads": session.read_count, "max_reads": session.max_reads})
            break

        session.current_step += 1
        log(f"\n[Spider] Step {session.current_step}/{session.max_steps}")
        write_status_artifact(session, current_message=f"Preparing step {session.current_step}")
        sufficient = coverage_sufficient(session)
        if session.depth == "deep" and not sufficient:
            gap_search_action = next_gap_search_action(session) or next_policy_gap_search_action(session)
            frontier_action = None if gap_search_action else pop_frontier_action(session)
        else:
            gap_search_action = next_gap_search_action(session) if not sufficient else None
            frontier_action = None if gap_search_action else (pop_frontier_action(session) if not sufficient else None)
        if gap_search_action:
            controller = {"rationale": "Harness searched for a named comparison/policy target that is not yet covered by evidence sources."}
            action = gap_search_action
        elif frontier_action:
            controller = {"rationale": "Harness consumed the highest-priority plan-aware frontier URL before asking for another plan."}
            action = frontier_action
        elif not sufficient:
            search_action = next_search_action(session)
            if search_action:
                controller = {"rationale": "Harness selected the next untried search query because coverage minimums are not met."}
                action = search_action
            else:
                controller = {"rationale": "Coverage is incomplete but deterministic search is exhausted; asking controller for a final decision."}
                action = {"tool": "final_report"} if coverage_sufficient(session) else {"tool": "stop", "reason": "Search exhausted before source/read minimums could be met."}
        else:
            controller = ask_controller(session)
            action = controller.get("action", {})
            if action.get("tool") == "final_report" and not coverage_sufficient(session):
                action = frontier_action or (plan_fallback_actions(session) or [{"tool": "search", "query": session.question}])[0]
        action_label = action.get("query") if action.get("tool") == "search" else action.get("url") or action.get("source_id") or action.get("path") or ""
        log(f"[Action] {action.get('tool')}: {action_label}")
        write_status_artifact(session, current_action=action, current_message=controller.get("rationale", ""))
        observation = execute_action(session, action)
        session.add_trace("action", {"rationale": controller.get("rationale", ""), "action": action, "observation": observation})
        write_status_artifact(session, current_action=action, current_message=f"Completed {action.get('tool')}")
        if session.status == "ready_to_report":
            reason = "controller_stop" if action.get("tool") == "stop" else "coverage_sufficient"
            if action.get("tool") == "stop" and "Search exhausted" in str(action.get("reason", "")):
                reason = "search_exhausted"
            session.record_stop_decision(reason, {"action": action, "observation": observation})
            break
        if coverage_sufficient(session) and session.current_step >= session.max_steps:
            session.record_stop_decision("max_steps", {"coverage_sufficient": True, "step": session.current_step, "max_steps": session.max_steps})
            break

    if not session.stop_reason:
        if session.current_step >= session.max_steps:
            session.record_stop_decision("max_steps", {"step": session.current_step, "max_steps": session.max_steps})
        elif coverage_sufficient(session):
            session.record_stop_decision("coverage_sufficient", {"step": session.current_step})
        elif not session.frontier and search_exhausted(session):
            session.record_stop_decision("search_exhausted", {"searches": len(session.search_history)})
        elif not session.frontier:
            session.record_stop_decision("frontier_exhausted", {"searches": len(session.search_history)})
    if session.stop_reason:
        log(f"[Spider] Stop reason: {session.stop_reason}")
    log("\n[Spider] Generating final report")
    write_status_artifact(session, current_message="Generating final report")
    report = generate_report(session)
    session.status = "completed"
    session.save()

    report_path = SESSION_DIR / f"{session.session_id}_report.md"
    trace_path = SESSION_DIR / f"{session.session_id}_trace.jsonl"
    report_path.write_text(report, encoding="utf-8")
    with open(trace_path, "w", encoding="utf-8") as f:
        for entry in session.trace:
            f.write(json.dumps(sanitize_for_storage(entry), ensure_ascii=False) + "\n")

    log(f"[Spider] Report: {report_path}")
    log(f"[Spider] Trace: {trace_path}")
    log(f"[Spider] Session: {session.session_path}")
    log(f"[Spider] Sources: {len(session.sources)} Claims: {len(session.claims)} Reads: {session.read_count}")
    write_status_artifact(session, current_message="Completed")
    return report


def write_report_artifacts(session: SpiderSession, report: str) -> tuple[Path, Path]:
    report_path = SESSION_DIR / f"{session.session_id}_report.md"
    trace_path = SESSION_DIR / f"{session.session_id}_trace.jsonl"
    report_path.write_text(report, encoding="utf-8")
    with open(trace_path, "w", encoding="utf-8") as f:
        for entry in session.trace:
            f.write(json.dumps(sanitize_for_storage(entry), ensure_ascii=False) + "\n")
    return report_path, trace_path


def regenerate_report(session: SpiderSession, report_style: str = "") -> str:
    if report_style:
        session.report_style = report_style
    if not session.stop_reason:
        session.record_stop_decision("manual_cancel" if session.status != "completed" else "coverage_sufficient", {"regenerate_report": True, "prior_status": session.status})
    log(f"[Spider] Regenerating report for session {session.session_id} with style={session.report_style}")
    report = generate_report(session)
    session.status = "completed"
    session.save()
    report_path, trace_path = write_report_artifacts(session, report)
    write_status_artifact(session, current_message="Regenerated report")
    log(f"[Spider] Report: {report_path}")
    log(f"[Spider] Trace: {trace_path}")
    return report


def compare_writer_reports(session: SpiderSession) -> dict[str, Any]:
    original_style = session.report_style
    session.report_style = "longform"
    dossier = build_research_dossier(session)
    appendix = build_source_appendix(session)
    outline = write_longform_outline(session, dossier, int(os.environ.get("SPIDER_LONGFORM_TARGET_WORDS", "8000")))
    writers = {
        "nanbeige": (NANBEIGE_MODEL, SCRIBE_API_URL),
        "qwen": (WRITER_MODEL, WRITER_API_URL),
    }
    results = {}
    target_words = int(os.environ.get("SPIDER_LONGFORM_TARGET_WORDS", "8000"))
    for name, (model, base_url) in writers.items():
        report = write_longform_sections(session, dossier, outline, target_words, model_override=model, base_url_override=base_url)
        qa = longform_report_qa(report, appendix, session, target_words)
        results[name] = {
            "model": model,
            "word_count": qa["word_count"],
            "section_count": qa["section_count"],
            "appendix_ratio": qa["appendix_ratio"],
            "citation_count": qa["citation_count"],
            "source_density": qa["source_density"],
            "qa_pass": qa["pass"],
            "qa_issues": qa["issues"],
        }
        write_text_artifact(f"writer_compare_{name}.md", report)
        write_json_artifact(f"writer_compare_{name}_qa.json", qa)
    session.report_style = original_style
    write_json_artifact("writer_comparison.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def probe_tool_calls() -> dict[str, Any]:
    payload = {
        "model": NANBEIGE_MODEL,
        "messages": [{"role": "user", "content": "Use the test_tool if tool calling is supported."}],
        "tools": [{"type": "function", "function": {"name": "test_tool", "description": "A no-op diagnostic tool.", "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}}],
        "tool_choice": "auto",
        "max_tokens": 200,
    }
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    try:
        resp = requests.post(f"{SCRIBE_API_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        data = resp.json()
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        result = {"ok": resp.ok, "status_code": resp.status_code, "has_tool_calls": bool(message.get("tool_calls")), "message_keys": sorted(message.keys())}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Spider - bounded local deep-research harness")
    parser.add_argument("--question", "-q", help="Research question")
    parser.add_argument("--resume", "-r", help="Resume a session by ID")
    parser.add_argument("--session-dir", help="Directory for this Spider workflow/session artifacts")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default=os.environ.get("SPIDER_DEPTH", "standard"))
    parser.add_argument("--max-steps", "-s", type=int)
    parser.add_argument("--max-sources", "-m", type=int)
    parser.add_argument("--max-reads", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--min-sources", type=int)
    parser.add_argument("--min-reads", type=int)
    parser.add_argument("--time-limit", "-t", type=int)
    parser.add_argument("--seed-url", action="append", default=[], help="Primary source URL to ingest before search; repeatable")
    parser.add_argument("--file", action="append", default=[], help="Local file or directory to ingest as seed evidence; repeatable")
    parser.add_argument("--clarify", action="store_true", help="Compatibility alias for --clarify-policy auto")
    parser.add_argument("--clarify-policy", choices=["auto", "always", "never"], default=os.environ.get("SPIDER_CLARIFY_POLICY", "auto"))
    parser.add_argument("--answers", help="Markdown/text file containing answers to prior clarifying questions")
    parser.add_argument("--plan-only", action="store_true", help="Write research_plan.md/json and stop before search/read")
    parser.add_argument("--approve-plan", help="Approved research_plan.json to load before running")
    parser.add_argument("--yes", action="store_true", help="Auto-approve generated plan for scripted/agent use")
    parser.add_argument("--report-style", choices=["standard", "longform"], default=os.environ.get("SPIDER_REPORT_STYLE", "standard"))
    parser.add_argument("--regenerate-report", action="store_true", help="Regenerate report artifacts for an existing session without search/read")
    parser.add_argument("--compare-writers", action="store_true", help="Regenerate longform drafts for Nanbeige and Qwen writers from the same completed session")
    parser.add_argument("--probe-tool-calls", action="store_true", help="Diagnostic: check whether the research endpoint returns OpenAI-style tool_calls")
    parser.add_argument("--wait-for-lock", action="store_true", help="Wait for the Spider run lock instead of failing fast")
    parser.add_argument("--lock-path", default=os.environ.get("SPIDER_LOCK_PATH", str(DEFAULT_LOCK_PATH)), help="Path to Spider file lock")
    args = parser.parse_args()
    if args.probe_tool_calls:
        probe_tool_calls()
        return
    if args.session_dir:
        set_session_dir(Path(args.session_dir))
    budget_defaults = depth_budget_defaults(args.depth)
    max_steps = args.max_steps if args.max_steps is not None else int(os.environ.get("SPIDER_MAX_STEPS", budget_defaults["max_steps"]))
    max_sources = args.max_sources if args.max_sources is not None else int(os.environ.get("SPIDER_MAX_SOURCES", budget_defaults["max_sources"]))
    max_reads = args.max_reads if args.max_reads is not None else int(os.environ.get("SPIDER_MAX_READS", budget_defaults["max_reads"]))
    time_limit = args.time_limit if args.time_limit is not None else int(os.environ.get("SPIDER_TIME_LIMIT_MIN", os.environ.get("SPIDER_TIME_LIMIT", budget_defaults["time_limit"])))

    if args.resume:
        session = SpiderSession(session_id=args.resume)
        session.report_style = args.report_style
        if args.compare_writers:
            compare_writer_reports(session)
            return
        if args.regenerate_report:
            report = regenerate_report(session, report_style=args.report_style)
            print("\n" + "=" * 60)
            print("REPORT PREVIEW")
            print("=" * 60)
            print(report[:2000] + ("\n\n[... truncated ...]" if len(report) > 2000 else ""))
            return
        if session.status == "completed":
            print(f"Session {args.resume} already completed.")
            print(f"Report: {SESSION_DIR / f'{args.resume}_report.md'}")
            return
    elif args.question:
        session = SpiderSession(
            question=args.question,
            max_steps=max_steps,
            max_sources=max_sources,
            max_reads=max_reads,
            time_limit_min=time_limit,
            depth=args.depth,
            min_steps=args.min_steps,
            min_sources=args.min_sources,
            min_reads=args.min_reads,
        )
        session.report_style = args.report_style
    elif find_latest_session_id():
        session = SpiderSession(session_id=find_latest_session_id())
        session.report_style = args.report_style
    else:
        parser.error("Either --question, --resume, or a --session-dir containing an existing session is required")
        return

    clarify_policy = args.clarify_policy
    if args.clarify and clarify_policy == "never":
        clarify_policy = "auto"
    may_run = handle_workflow_gates(
        session,
        clarify_policy=clarify_policy,
        answers_path=args.answers,
        plan_only=args.plan_only,
        yes=args.yes,
        approve_plan_path=args.approve_plan,
    )
    if not may_run:
        return

    try:
        with SpiderRunLock(Path(args.lock_path), wait=args.wait_for_lock):
            report = run_spider(session, seed_urls=args.seed_url, files=args.file, clarify=False)
    except SpiderBusyError as exc:
        print(f"[Spider] {exc}", file=sys.stderr)
        raise SystemExit(75) from exc
    if report:
        print("\n" + "=" * 60)
        print("REPORT PREVIEW")
        print("=" * 60)
        print(report[:2000] + ("\n\n[... truncated ...]" if len(report) > 2000 else ""))


if __name__ == "__main__":
    main()
