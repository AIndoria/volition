#!/usr/bin/env python3
"""
Morning Brief Generator.

Fast morning-room agenda builder for chat:watercooler. It uses BRIEF_MODEL /
BRIEF_API_URL by default, keeps posting opt-in, and never posts to
chat:synchronous.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRIEF_API_URL = os.environ.get("BRIEF_API_URL", os.environ.get("FLASH_API_URL", "http://127.0.0.1:8080/v1")).rstrip("/")
BRIEF_MODEL = os.environ.get("BRIEF_MODEL", os.environ.get("MODEL_FLASH", "gemma-4-26B-it")).replace("local/", "").replace(":thinking", "")
BRIEF_API_KEY = os.environ.get("BRIEF_API_KEY", "")
if not BRIEF_API_KEY and not any(local in BRIEF_API_URL for local in ("localhost", "127.0.0.1", "10.")):
    BRIEF_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
DEBUG_DIR = Path.home() / "logs" / "debug"


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser()


SEARCH_CACHE_DIR = expand_path(os.environ.get("MORNING_BRIEF_SEARCH_CACHE_DIR", str(Path.home() / ".cache" / "morning_brief")))
DEFAULT_OUTPUT_DIR = expand_path(os.environ.get("MORNING_BRIEF_OUTPUT_DIR", str(Path.home() / "morning_briefs")))
MORNING_BRIEF_API_CALLS = 0


CATEGORIES = {
    "ai": {
        "name": "AI / LLM / Runtime",
        "agenda_section": "AI / LLM / Runtime",
        "queries": [
            "LLM model release inference runtime agent tools",
            "AI open source model release benchmark runtime",
            "local LLM serving llama.cpp vLLM Ollama release",
        ],
    },
    "tech": {
        "name": "Tech / Homelab / Self-hosting",
        "agenda_section": "Tech / Homelab / Self-hosting",
        "queries": [
            "Linux self-hosting homelab Proxmox Docker storage networking",
            "open source infrastructure release Linux Docker Kubernetes",
            "homelab security storage backup networking",
        ],
    },
    "security": {
        "name": "Security / Watch",
        "agenda_section": "Action / Watch Items",
        "queries": [
            "critical CVE security advisory today Linux",
            "active exploitation vulnerability advisory CISA KEV",
            "major outage incident security advisory",
        ],
    },
    "world": {
        "name": "World / Politics Context",
        "agenda_section": "World / Politics Context",
        "queries": [
            "major world news today geopolitics technology policy",
            "global politics context technology economy today",
        ],
    },
    "infra": {
        "name": "Tech / Homelab / Self-hosting",
        "agenda_section": "Tech / Homelab / Self-hosting",
        "queries": [
            "Proxmox Docker Linux self hosted release",
            "storage networking homelab outage advisory",
        ],
    },
}


def strip_thinking(text: Any) -> str:
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


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return strip_thinking(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    return value


def normalized_query_key(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def brief_brave_api_key() -> str:
    return os.environ.get("MORNING_BRIEF_BRAVE_API_KEY", os.environ.get("SPIDER_BRAVE_API_KEY", "")).strip()


def search_backends() -> list[str]:
    configured = os.environ.get("MORNING_BRIEF_SEARCH_BACKENDS", "").strip()
    if configured:
        return [item.strip().lower() for item in configured.split(",") if item.strip()]
    return ["brave", "searxng"] if brief_brave_api_key() else ["searxng"]


def source_quality(url: str) -> int:
    host = urlparse(url).netloc.lower()
    if not host:
        return 0
    score = 8
    primary_domains = (
        "openai.com",
        "anthropic.com",
        "googleblog.com",
        "microsoft.com",
        "github.com",
        "huggingface.co",
        "arxiv.org",
        "cisa.gov",
        "nvd.nist.gov",
        "kernel.org",
        "debian.org",
        "ubuntu.com",
        "proxmox.com",
        "docker.com",
    )
    if any(host.endswith(domain) for domain in primary_domains):
        score += 20
    if any(news in host for news in ("reuters.com", "apnews.com", "bbc.", "theregister.com", "bleepingcomputer.com", "securityweek.com")):
        score += 12
    if any(low in host for low in ("medium.com", "forbes.com", "analyticsinsight", "sponsored", "pressrelease")):
        score -= 8
    return score


def score_candidate(item: dict[str, Any], category: str) -> int:
    text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    score = source_quality(item.get("url", ""))
    if category == "security" and any(w in text for w in ("critical", "cve-", "exploited", "patch", "advisory", "kev")):
        score += 15
    if category == "ai" and any(w in text for w in ("model", "llm", "runtime", "agent", "inference", "benchmark", "release")):
        score += 12
    if category in {"tech", "infra"} and any(w in text for w in ("linux", "docker", "proxmox", "storage", "network", "self-host", "homelab")):
        score += 12
    if category == "world" and any(w in text for w in ("war", "election", "policy", "economy", "sanction", "court", "government")):
        score += 8
    if len(text) < 80:
        score -= 4
    if not is_specific_candidate(item):
        score -= 25
    return score


def is_specific_candidate(item: dict[str, Any]) -> bool:
    title = (item.get("title") or "").lower()
    url = item.get("url", "")
    parsed = urlparse(url)
    path_bits = [bit for bit in parsed.path.strip("/").split("/") if bit]
    generic_titles = ("news", "latest", "breaking", "homepage", "github advisory database", "security news")
    if len(path_bits) <= 1 and any(token in title for token in generic_titles):
        return False
    if any(token in title for token in ("cve-", "advisory", "release", "released", "discloses", "incident", "vulnerability", "launches", "announces")):
        return True
    if "github.com" in parsed.netloc and any(bit in {"releases", "issues", "pull", "advisories"} for bit in path_bits):
        return True
    return len(path_bits) >= 2


def fetch_searxng(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "format": "json",
        "language": "en",
        "time_range": os.environ.get("MORNING_BRIEF_TIME_RANGE", "day"),
        "pageno": 1,
    }
    try:
        resp = requests.get(SEARXNG_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:max_results]
    except Exception as exc:
        print(f"[WARN] SearXNG fetch failed for {query!r}: {exc}", file=sys.stderr)
        return []
    out = []
    for row in results:
        out.append(
            {
                "title": strip_thinking(row.get("title", "")),
                "url": row.get("url", ""),
                "snippet": strip_thinking(row.get("content", "")),
                "published": row.get("publishedDate") or row.get("published"),
                "query": query,
            }
        )
    return out


def normalize_brave_results(data: dict[str, Any], query: str, max_results: int = 6) -> list[dict[str, Any]]:
    rows = (data.get("web") or {}).get("results") or []
    out = []
    for row in rows[:max_results]:
        extra = row.get("extra_snippets") or []
        snippet = row.get("description", "") or (extra[0] if extra else "")
        out.append(
            {
                "title": strip_thinking(row.get("title", "")),
                "url": row.get("url", ""),
                "snippet": strip_thinking(snippet),
                "published": row.get("age"),
                "query": query,
                "backend": "brave",
            }
        )
    return out


def fetch_brave(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    api_key = brief_brave_api_key()
    if not api_key:
        raise RuntimeError("Brave API key is not set")
    resp = requests.get(
        BRAVE_SEARCH_URL,
        params={"q": query, "count": max_results, "search_lang": "en", "safesearch": "moderate", "freshness": "pd"},
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
        timeout=15,
    )
    if resp.status_code == 429:
        raise RuntimeError("Brave rate limit reached")
    resp.raise_for_status()
    return normalize_brave_results(resp.json(), query, max_results=max_results)


def search_cache_path(backend: str, query: str) -> Path:
    key = hashlib.sha256(f"{backend}:{normalized_query_key(query)}".encode("utf-8")).hexdigest()[:24]
    return SEARCH_CACHE_DIR / f"{key}.json"


def get_cached_search_results(backend: str, query: str) -> list[dict[str, Any]] | None:
    ttl_hours = float(os.environ.get("MORNING_BRIEF_SEARCH_CACHE_TTL_HOURS", "6"))
    if ttl_hours <= 0:
        return None
    path = search_cache_path(backend, query)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (time.time() - float(data.get("cached_at", 0))) / 3600 > ttl_hours:
            return None
        return data.get("results", [])
    except Exception:
        return None


def write_search_cache(backend: str, query: str, results: list[dict[str, Any]]) -> None:
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"backend": backend, "query": normalized_query_key(query), "cached_at": time.time(), "results": sanitize(results)}
    search_cache_path(backend, query).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_search(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    global MORNING_BRIEF_API_CALLS
    max_api_calls = int(os.environ.get("MORNING_BRIEF_MAX_API_CALLS", "12"))
    for backend in search_backends():
        cached = get_cached_search_results(backend, query)
        if cached is not None:
            return cached
        if backend == "brave":
            if MORNING_BRIEF_API_CALLS >= max_api_calls:
                continue
            try:
                MORNING_BRIEF_API_CALLS += 1
                results = fetch_brave(query, max_results=max_results)
                if results or os.environ.get("MORNING_BRIEF_SEARCH_CACHE_EMPTY", "0") == "1":
                    write_search_cache(backend, query, results)
                if results or os.environ.get("MORNING_BRIEF_SEARCH_FALLBACK_ON_EMPTY", "1") != "1":
                    return results
            except Exception as exc:
                print(f"[WARN] Brave fetch failed for {query!r}: {exc}", file=sys.stderr)
                if os.environ.get("MORNING_BRIEF_SEARCH_FALLBACK_ON_ERROR", "1") != "1":
                    return []
        elif backend == "searxng":
            results = fetch_searxng(query, max_results=max_results)
            if results or os.environ.get("MORNING_BRIEF_SEARCH_CACHE_EMPTY", "0") == "1":
                write_search_cache(backend, query, results)
            return results
    return []


def extract_article_text(url: str, max_chars: int = 1800) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; VolitionMorningBrief/1.0)"},
        )
        resp.raise_for_status()
        raw = resp.text or ""
    except Exception:
        return ""
    raw = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style\b[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"</(p|h1|h2|h3|li)>", "\n", raw, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:max_chars]


def cluster_key(item: dict[str, Any]) -> str:
    title = re.sub(r"[^a-z0-9 ]+", " ", item.get("title", "").lower())
    words = [w for w in title.split() if len(w) > 3 and w not in {"with", "from", "that", "this", "about", "after"}]
    return " ".join(words[:7]) or item.get("url", "")


def collect_candidates(categories: list[str], fetch_articles: bool = True, min_score: int = 0) -> list[dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for category in categories:
        config = CATEGORIES.get(category)
        if not config:
            print(f"[WARN] Unknown category: {category}", file=sys.stderr)
            continue
        for query in config["queries"]:
            for item in fetch_search(query):
                url = item.get("url", "")
                if not url:
                    continue
                existing = by_url.setdefault(url, {**item, "categories": set(), "score": 0})
                existing["categories"].add(category)
                existing["score"] = max(existing.get("score", 0), score_candidate(item, category))

    clusters: dict[str, dict[str, Any]] = {}
    for item in by_url.values():
        item["categories"] = sorted(item["categories"])
        item["cluster_key"] = cluster_key(item)
        key = item["cluster_key"]
        if key not in clusters or item["score"] > clusters[key]["score"]:
            clusters[key] = item
        else:
            clusters[key].setdefault("duplicates", []).append({"title": item["title"], "url": item["url"]})

    candidates = [item for item in clusters.values() if item.get("score", 0) >= min_score]
    candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    if fetch_articles:
        for item in candidates[: int(os.environ.get("MORNING_BRIEF_FETCH_LIMIT", "10"))]:
            item["article_excerpt"] = strip_thinking(extract_article_text(item["url"]))
    return candidates


def save_raw_model_output(raw_content: str, prefix: str = "morning_brief") -> None:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        path = DEBUG_DIR / f"{prefix}_raw_{int(time.time())}.txt"
        path.write_text(raw_content or "", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to save raw model output: {exc}", file=sys.stderr)


def call_brief_model(prompt: str, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = int(os.environ.get("MORNING_BRIEF_MAX_TOKENS", "3500"))
    headers = {"Content-Type": "application/json"}
    if BRIEF_API_KEY:
        headers["Authorization"] = f"Bearer {BRIEF_API_KEY}"
    payload = {
        "model": BRIEF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are preparing a fast morning-room agenda for chat:watercooler. "
                    "Use the provided candidates only. Separate watch/action items from non-urgent discussion. "
                    "Suggest fleet prompts and Arthur/Human-Abe candidates. Do not post to chat:synchronous. "
                    "Do not include hidden reasoning or <think> blocks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": float(os.environ.get("MORNING_BRIEF_TEMPERATURE", "0.4")),
        "top_p": float(os.environ.get("MORNING_BRIEF_TOP_P", "0.95")),
    }
    try:
        resp = requests.post(
            f"{BRIEF_API_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=int(os.environ.get("BRIEF_MODEL_TIMEOUT", "300")),
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"].get("content", "")
        save_raw_model_output(raw, prefix="morning_brief_editor")
        return strip_thinking(raw)
    except Exception as exc:
        print(f"[WARN] Brief model call failed: {exc}", file=sys.stderr)
        return ""


def fallback_agenda(candidates: list[dict[str, Any]]) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    sections = {
        "Action / Watch Items": [],
        "AI / LLM / Runtime": [],
        "Tech / Homelab / Self-hosting": [],
        "World / Politics Context": [],
    }
    for item in candidates[:18]:
        cats = set(item.get("categories", []))
        line = f"- {item.get('title', 'Untitled')} ({item.get('url')})"
        if "security" in cats:
            sections["Action / Watch Items"].append(line)
        if "ai" in cats:
            sections["AI / LLM / Runtime"].append(line)
        if cats & {"tech", "infra"}:
            sections["Tech / Homelab / Self-hosting"].append(line)
        if "world" in cats:
            sections["World / Politics Context"].append(line)

    parts = [f"# Morning Room - {date}", ""]
    for heading, lines in sections.items():
        parts.append(f"## {heading}")
        parts.extend(lines[:5] or ["- No high-confidence item found in this pass."])
        parts.append("")
    parts.append("## Suggested Fleet Discussion")
    parts.append("- @abe-06: scan the watch items and decide whether anything belongs in Human-Abe's view.")
    parts.append("- @abe-04: security/watch items are available for non-urgent review if relevant.")
    parts.append("")
    parts.append("## Arthur / Human-Abe Candidates")
    human_candidates = [
        item
        for item in candidates
        if item.get("score", 0) >= int(os.environ.get("MORNING_BRIEF_HUMAN_SCORE", "40"))
        and source_quality(item.get("url", "")) >= int(os.environ.get("MORNING_BRIEF_HUMAN_SOURCE_SCORE", "20"))
        and is_specific_candidate(item)
    ]
    if human_candidates:
        for item in human_candidates[:3]:
            parts.append(f"- Review before surfacing: {item.get('title', 'Untitled')} ({item.get('url')})")
    else:
        parts.append("- No high-confidence Human-Abe candidate in this pass.")
    return "\n".join(parts)


def agenda_is_sparse(markdown: str) -> bool:
    if len(markdown.strip()) < 500:
        return True
    bullet_count = len(re.findall(r"(?m)^\s*- ", markdown))
    return bullet_count < 4


def generate_agenda(categories: list[str], output_json: bool = False, min_score: int = 0) -> dict[str, Any]:
    candidates = collect_candidates(categories, min_score=min_score)
    compact_candidates = [
        {
            "title": c.get("title"),
            "url": c.get("url"),
            "snippet": c.get("snippet"),
            "article_excerpt": c.get("article_excerpt"),
            "categories": c.get("categories"),
            "score": c.get("score"),
            "published": c.get("published"),
        }
        for c in candidates[:18]
    ]
    date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Create a structured morning-room agenda for chat:watercooler.

Date: {date}
Categories requested: {', '.join(categories)}
Candidates:
{json.dumps(compact_candidates, indent=2, ensure_ascii=False)}

Return markdown with exactly this shape:
# Morning Room - {date}
## Action / Watch Items
## AI / LLM / Runtime
## Tech / Homelab / Self-hosting
## World / Politics Context
## Suggested Fleet Discussion
## Arthur / Human-Abe Candidates

Use concise bullets with URLs. Keep it non-urgent unless there is a real watch item. Include suggested fleet prompts such as @abe-04 or @abe-06 only when useful."""
    markdown = call_brief_model(prompt)
    if not markdown or agenda_is_sparse(markdown):
        markdown = fallback_agenda(candidates)
    markdown = strip_thinking(markdown)
    return sanitize(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "channel": "chat:watercooler",
            "model": BRIEF_MODEL,
            "categories": categories,
            "min_score": min_score,
            "candidates": compact_candidates,
            "markdown": markdown,
        }
    )


def has_postable_items(result: dict[str, Any]) -> bool:
    return bool(result.get("candidates"))


def save_outputs(result: dict[str, Any], *, save_markdown: bool = False, save_json: bool = False, output_dir: Path | None = None) -> dict[str, str]:
    if not save_markdown and not save_json:
        return {}
    target = output_dir or DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, str] = {}
    if save_markdown:
        paths["markdown"] = str(target / f"morning_room_{stamp}.md")
    if save_json:
        paths["json"] = str(target / f"morning_room_{stamp}.json")
    result["saved_paths"] = paths
    if save_markdown:
        Path(paths["markdown"]).write_text(result.get("markdown", ""), encoding="utf-8")
    if save_json:
        Path(paths["json"]).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def post_watercooler(markdown: str) -> None:
    """Opt-in Redis post to chat:watercooler only."""
    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD") or os.environ.get("REDISCLI_AUTH")
    cmd = [
        "redis-cli",
        "-h",
        host,
        "-p",
        port,
        "XADD",
        "chat:watercooler",
        "*",
        "from",
        os.environ.get("MORNING_BRIEF_FROM", "MorningBrief"),
        "content",
        markdown,
        "timestamp",
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "type",
        "morning_brief",
    ]
    env = os.environ.copy()
    if password:
        env["REDISCLI_AUTH"] = password
    try:
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("[MorningBrief] Posted morning brief to chat:watercooler", file=sys.stderr)
    except Exception as exc:
        print(f"[WARN] Failed to post to chat:watercooler: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Morning brief watercooler agenda generator")
    parser.add_argument("--categories", "-c", default="tech,security,ai,world,infra", help="Comma-separated categories")
    parser.add_argument("--output", "-o", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--post-watercooler", action="store_true", help="Opt-in post to chat:watercooler. Never posts to chat:synchronous.")
    parser.add_argument("--min-score", type=int, default=int(os.environ.get("MORNING_BRIEF_MIN_SCORE", "20")), help="Drop candidates below this score before synthesis")
    parser.add_argument("--no-post-if-empty", action="store_true", help="When posting is requested, skip Redis post if no candidates survive filtering")
    parser.add_argument("--no-fetch-articles", action="store_true", help="Reserved compatibility flag; article fetch limit can also be set with MORNING_BRIEF_FETCH_LIMIT=0")
    parser.add_argument("--save-markdown", action="store_true", help="Save markdown artifact to output directory")
    parser.add_argument("--save-json", action="store_true", help="Save JSON artifact to output directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved morning brief artifacts")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.no_fetch_articles:
        os.environ["MORNING_BRIEF_FETCH_LIMIT"] = "0"
    result = generate_agenda(categories, output_json=args.output == "json", min_score=args.min_score)
    saved_paths = save_outputs(result, save_markdown=args.save_markdown, save_json=args.save_json, output_dir=Path(args.output_dir).expanduser())
    if saved_paths:
        result["saved_paths"] = saved_paths
    if args.output == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["markdown"])
        for kind, path in saved_paths.items():
            print(f"[MorningBrief] Saved {kind}: {path}", file=sys.stderr)
    if args.post_watercooler:
        if args.no_post_if_empty and not has_postable_items(result):
            print("[MorningBrief] Skipping chat:watercooler post because no candidates survived filtering.", file=sys.stderr)
        else:
            post_watercooler(result["markdown"])


if __name__ == "__main__":
    main()
