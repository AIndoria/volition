import importlib.util
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spider = load_module("spider_script", ROOT / "spider.py")
morning_brief = load_module("morning_brief_script", ROOT / "morning_brief.py")


class SpiderUtilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        spider.SESSION_DIR = root
        spider.CACHE_DIR = root / ".cache"
        spider.DOC_DIR = root / ".docs"
        morning_brief.SEARCH_CACHE_DIR = root / ".morning_brief_search_cache"
        morning_brief.MORNING_BRIEF_API_CALLS = 0
        spider.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        spider.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        spider.DOC_DIR.mkdir(parents=True, exist_ok=True)
        morning_brief.SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        spider.SEARCH_BACKEND_UNAVAILABLE = False

    def test_strip_thinking_variants(self):
        self.assertEqual(spider.strip_thinking("<think>secret</think>final"), "final")
        self.assertEqual(spider.strip_thinking("</think>final"), "final")
        self.assertEqual(spider.strip_thinking("visible <think>secret"), "visible")
        self.assertEqual(spider.strip_thinking("```thinking\nsecret\n```\nfinal"), "final")
        self.assertNotIn("<think", spider.strip_thinking("<think>secret"))

    def test_extract_json_value_variants(self):
        self.assertEqual(spider.extract_json_value('{"a": 1}'), {"a": 1})
        self.assertEqual(spider.extract_json_value('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(spider.extract_json_value('Here:\n{"a": 1}\nThanks'), {"a": 1})
        self.assertEqual(spider.extract_json_value('["x", "y"]'), ["x", "y"])
        self.assertEqual(spider.extract_json_value('"{\\"a\\": 1}"'), {"a": 1})

    def test_url_normalization(self):
        hf = spider.normalize_source_variants("https://huggingface.co/Nanbeige/Nanbeige4.1-3B")
        self.assertIn("https://huggingface.co/Nanbeige/Nanbeige4.1-3B/raw/main/README.md", hf)
        self.assertIn("https://huggingface.co/api/models/Nanbeige/Nanbeige4.1-3B", hf)

        arxiv = spider.normalize_source_variants("https://arxiv.org/abs/2602.13367")
        self.assertIn("https://export.arxiv.org/api/query?id_list=2602.13367", arxiv)
        self.assertIn("https://arxiv.org/pdf/2602.13367", arxiv)

        github = spider.normalize_source_variants("https://github.com/owner/repo/blob/main/README.md")
        self.assertIn("https://raw.githubusercontent.com/owner/repo/main/README.md", github)

        issue = spider.normalize_source_variants("https://github.com/owner/repo/issues/123")
        self.assertIn("https://api.github.com/repos/owner/repo/issues/123", issue)
        self.assertNotIn("https://raw.githubusercontent.com/owner/repo/main/README.md", issue)
        discussion = spider.normalize_source_variants("https://github.com/owner/repo/discussions/123")
        self.assertNotIn("https://raw.githubusercontent.com/owner/repo/main/README.md", discussion)
        self.assertEqual(
            spider.canonical_url_key("https://docs.python.org/3.14/howto/free-threading-extensions.html"),
            spider.canonical_url_key("https://docs.python.org/3/howto/free-threading-extensions.html"),
        )

    def test_source_identity_keys_only_merge_proven_equivalences(self):
        github_blob = spider.source_identity_keys("https://github.com/owner/repo/blob/main/README.md")
        github_raw = spider.source_identity_keys("https://raw.githubusercontent.com/owner/repo/main/README.md")
        self.assertTrue(github_blob & github_raw)

        arxiv_abs = spider.source_identity_keys("https://arxiv.org/abs/2602.13367")
        arxiv_pdf = spider.source_identity_keys("https://arxiv.org/pdf/2602.13367")
        self.assertTrue(arxiv_abs & arxiv_pdf)

        python_313 = spider.source_identity_keys("https://docs.python.org/3.13/howto/free-threading-extensions.html")
        python_314 = spider.source_identity_keys("https://docs.python.org/3.14/howto/free-threading-extensions.html")
        self.assertFalse(python_313 & python_314)

        repo_root = spider.source_identity_keys("https://github.com/owner/repo")
        releases = spider.source_identity_keys("https://github.com/owner/repo/releases")
        self.assertFalse(repo_root & releases)

    def test_configured_paths_expand_home_without_creating_directories(self):
        self.assertEqual(spider.expand_path("~/spider_sessions"), Path.home() / "spider_sessions")
        self.assertEqual(morning_brief.expand_path("~/morning_briefs"), Path.home() / "morning_briefs")

    def test_source_scoring(self):
        raw_readme = "# Model Card\n\nThis model card describes reasoning, alignment, benchmark evaluation, license, citation, and code generation capabilities."
        frontend_js = "window.__APP__ = {}; document.documentElement; webpack chunk localStorage cookie.match javascript frontend"
        arxiv_text = "Abstract This arxiv paper presents a technical report with evaluation benchmark reasoning and tool use."
        self.assertGreater(spider.score_source_text(raw_readme, "https://huggingface.co/x/y/raw/main/README.md"), spider.score_source_text(frontend_js, "https://huggingface.co/x/y"))
        self.assertGreater(spider.score_source_text(arxiv_text, "https://export.arxiv.org/api/query?id_list=1234.5678"), 25)
        self.assertLess(spider.score_source_text(frontend_js, "https://huggingface.co/x/y"), 0)

    def test_extract_links_from_html_scores_and_enqueues_subpages(self):
        session = spider.SpiderSession(question="Python free-threading extension migration testing", max_sources=5, max_reads=3)
        html = """
        <a href="/porting/">Porting Python Packages to Support Free-Threading</a>
        <a href="/testing/">Testing, Debugging, and Profiling</a>
        <a href="https://other.example.test/nope">External</a>
        """
        links = spider.extract_links_from_html(html, "https://py-free-threading.github.io/")
        self.assertEqual(len(links), 2)
        self.assertLess(spider.score_link_candidate({"url": "https://docs.python.org/3/howto/remote_debugging.html", "text": "Remote debugging attachment protocol"}, session.question), 8)
        source = {"links": links}
        added = spider.enqueue_source_links(session, "S1", source)
        self.assertEqual(added, 2)
        self.assertEqual(session.frontier[0]["discovery"], "source_link")
        self.assertIn("testing", {item["url"].rstrip("/").split("/")[-1] for item in session.frontier})

    def test_seed_ingestion_creates_source_id_claim_and_text_ref(self):
        session = spider.SpiderSession(question="What does source say?", max_sources=5, max_reads=3)
        source_text = "Full source text " + ("important evidence " * 200)
        fake_source = {
            "requested_url": "https://example.test/source",
            "url": "https://example.test/source",
            "final_url": "https://example.test/source",
            "adapter": "web",
            "text": source_text,
            "score": 50,
            "extracted_chars": len(source_text),
            "fetched_at": spider.now_iso(),
        }
        model_json = json.dumps({"summary": "Source says important evidence.", "claims": [{"claim": "Important evidence exists.", "confidence": "high"}], "contradictions": []})
        with patch.object(spider, "fetch_webpage", return_value=fake_source), patch.object(spider, "call_nanbeige", return_value=model_json):
            with contextlib.redirect_stdout(io.StringIO()):
                spider.ingest_seed_urls(session, ["https://example.test/source"])

        self.assertEqual(session.sources[0]["source_id"], "S1")
        self.assertTrue(Path(session.sources[0]["text_ref"]).exists())
        self.assertIn("important evidence", spider.get_source_text(session, "S1"))
        self.assertEqual(session.claims[0]["claim"], "Important evidence exists.")

    def test_search_action_adds_frontier_and_pop_reads_highest_score_first(self):
        session = spider.SpiderSession(question="frontier test", max_sources=5, max_reads=3)
        results = [
            {"title": "Low", "url": "https://example.test/low", "snippet": "low", "score": 3},
            {"title": "High", "url": "https://example.test/high", "snippet": "high", "score": 99},
        ]
        with patch.object(spider, "fetch_search", return_value=(results, {"backend": "searxng", "cache_hit": False, "attempts": []})):
            observation = spider.execute_action(session, {"tool": "search", "query": "frontier"})

        self.assertEqual(observation["result_count"], 2)
        action = spider.pop_frontier_action(session)
        self.assertEqual(action["url"], "https://example.test/high")

    def test_duplicate_read_url_is_skipped_without_fetch_or_read_increment(self):
        session = spider.SpiderSession(question="duplicate", max_sources=5, max_reads=3)
        sid = session.add_source(
            {
                "requested_url": "https://py-free-threading.github.io/",
                "url": "https://py-free-threading.github.io/",
                "final_url": "https://py-free-threading.github.io/",
                "adapter": "web",
                "text": "already read",
                "score": 30,
                "relevance": "relevant",
            },
            seed=False,
        )
        session.read_count = 1
        with patch.object(spider, "fetch_webpage") as fetch, patch.object(spider, "call_nanbeige") as model:
            observation = spider.execute_action(session, {"tool": "read_url", "url": "https://py-free-threading.github.io"})
        fetch.assert_not_called()
        model.assert_not_called()
        self.assertEqual(observation["status"], "already_read")
        self.assertEqual(observation["source_id"], sid)
        self.assertEqual(session.read_count, 1)

    def test_source_and_frontier_deduplicate_equivalent_urls(self):
        session = spider.SpiderSession(question="duplicate variants", max_sources=5, max_reads=3)
        blob_url = "https://github.com/owner/repo/blob/main/README.md"
        raw_url = "https://raw.githubusercontent.com/owner/repo/main/README.md"
        source_id = session.add_source(
            {
                "requested_url": blob_url,
                "url": blob_url,
                "final_url": blob_url,
                "adapter": "web",
                "text": "README evidence",
                "score": 30,
            }
        )
        self.assertEqual(
            session.add_source(
                {
                    "requested_url": raw_url,
                    "url": raw_url,
                    "final_url": raw_url,
                    "adapter": "web",
                    "text": "README evidence",
                    "score": 30,
                }
            ),
            source_id,
        )
        self.assertEqual(len(session.sources), 1)

        first = {"tool": "read_url", "url": "https://example.test/article", "title": "Article", "snippet": "evidence", "search_score": 80}
        equivalent = {"tool": "read_url", "url": "https://example.test/article/", "title": "Article", "snippet": "evidence", "search_score": 80}
        self.assertTrue(spider.add_frontier_candidate(session, first))
        self.assertFalse(spider.add_frontier_candidate(session, equivalent))
        self.assertEqual(len(session.frontier), 1)

    def test_search_frontier_skips_unreadable_model_artifacts(self):
        session = spider.SpiderSession(question="artifact filter", max_sources=5, max_reads=3)
        results = [
            {"title": "Weights", "url": "https://huggingface.co/x/y/blob/main/model.safetensors", "snippet": "weights", "score": 100},
            {"title": "Paper", "url": "https://arxiv.org/abs/2602.13367", "snippet": "paper", "score": 50},
        ]
        with patch.object(spider, "fetch_search", return_value=(results, {"backend": "searxng", "cache_hit": False, "attempts": []})):
            spider.execute_action(session, {"tool": "search", "query": "nanbeige"})
        self.assertEqual([item["url"] for item in session.frontier], ["https://arxiv.org/abs/2602.13367"])

    def test_standard_frontier_skips_low_score_and_blocklisted_domains(self):
        session = spider.SpiderSession(question="quality filter", depth="standard", max_sources=5, max_reads=3)
        results = [
            {"title": "SEO", "url": "https://opc.csdn.net/article", "snippet": "seo", "score": 50},
            {"title": "Weak", "url": "https://example.test/weak", "snippet": "weak", "score": 10},
            {"title": "Strong", "url": "https://arxiv.org/abs/2602.13367", "snippet": "paper", "score": 50},
        ]
        with patch.object(spider, "fetch_search", return_value=(results, {"backend": "searxng", "cache_hit": False, "attempts": []})):
            spider.execute_action(session, {"tool": "search", "query": "nanbeige"})
        self.assertEqual([item["url"] for item in session.frontier], ["https://arxiv.org/abs/2602.13367"])

    def test_free_threading_sources_score_above_standard_threshold(self):
        row = {
            "title": "C API Extension Support for Free Threading",
            "url": "https://docs.python.org/3/howto/free-threading-extensions.html",
            "content": "PEP 703 free-threaded CPython extension module Py_mod_gil C API guide",
        }
        self.assertGreaterEqual(spider.score_search_result(row, "free-threaded CPython extension modules Py_mod_gil"), 20)

    def test_invalid_controller_fallback_prefers_frontier(self):
        session = spider.SpiderSession(question="fallback", max_sources=5, max_reads=3)
        session.frontier.append({"tool": "read_url", "url": "https://example.test/read", "search_score": 30})
        with patch.object(spider, "call_controller_model", return_value="not json"):
            controller = spider.ask_controller(session)
        self.assertEqual(controller["action"]["tool"], "read_url")
        self.assertEqual(controller["action"]["url"], "https://example.test/read")

    def test_empty_searches_advance_to_untried_queries(self):
        session = spider.SpiderSession(question="Nanbeige evidence?", max_sources=5, max_reads=3)
        first = spider.next_search_action(session)
        with patch.object(spider, "fetch_search", return_value=([], {"backend": "searxng", "cache_hit": False, "attempts": []})):
            spider.execute_action(session, first)
        second = spider.next_search_action(session)
        self.assertIsNotNone(second)
        self.assertNotEqual(first["query"], second["query"])

    def test_python_free_threading_queries_are_targeted(self):
        session = spider.SpiderSession(question="For Python extension authors, what are migration requirements of CPython free-threading from PEP 703?", max_sources=5, max_reads=3)
        queries = spider.search_query_candidates(session)
        self.assertIn("CPython free-threading C extension HOWTO", queries)
        self.assertIn("free-threaded CPython extension modules Py_mod_gil", queries)

    def test_local_llm_serving_queries_are_targeted(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=5, max_reads=3)
        queries = spider.search_query_candidates(session)
        self.assertIn("llama.cpp server documentation OpenAI compatible API", queries)
        self.assertIn("Ollama documentation API model serving", queries)
        self.assertIn("vLLM documentation serving OpenAI compatible server", queries)

    def test_long_memory_agent_prompt_generates_concise_queries(self):
        long_question = (
            "Research how different projects do Brains and Dreaming according to OpenAI memories and handle "
            "memories RAG embeddings for agents. Include labs, harnesses, papers, homelab feasibility. "
            + ("This sentence makes the prompt very long. " * 80)
            + " https://github.com/aindoria/volition"
        )
        session = spider.SpiderSession(question=long_question, max_sources=5, max_reads=3)
        queries = spider.search_query_candidates(session)
        self.assertIn("LLM agent memory architecture RAG embeddings long term memory", queries)
        self.assertIn("agent memory systems MemGPT Letta Zep LangGraph LangChain", queries)
        self.assertTrue(all(len(spider.normalize_search_query(query)) <= 220 for query in queries))

    def test_memory_research_does_not_queue_ten_openai_sibling_docs(self):
        session = spider.SpiderSession(question="Research agent memory RAG embeddings and memory architectures", depth="deep", max_sources=50, max_reads=40)
        for i in range(3):
            session.add_source(
                {
                    "requested_url": f"https://developers.openai.com/api/docs/guides/agents/page-{i}",
                    "url": f"https://developers.openai.com/api/docs/guides/agents/page-{i}",
                    "final_url": f"https://developers.openai.com/api/docs/guides/agents/page-{i}",
                    "adapter": "web",
                    "text": "OpenAI agents docs.",
                    "score": 40,
                    "relevance": "partial",
                },
                seed=False,
            )
        for i in range(10):
            spider.add_frontier_candidate(
                session,
                {
                    "tool": "read_url",
                    "url": f"https://developers.openai.com/api/docs/guides/agents/sibling-{i}",
                    "title": "OpenAI Agents generic tools docs",
                    "snippet": "generic tools docs",
                    "search_score": 80,
                },
            )
        self.assertFalse(any("sibling-" in item.get("url", "") for item in session.frontier))
        self.assertTrue(any(item.get("frontier_state") == "skipped_source_family_saturated" for item in session.frontier_audit))

    def test_migration_docs_score_high_for_migration_plan(self):
        session = spider.SpiderSession(question="Research how to migrate an OpenAI Responses integration", depth="deep", max_sources=10, max_reads=10)
        item = {"url": "https://developers.openai.com/api/docs/guides/migrate-to-responses", "title": "Migrate to Responses", "snippet": "migration guide", "search_score": 20}
        scored = spider.policy_score_item(session, item)
        self.assertEqual(scored["frontier_state"], "read_now")
        self.assertGreaterEqual(scored["policy_score"], spider.min_frontier_score(session))

    def test_migration_docs_reserve_for_memory_architecture_plan(self):
        session = spider.SpiderSession(question="Research agent memory RAG embeddings and memory architectures", depth="deep", max_sources=10, max_reads=10)
        item = {"url": "https://developers.openai.com/api/docs/guides/migrate-to-responses", "title": "Migrate to Responses", "snippet": "migration guide", "search_score": 50}
        scored = spider.policy_score_item(session, item)
        self.assertIn(scored["frontier_state"], {"reserve", "skipped_low_policy_score"})
        self.assertIn("migration", scored["source_policy_matches"]["low_value_penalized"])

    def test_sandbox_docs_contextual_scoring(self):
        sandbox = {"url": "https://developers.openai.com/api/docs/guides/agents/sandboxes", "title": "Agent Sandboxes", "snippet": "sandbox runtime isolation", "search_score": 20}
        runtime_session = spider.SpiderSession(question="Research agent sandbox runtime isolation design", depth="deep", max_sources=10, max_reads=10)
        memory_session = spider.SpiderSession(question="Research agent memory RAG embeddings architecture", depth="deep", max_sources=10, max_reads=10)
        self.assertEqual(spider.policy_score_item(runtime_session, sandbox)["frontier_state"], "read_now")
        self.assertIn(spider.policy_score_item(memory_session, sandbox)["frontier_state"], {"reserve", "skipped_low_policy_score"})

    def test_same_path_family_cap_works(self):
        session = spider.SpiderSession(question="Research agent memory systems", depth="deep", max_sources=20, max_reads=20)
        for i in range(3):
            session.add_source(
                {
                    "requested_url": f"https://developers.openai.com/api/docs/guides/agents/doc-{i}",
                    "url": f"https://developers.openai.com/api/docs/guides/agents/doc-{i}",
                    "final_url": f"https://developers.openai.com/api/docs/guides/agents/doc-{i}",
                    "adapter": "web",
                    "text": "agent docs",
                    "score": 40,
                    "relevance": "partial",
                },
                seed=False,
            )
        candidate = {"url": "https://developers.openai.com/api/docs/guides/agents/doc-4", "title": "More agent docs", "snippet": "memory", "search_score": 90}
        self.assertEqual(spider.policy_score_item(session, candidate)["frontier_state"], "skipped_source_family_saturated")

    def test_github_repo_families_are_separate(self):
        session = spider.SpiderSession(question="Research agent memory projects", depth="deep", max_sources=20, max_reads=20)
        for i in range(3):
            session.add_source(
                {
                    "requested_url": f"https://github.com/mem0ai/mem0/blob/main/doc-{i}.md",
                    "url": f"https://github.com/mem0ai/mem0/blob/main/doc-{i}.md",
                    "final_url": f"https://github.com/mem0ai/mem0/blob/main/doc-{i}.md",
                    "adapter": "web",
                    "text": "mem0 memory project",
                    "score": 40,
                    "relevance": "partial",
                },
                seed=False,
            )
        mem0 = {"url": "https://github.com/mem0ai/mem0/blob/main/README.md", "title": "mem0 README", "snippet": "memory", "search_score": 80}
        langgraph = {"url": "https://github.com/langchain-ai/langgraph/blob/main/README.md", "title": "LangGraph README", "snippet": "memory", "search_score": 80}
        self.assertEqual(spider.policy_score_item(session, mem0)["frontier_state"], "skipped_source_family_saturated")
        self.assertNotEqual(spider.policy_score_item(session, langgraph)["frontier_state"], "skipped_source_family_saturated")

    def test_arxiv_paper_families_are_separate(self):
        session = spider.SpiderSession(question="Research agent memory papers", depth="deep", max_sources=20, max_reads=20)
        for i in range(3):
            session.add_source(
                {
                    "requested_url": f"https://arxiv.org/html/2601.00001v{i}",
                    "url": f"https://arxiv.org/html/2601.00001v{i}",
                    "final_url": f"https://arxiv.org/abs/2601.00001",
                    "adapter": "arxiv",
                    "text": "paper about memory",
                    "score": 40,
                    "relevance": "partial",
                },
                seed=False,
            )
        same_paper = {"url": "https://arxiv.org/html/2601.00001v4", "title": "Same paper", "snippet": "memory", "search_score": 80}
        other_paper = {"url": "https://arxiv.org/pdf/2601.00002", "title": "Other paper", "snippet": "memory", "search_score": 80}
        self.assertEqual(spider.policy_score_item(session, same_paper)["frontier_state"], "skipped_source_family_saturated")
        self.assertNotEqual(spider.policy_score_item(session, other_paper)["frontier_state"], "skipped_source_family_saturated")

    def test_frontier_ranking_prefers_uncovered_must_cover_target(self):
        session = spider.SpiderSession(question="Research agent memory projects", depth="deep", max_sources=20, max_reads=20)
        session.research_plan = {
            "source_policy": {
                "must_cover_targets": ["zep"],
                "high_value_terms": ["memory", "zep"],
                "low_value_unless_plan_relevant": [],
            }
        }
        session.frontier = [
            {"tool": "read_url", "url": "https://developers.openai.com/api/docs/guides/agents/tools", "title": "OpenAI tools", "snippet": "memory", "search_score": 90},
            {"tool": "read_url", "url": "https://github.com/getzep/zep", "title": "Zep memory", "snippet": "agent memory", "search_score": 35},
        ]
        action = spider.pop_frontier_action(session)
        self.assertEqual(action["url"], "https://github.com/getzep/zep")

    def test_reserved_and_skipped_frontier_entries_appear_in_trace_and_status(self):
        session = spider.SpiderSession(question="Research agent memory RAG embeddings architecture", depth="deep", max_sources=20, max_reads=20)
        spider.add_frontier_candidate(
            session,
            {"tool": "read_url", "url": "https://developers.openai.com/api/docs/guides/migrate-to-responses", "title": "Migrate to Responses", "snippet": "migration guide", "search_score": 50},
        )
        spider.add_frontier_candidate(
            session,
            {"tool": "read_url", "url": "https://example.test/weak", "title": "Weak", "snippet": "unrelated", "search_score": 1},
        )
        status_path = spider.write_status_artifact(session, current_message="testing status")
        status = json.loads(status_path.read_text())
        self.assertGreaterEqual(status["frontier_counts"].get("reserve", 0), 1)
        self.assertGreaterEqual(status["frontier_counts"].get("skipped_low_policy_score", 0), 1)
        self.assertTrue(any(entry["type"] == "frontier_state" for entry in session.trace))

    def test_status_json_updates_after_search_action(self):
        session = spider.SpiderSession(question="frontier status test", depth="deep", max_sources=5, max_reads=3)
        results = [{"title": "Memory paper", "url": "https://arxiv.org/abs/2601.00002", "snippet": "agent memory paper", "score": 70}]
        with patch.object(spider, "fetch_search", return_value=(results, {"backend": "brave", "cache_hit": False, "attempts": []})):
            action = {"tool": "search", "query": "agent memory paper"}
            spider.execute_action(session, action)
            spider.write_status_artifact(session, current_action=action, current_message="Completed search")
        status = json.loads((spider.SESSION_DIR / "status.json").read_text())
        self.assertEqual(status["session_id"], session.session_id)
        self.assertEqual(status["current_action"]["tool"], "search")
        self.assertEqual(status["counts"]["searches"], 1)

    def test_stop_decision_trace_and_status_are_written(self):
        session = spider.SpiderSession(question="stop test", depth="quick", max_steps=1, max_sources=5, max_reads=3)
        session.current_step = 1
        session.record_stop_decision("max_steps", {"max_steps": 1})
        status_path = spider.write_status_artifact(session)
        status = json.loads(status_path.read_text())
        self.assertEqual(session.trace[-1]["type"], "stop_decision")
        self.assertEqual(status["stop_reason"], "max_steps")

    def test_source_card_structurer_failure_falls_back_to_partial_not_failed(self):
        session = spider.SpiderSession(question="memory", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "memory evidence", "score": 20}, seed=False)
        card = "SOURCE_RELEVANCE: relevant\nSUMMARY: Memory evidence.\nKEY_CLAIMS:\n- Memory exists.\nEVIDENCE_SNIPPETS:\n- memory evidence\nCONTRADICTIONS: none\nFOLLOW_UP_LINKS: none"
        with patch.object(spider, "call_nanbeige", return_value=card), patch.object(spider, "call_structurer_model", return_value="not json"):
            spider.summarize_source(session, sid, "memory evidence")
        source = spider.get_source(session, sid)
        self.assertEqual(source["relevance"], "partial")
        self.assertEqual(source["structuring_status"], "fallback_markdown_card")
        self.assertNotEqual(source["relevance"], "summarization_failed")

    def test_regenerate_report_reuses_completed_session_without_search(self):
        session = spider.SpiderSession(question="regen", max_sources=5, max_reads=3)
        session.status = "completed"
        session.report_style = "standard"
        with patch.object(spider, "generate_report", return_value="# regenerated") as generate, patch.object(spider, "fetch_search") as search:
            report = spider.regenerate_report(session)
        generate.assert_called_once()
        search.assert_not_called()
        self.assertEqual(report, "# regenerated")
        self.assertTrue((spider.SESSION_DIR / f"{session.session_id}_report.md").exists())

    def test_longform_pipeline_writes_separate_appendix_and_qa(self):
        session = spider.SpiderSession(question="longform memory", depth="deep", max_sources=5, max_reads=3)
        session.report_style = "longform"
        sid = session.add_source(
            {
                "requested_url": "https://example.test/memory",
                "url": "https://example.test/memory",
                "final_url": "https://example.test/memory",
                "adapter": "web",
                "text": "Memory architecture evidence.",
                "score": 40,
                "relevance": "relevant",
                "source_card": "SOURCE_RELEVANCE: relevant\nSUMMARY: Memory architecture.",
            },
            seed=False,
        )
        session.add_claim("Memory architecture exists.", [sid], "high", evidence="Memory architecture evidence")
        long_section = " ".join(["prose"] * 700)

        def fake_writer(prompt, *args, **kwargs):
            if "thesis-driven outline" in prompt:
                return "# Outline\n\n" + "\n".join(f"## {s}" for s in spider.LONGFORM_SECTIONS)
            return long_section

        with patch.dict(os.environ, {"SPIDER_LONGFORM_TARGET_WORDS": "1000"}), patch.object(spider, "verify_claim_ledger"), patch.object(spider, "call_writer_model", side_effect=fake_writer):
            report = spider.generate_longform_report(session)
        self.assertNotIn("# Source Appendix", report)
        self.assertTrue((spider.SESSION_DIR / "source_appendix.md").exists())
        qa = json.loads((spider.SESSION_DIR / "longform_qa.json").read_text())
        self.assertGreaterEqual(qa["word_count"], 750)

    def test_longform_qa_rejects_compact_summary_evidence_table(self):
        session = spider.SpiderSession(question="qa", depth="deep", max_sources=5, max_reads=3)
        session.report_style = "longform"
        report = "# Spider\n\n## Executive Summary\nShort.\n\n## Evidence Table\n| A | B |\n| - | - |\n"
        qa = spider.longform_report_qa(report, "# Source Appendix\n", session, target_words=8000)
        self.assertFalse(qa["pass"])
        self.assertIn("looks_like_compact_summary_plus_evidence_table", qa["issues"])

    def test_fetch_search_normalizes_backend_query_before_brave(self):
        session = spider.SpiderSession(question="search", max_sources=5, max_reads=3)
        seen = {}

        def fake_brave(query, max_results=6):
            seen["query"] = query
            return [{"title": "Good", "url": "https://example.test/good", "snippet": "agent memory", "score": 50, "backend": "brave"}]

        long_query = "Research agent memory RAG embeddings " + ("extra words " * 100)
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "SPIDER_SEARCH_BACKENDS": "brave", "SPIDER_SEARCH_CACHE_TTL_HOURS": "0"}):
            with patch.object(spider, "fetch_brave", side_effect=fake_brave):
                results, meta = spider.fetch_search(session, long_query, max_results=5)
        self.assertTrue(results)
        self.assertLessEqual(len(seen["query"]), 220)
        self.assertEqual(meta["original_query"], long_query)

    def test_question_embedded_urls_are_ingested_as_seed_urls(self):
        session = spider.SpiderSession(question="Research this https://example.test/source and compare memory systems.", max_sources=5, max_reads=3)
        fake_source = {
            "requested_url": "https://example.test/source",
            "url": "https://example.test/source",
            "final_url": "https://example.test/source",
            "adapter": "web",
            "text": "Embedded source evidence benchmark reasoning alignment.",
            "score": 40,
            "extracted_chars": 52,
            "fetched_at": spider.now_iso(),
        }
        model_json = json.dumps({"summary": "Embedded source.", "claims": [{"claim": "Embedded source was read.", "confidence": "high", "evidence": "Embedded source evidence"}], "contradictions": []})
        with patch.object(spider, "fetch_webpage", return_value=fake_source), patch.object(spider, "call_nanbeige", return_value=model_json):
            spider.ingest_seed_urls(session, [])
        self.assertEqual(session.sources[0]["url"], "https://example.test/source")
        self.assertEqual(session.claims[0]["claim"], "Embedded source was read.")

    def test_comparison_gap_search_prioritizes_missing_targets(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=5, max_reads=3)
        session.search_history = [{"query": "llama.cpp server documentation OpenAI compatible API"}]
        session.sources = [{"source_id": "S1", "title": "llama.cpp server", "url": "https://github.com/ggml-org/llama.cpp", "final_url": "https://github.com/ggml-org/llama.cpp", "relevance": "relevant"}]
        action = spider.next_gap_search_action(session)
        self.assertEqual(action["query"], "Ollama documentation API model serving")

    def test_comparison_frontier_prefers_missing_target_over_score(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=8, max_reads=5)
        session.sources = [{"source_id": "S1", "title": "llama.cpp server", "url": "https://github.com/ggml-org/llama.cpp", "final_url": "https://github.com/ggml-org/llama.cpp", "relevance": "relevant"}]
        session.frontier = [
            {"tool": "read_url", "url": "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md", "title": "llama.cpp server", "search_score": 80, "target_hint": "llama.cpp"},
            {"tool": "read_url", "url": "https://github.com/ollama/ollama/blob/main/docs/api.md", "title": "Ollama API", "search_score": 35, "target_hint": "ollama"},
            {"tool": "read_url", "url": "https://docs.vllm.ai/en/latest/getting_started/quickstart/", "title": "vLLM Quickstart", "search_score": 30, "target_hint": "vllm"},
        ]
        action = spider.pop_frontier_action(session)
        self.assertEqual(action["target_hint"], "ollama")

    def test_search_action_preserves_coverage_gap_on_frontier(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=8, max_reads=5)
        results = [{"title": "Ollama API", "url": "https://github.com/ollama/ollama/blob/main/docs/api.md", "snippet": "API", "score": 35}]
        with patch.object(spider, "fetch_search", return_value=(results, {"backend": "brave", "cache_hit": False, "attempts": []})):
            spider.execute_action(session, {"tool": "search", "query": "Ollama documentation API model serving", "coverage_gap": "ollama"})
        self.assertEqual(session.frontier[0]["target_hint"], "ollama")

    def test_fetch_searxng_has_no_default_time_range(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        seen = {}

        def fake_get(url, params=None, timeout=None):
            seen.update(params or {})
            return FakeResponse()

        with patch.object(spider.requests, "get", side_effect=fake_get):
            spider.fetch_searxng("stable docs")
        self.assertNotIn("time_range", seen)

    def test_brave_result_normalization(self):
        data = {
            "web": {
                "results": [
                    {
                        "title": "C API Extension Support for Free Threading",
                        "url": "https://docs.python.org/3/howto/free-threading-extensions.html",
                        "description": "Py_mod_gil and free-threaded CPython extension guidance",
                    }
                ]
            }
        }
        result = spider.normalize_brave_results(data, "free-threaded CPython extension modules Py_mod_gil")[0]
        self.assertEqual(result["backend"], "brave")
        self.assertEqual(result["source_type"], "web")
        self.assertGreaterEqual(result["score"], 20)

    def test_brave_primary_prevents_searxng_call_when_results_are_good(self):
        session = spider.SpiderSession(question="search", max_sources=5, max_reads=3)
        brave_results = [{"title": "Good", "url": "https://example.test/good", "snippet": "good", "score": 50, "backend": "brave"}]
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "SPIDER_SEARCH_BACKENDS": "brave,searxng", "SPIDER_SEARCH_CACHE_TTL_HOURS": "0"}):
            with patch.object(spider, "fetch_brave", return_value=brave_results) as brave, patch.object(spider, "fetch_searxng") as searx:
                results, meta = spider.fetch_search(session, "query", max_results=5)
        brave.assert_called_once()
        searx.assert_not_called()
        self.assertEqual(results, brave_results)
        self.assertEqual(meta["backend"], "brave")
        self.assertEqual(session.coverage["search_api_calls"], 1)

    def test_brave_empty_or_error_falls_back_to_searxng(self):
        session = spider.SpiderSession(question="search", max_sources=5, max_reads=3)
        searx_results = [{"title": "Fallback", "url": "https://example.test/fallback", "snippet": "fallback", "score": 30}]
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "SPIDER_SEARCH_BACKENDS": "brave,searxng", "SPIDER_SEARCH_CACHE_TTL_HOURS": "0"}):
            with patch.object(spider, "fetch_brave", return_value=[]), patch.object(spider, "fetch_searxng", return_value=searx_results) as searx:
                results, meta = spider.fetch_search(session, "query", max_results=5)
        searx.assert_called_once()
        self.assertEqual(results, searx_results)
        self.assertEqual(meta["backend"], "searxng")
        self.assertEqual(meta["fallback_reason"], "brave_empty")

    def test_brave_budget_prevents_further_api_calls(self):
        session = spider.SpiderSession(question="search", max_sources=5, max_reads=3)
        session.coverage["search_api_calls"] = 1
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "SPIDER_SEARCH_BACKENDS": "brave,searxng", "SPIDER_SEARCH_MAX_API_CALLS": "1", "SPIDER_SEARCH_CACHE_TTL_HOURS": "0"}):
            with patch.object(spider, "fetch_brave") as brave, patch.object(spider, "fetch_searxng", return_value=[]) as searx:
                spider.fetch_search(session, "query", max_results=5)
        brave.assert_not_called()
        searx.assert_called_once()

    def test_search_cache_hit_prevents_backend_calls(self):
        session = spider.SpiderSession(question="search", max_sources=5, max_reads=3)
        cached = [{"title": "Cached", "url": "https://example.test/cached", "snippet": "cached", "score": 30}]
        spider.write_search_cache("brave", "cached query", cached)
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "SPIDER_SEARCH_BACKENDS": "brave,searxng", "SPIDER_SEARCH_CACHE_TTL_HOURS": "24"}):
            with patch.object(spider, "fetch_brave") as brave, patch.object(spider, "fetch_searxng") as searx:
                results, meta = spider.fetch_search(session, "cached query", max_results=5)
        brave.assert_not_called()
        searx.assert_not_called()
        self.assertEqual(results, cached)
        self.assertTrue(meta["cache_hit"])

    def test_morning_brief_brave_primary_prevents_searxng_call_when_results_are_good(self):
        brave_results = [{"title": "AI release", "url": "https://example.test/ai", "snippet": "model release", "backend": "brave"}]
        with patch.dict(os.environ, {"SPIDER_BRAVE_API_KEY": "test-key", "MORNING_BRIEF_SEARCH_BACKENDS": "brave,searxng", "MORNING_BRIEF_SEARCH_CACHE_TTL_HOURS": "0"}):
            with patch.object(morning_brief, "fetch_brave", return_value=brave_results) as brave, patch.object(morning_brief, "fetch_searxng") as searx:
                results = morning_brief.fetch_search("ai query", max_results=5)
        brave.assert_called_once()
        searx.assert_not_called()
        self.assertEqual(results, brave_results)

    def test_morning_brief_save_outputs_writes_markdown_and_json(self):
        root = Path(self.tmp.name) / "briefs"
        result = {"markdown": "# brief", "candidates": []}
        paths = morning_brief.save_outputs(result, save_markdown=True, save_json=True, output_dir=root)
        self.assertTrue(Path(paths["markdown"]).exists())
        self.assertTrue(Path(paths["json"]).exists())
        saved = json.loads(Path(paths["json"]).read_text())
        self.assertIn("saved_paths", saved)

    def test_morning_brief_fallback_human_candidates_are_score_gated(self):
        candidates = [
            {"title": "Weak blog", "url": "https://example.test/blog", "categories": ["ai"], "score": 10},
            {"title": "GitHub Advisory Database · GitHub", "url": "https://github.com/advisories", "categories": ["security"], "score": 60},
            {"title": "Official advisory", "url": "https://cisa.gov/news", "categories": ["security"], "score": 55},
        ]
        markdown = morning_brief.fallback_agenda(candidates)
        self.assertIn("Official advisory", markdown)
        human_section = markdown.split("## Arthur / Human-Abe Candidates", 1)[1]
        self.assertNotIn("Weak blog", human_section)
        self.assertNotIn("GitHub Advisory Database", human_section)

    def test_morning_brief_specificity_penalizes_generic_landing_pages(self):
        generic = {"title": "Reuters Tech News | Latest Technology News", "url": "https://www.reuters.com/technology/", "snippet": "Latest news"}
        generic_advisory = {"title": "GitHub Advisory Database · GitHub", "url": "https://github.com/advisories", "snippet": "Security advisories"}
        specific = {"title": "ServiceNow discloses security incident exposing customer data", "url": "https://www.bleepingcomputer.com/news/security/servicenow-discloses-security-incident-exposing-customer-data/", "snippet": "incident advisory"}
        self.assertFalse(morning_brief.is_specific_candidate(generic))
        self.assertLess(morning_brief.score_candidate(generic_advisory, "security"), 20)
        self.assertTrue(morning_brief.is_specific_candidate(specific))

    def test_search_backend_unavailable_stops_more_searches(self):
        session = spider.SpiderSession(question="search health", max_sources=5, max_reads=3)
        spider.SEARCH_BACKEND_UNAVAILABLE = True
        self.assertIsNone(spider.next_search_action(session))
        self.assertTrue(spider.search_exhausted(session))
        self.assertTrue(session.coverage["search_backend_unavailable"])

    def test_extract_claims_uses_full_source_text_not_snippet(self):
        session = spider.SpiderSession(question="full text", max_sources=5, max_reads=3)
        long_text = "start " + ("middle " * 200) + "needle-at-end"
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": long_text, "score": 20}, seed=True)

        seen_prompts = []

        def fake_model(prompt, *args, **kwargs):
            seen_prompts.append(prompt)
            return json.dumps({"summary": "Saw full text.", "claims": [{"claim": "Needle was visible.", "confidence": "high"}], "contradictions": []})

        with patch.object(spider, "call_nanbeige", side_effect=fake_model):
            spider.execute_action(session, {"tool": "extract_claims", "source_id": sid})

        self.assertIn("needle-at-end", seen_prompts[0])
        self.assertEqual(session.claims[0]["claim"], "Needle was visible.")

    def test_low_quality_read_is_marked_irrelevant_without_model_summary(self):
        session = spider.SpiderSession(question="quality", depth="standard", max_sources=5, max_reads=3)
        fake_source = {
            "requested_url": "https://example.test/pdf",
            "url": "https://example.test/pdf",
            "final_url": "https://example.test/pdf",
            "adapter": "web",
            "text": "[PDF fetched; no extraction]",
            "score": 0,
            "extracted_chars": 28,
        }
        with patch.object(spider, "fetch_webpage", return_value=fake_source), patch.object(spider, "call_nanbeige") as model:
            spider.execute_action(session, {"tool": "read_url", "url": "https://example.test/pdf"})
        model.assert_not_called()
        self.assertEqual(session.sources[0]["relevance"], "irrelevant")

    def test_irrelevant_source_is_marked_and_does_not_add_claims(self):
        session = spider.SpiderSession(question="target topic", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "unrelated page", "score": 20}, seed=False)
        model_json = json.dumps({"relevance": "irrelevant", "relevance_reason": "No mention of target topic.", "summary": "No relevant information.", "claims": [{"claim": "Should not be added.", "confidence": "high"}], "contradictions": []})
        with patch.object(spider, "call_nanbeige", return_value=model_json):
            spider.summarize_source(session, sid, "unrelated page")
        self.assertEqual(spider.get_source(session, sid)["relevance"], "irrelevant")
        self.assertEqual(session.claims, [])

    def test_unparseable_summary_keeps_markdown_card_as_partial(self):
        session = spider.SpiderSession(question="target topic", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "relevant source", "score": 20}, seed=False)
        with patch.object(spider, "call_nanbeige", return_value="not json but some prose"):
            spider.summarize_source(session, sid, "relevant source")
        self.assertEqual(spider.get_source(session, sid)["relevance"], "partial")
        self.assertEqual(spider.get_source(session, sid)["structuring_status"], "fallback_markdown_card")

    def test_source_card_to_json_structurer_adds_claims(self):
        session = spider.SpiderSession(question="target topic", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "target topic evidence", "score": 20}, seed=False)
        card = "SOURCE_RELEVANCE: partial\nSUMMARY: Retried.\nKEY_CLAIMS:\n- Retried claim.\nEVIDENCE_SNIPPETS:\n- target topic evidence\nCONTRADICTIONS: none\nFOLLOW_UP_LINKS: none"
        structured = json.dumps({"relevance": "partial", "summary": "Retried.", "claims": [{"claim": "Retried claim.", "confidence": "medium", "evidence": "target topic evidence"}], "contradictions": []})
        with patch.object(spider, "call_nanbeige", return_value=card), patch.object(spider, "call_structurer_model", return_value=structured):
            spider.summarize_source(session, sid, "target topic evidence")
        self.assertEqual(spider.get_source(session, sid)["relevance"], "partial")
        self.assertEqual(session.claims[0]["claim"], "Retried claim.")

    def test_target_specific_source_gets_relevance_retry_for_comparison_questions(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=5, max_reads=3)
        sid = session.add_source(
            {
                "requested_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
                "url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
                "final_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
                "adapter": "web",
                "text": "Ollama API runs on localhost:11434 and exposes model endpoints.",
                "score": 35,
                "target_hint": "ollama",
            },
            seed=False,
        )
        card = "SOURCE_RELEVANCE: partial target-specific evidence\nSUMMARY: Ollama API docs.\nKEY_CLAIMS:\n- Ollama serves an API on localhost:11434.\nEVIDENCE_SNIPPETS:\n- localhost:11434\nCONTRADICTIONS: none\nFOLLOW_UP_LINKS: none"
        structured = json.dumps({"relevance": "partial", "relevance_reason": "Target-specific Ollama evidence.", "summary": "Ollama API docs.", "claims": [{"claim": "Ollama serves an API on localhost:11434.", "confidence": "high", "evidence": "localhost:11434"}], "contradictions": []})
        with patch.object(spider, "call_nanbeige", return_value=card), patch.object(spider, "call_structurer_model", return_value=structured):
            spider.summarize_source(session, sid, "Ollama API runs on localhost:11434 and exposes model endpoints.")
        self.assertEqual(spider.get_source(session, sid)["relevance"], "partial")
        self.assertEqual(session.claims[0]["claim"], "Ollama serves an API on localhost:11434.")

    def test_target_specific_source_without_claims_gets_claim_retry(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=5, max_reads=3)
        sid = session.add_source(
            {
                "requested_url": "https://huggingface.co/docs/inference-endpoints/engines/llama_cpp",
                "url": "https://huggingface.co/docs/inference-endpoints/engines/llama_cpp",
                "final_url": "https://huggingface.co/docs/inference-endpoints/engines/llama_cpp",
                "adapter": "web",
                "text": "llama.cpp engine supports local serving with CPU and GPU execution.",
                "score": 35,
                "target_hint": "llama.cpp",
            },
            seed=False,
        )
        card = "SOURCE_RELEVANCE: partial\nSUMMARY: llama.cpp docs.\nKEY_CLAIMS:\n- llama.cpp can be used for local serving.\nEVIDENCE_SNIPPETS:\n- local serving\nCONTRADICTIONS: none\nFOLLOW_UP_LINKS: none"
        structured = json.dumps({"relevance": "partial", "summary": "llama.cpp docs.", "claims": [{"claim": "llama.cpp can be used for local serving.", "confidence": "medium", "evidence": "local serving"}], "contradictions": []})
        with patch.object(spider, "call_nanbeige", return_value=card), patch.object(spider, "call_structurer_model", return_value=structured):
            spider.summarize_source(session, sid, "llama.cpp engine supports local serving with CPU and GPU execution.")
        self.assertEqual(session.claims[0]["claim"], "llama.cpp can be used for local serving.")

    def test_depth_gating_blocks_premature_final_report(self):
        session = spider.SpiderSession(question="deep", depth="deep", max_steps=120, max_sources=200, max_reads=80)
        session.current_step = 2
        session.read_count = 2
        session.sources = [{"source_id": "S1", "seed": True, "score": 50, "relevance": "seed"}]
        session.claims = [{"claim_id": "C1", "claim": "Claim", "source_ids": ["S1"], "confidence": "high"} for _ in range(5)]
        self.assertFalse(spider.coverage_sufficient(session))

        session.current_step = session.min_steps
        session.read_count = session.min_reads
        session.sources = [{"source_id": f"S{i}", "seed": i == 1, "score": 50, "relevance": "seed" if i == 1 else "relevant"} for i in range(1, session.min_sources + 1)]
        self.assertTrue(spider.coverage_sufficient(session))

    def test_evaluate_coverage_clears_stale_missing_comparison_targets(self):
        session = spider.SpiderSession(question="Compare llama.cpp server, Ollama, and vLLM for local LLM serving", max_sources=8, max_reads=5)
        session.coverage["missing_comparison_targets"] = ["llama.cpp", "ollama", "vllm"]
        session.sources = [
            {"source_id": "S1", "title": "llama.cpp server", "url": "https://github.com/ggml-org/llama.cpp", "final_url": "https://github.com/ggml-org/llama.cpp", "relevance": "partial"},
            {"source_id": "S2", "title": "Ollama API", "url": "https://github.com/ollama/ollama", "final_url": "https://github.com/ollama/ollama", "relevance": "partial"},
            {"source_id": "S3", "title": "vLLM Quickstart", "url": "https://docs.vllm.ai", "final_url": "https://docs.vllm.ai", "relevance": "partial"},
        ]
        spider.evaluate_coverage(session)
        self.assertNotIn("missing_comparison_targets", session.coverage)

    def test_irrelevant_sources_do_not_satisfy_coverage_source_minimum(self):
        session = spider.SpiderSession(question="coverage", depth="standard", max_sources=10, max_reads=10)
        session.current_step = session.min_steps
        session.read_count = session.min_reads
        session.claims = [{"claim_id": f"C{i}", "claim": "Claim", "source_ids": ["S1"], "confidence": "high"} for i in range(5)]
        session.sources = [
            {"source_id": "S1", "seed": True, "score": 50, "relevance": "seed"},
            {"source_id": "S2", "seed": False, "score": 50, "relevance": "irrelevant"},
            {"source_id": "S3", "seed": False, "score": 50, "relevance": "irrelevant"},
            {"source_id": "S4", "seed": False, "score": 50, "relevance": "irrelevant"},
        ]
        self.assertFalse(spider.coverage_sufficient(session))

    def test_verify_claim_ledger_updates_statuses_and_contradictions(self):
        session = spider.SpiderSession(question="verify claims", max_sources=5, max_reads=3)
        sid = session.add_source(
            {
                "requested_url": "https://example.test/source",
                "url": "https://example.test/source",
                "final_url": "https://example.test/source",
                "adapter": "web",
                "text": "The system supports feature A. It does not support feature B.",
                "score": 50,
                "relevance": "relevant",
            },
            seed=False,
        )
        session.add_claim("The system supports feature A.", [sid], "high")
        session.add_claim("The system supports feature B.", [sid], "medium")
        verifier_json = json.dumps(
            [
                {"claim_id": "C1", "verification": "supported", "note": "Feature A is explicit."},
                {"claim_id": "C2", "verification": "contradicted", "note": "Source says feature B is not supported."},
            ]
        )
        with patch.object(spider, "call_nanbeige", return_value=verifier_json):
            spider.verify_claim_ledger(session)
        self.assertEqual(session.claims[0]["verification"], "supported")
        self.assertEqual(session.claims[1]["verification"], "contradicted")
        self.assertEqual(session.contradictions[0]["claim_id"], "C2")

    def test_verify_claim_ledger_accepts_object_wrapper(self):
        session = spider.SpiderSession(question="verify wrapped", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "Claim.", "score": 20, "relevance": "relevant"}, seed=False)
        session.add_claim("Claim.", [sid], "medium")
        verifier_json = json.dumps({"verifications": [{"claim_id": "C1", "verification": "supported", "note": "Wrapped response."}]})
        with patch.object(spider, "call_nanbeige", return_value=verifier_json):
            spider.verify_claim_ledger(session)
        self.assertEqual(session.claims[0]["verification"], "supported")

    def test_verify_claim_ledger_retries_unparseable_verifier_output(self):
        session = spider.SpiderSession(question="verify retry", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "Claim.", "score": 20, "relevance": "relevant"}, seed=False)
        session.add_claim("Claim.", [sid], "medium")
        retry_json = json.dumps([{"claim_id": "C1", "verification": "supported", "note": "Retry parsed."}])
        with patch.object(spider, "call_nanbeige", side_effect=["not json", retry_json]):
            spider.verify_claim_ledger(session)
        self.assertEqual(session.claims[0]["verification"], "supported")
        self.assertEqual(session.claims[0]["verification_note"], "Retry parsed.")

    def test_verify_claim_ledger_heuristic_fallback(self):
        session = spider.SpiderSession(question="verify fallback", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "Claim.", "score": 20, "relevance": "relevant"}, seed=False)
        session.add_claim("Claim.", [sid], "medium")
        with patch.object(spider, "call_nanbeige", return_value="not json"):
            spider.verify_claim_ledger(session)
        self.assertEqual(session.claims[0]["verification"], "source_supported_unverified")
        self.assertIn("not independently verified", session.claims[0]["verification_note"])

    def test_sanitize_report_corrects_nanbeige_parameter_count(self):
        session = spider.SpiderSession(question="Nanbeige4.1-3B", max_sources=5, max_reads=3)
        report = "Nanbeige4.1-3B is a 30-billion-parameter model."
        self.assertIn("3B-parameter", spider.sanitize_report(report, session))
        self.assertNotIn("30-billion", spider.sanitize_report(report, session))

    def test_sanitize_report_corrects_free_threading_allocator_typo(self):
        session = spider.SpiderSession(question="CPython free-threading from PEP 703", max_sources=5, max_reads=3)
        report = "Optimistic locking via Maced allocator helps. Another section says via Maced (mimalloc allocator). Markdown says via **Maced** (mimalloc allocator)."
        cleaned = spider.sanitize_report(report, session)
        self.assertIn("mimalloc allocator", cleaned)
        self.assertNotIn("Maced", cleaned)

    def test_sanitize_report_corrects_free_threading_overclaims(self):
        session = spider.SpiderSession(question="CPython free-threading from PEP 703", max_sources=5, max_reads=3)
        report = "Extensions must be compiled with `--disable-gil` to work. Build extensions with `--disable-gil` configure option. Add `Py_mod_gil = 1` slot. Use `PyMem_Malloc` exclusively; handle RC. Avoid `PyObject_Malloc`."
        cleaned = spider.sanitize_report(report, session)
        self.assertIn("CPython itself is configured with `--disable-gil`", cleaned)
        self.assertIn("Build and test extensions against a free-threaded Python", cleaned)
        self.assertIn("`Py_MOD_GIL_NOT_USED`", cleaned)
        self.assertIn("documented memory domain", cleaned)
        self.assertNotIn("PyMemAlloc", spider.sanitize_report("PyMemAlloc typo.", session))

    def test_sanitize_report_appends_excluded_source_audit(self):
        session = spider.SpiderSession(question="audit", max_sources=5, max_reads=3)
        session.search_history = [{"query": "audit", "result_count": 1}]
        session.sources = [
            {"source_id": "S1", "title": "Good", "url": "https://example.test/good", "final_url": "https://example.test/good", "relevance": "relevant"},
            {"source_id": "S2", "title": "Bad", "url": "https://example.test/bad", "final_url": "https://example.test/bad", "relevance": "irrelevant", "relevance_reason": "Unrelated."},
        ]
        report = spider.sanitize_report("# Report\nNo external searches conducted beyond supplied materials.", session)
        self.assertIn("Reviewed But Excluded Sources", report)
        self.assertIn("External searches were conducted", report)
        self.assertIn("[S2]", report)
        self.assertIn("Unrelated.", report)

    def test_sanitize_report_rewrites_misleading_no_external_research_wording(self):
        session = spider.SpiderSession(question="audit", max_sources=5, max_reads=3)
        session.sources = [{"source_id": "S1", "title": "Good", "url": "https://example.test/good", "final_url": "https://example.test/good", "relevance": "relevant"}]
        report = spider.sanitize_report("# Report\nNo additional external research was performed.", session)
        self.assertIn("No research was performed beyond the retrieved/cited sources and stored claim ledger.", report)
        self.assertNotIn("No additional external research was performed", report)

    def test_sanitize_report_appends_report_qa_notes(self):
        session = spider.SpiderSession(question="qa", max_sources=5, max_reads=3)
        session.sources = [{"source_id": "S1", "title": "Good", "url": "https://example.test/good", "final_url": "https://example.test/good", "relevance": "relevant", "score": 50}]
        session.claims = [{"claim_id": "C1", "claim": "Claim", "source_ids": ["S1"], "confidence": "medium", "verification": "source_supported_unverified", "evidence": []}]
        session.coverage["evaluation"] = spider.evaluate_coverage(session)
        report = spider.sanitize_report("# Report\nNo citations here.", session)
        self.assertIn("Report QA Notes", report)
        self.assertIn("not independently verified", report)
        self.assertIn("lack explicit evidence snippets", report)

    def test_deep_report_excludes_no_evidence_claims_from_synthesis(self):
        session = spider.SpiderSession(question="deep report", depth="deep", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "Evidence text.", "score": 40, "relevance": "relevant"}, seed=False)
        session.add_claim("Claim with evidence.", [sid], "high", evidence="Evidence text")
        session.add_claim("Claim without evidence.", [sid], "high")
        seen_prompts = []

        def fake_model(prompt, *args, **kwargs):
            seen_prompts.append(prompt)
            return "# Spider Research Report\n\n## Executive Summary\n\nDone."

        with patch.object(spider, "verify_claim_ledger"), patch.object(spider, "call_nanbeige", side_effect=fake_model):
            report = spider.generate_report(session)

        self.assertIn("Claim with evidence.", seen_prompts[0])
        self.assertNotIn("Claim without evidence.", seen_prompts[0])
        self.assertEqual(session.coverage["claims_excluded_from_report_missing_evidence"], ["C2"])
        self.assertNotIn("claims lack explicit evidence snippets", report)
        self.assertIn("no-evidence claims were excluded", report)

    def test_deep_report_prompt_requests_richer_sections(self):
        session = spider.SpiderSession(question="deep report", depth="deep", max_sources=5, max_reads=3)
        sid = session.add_source({"requested_url": "file", "url": "file", "final_url": "file", "adapter": "file", "text": "Evidence text.", "score": 40, "relevance": "relevant"}, seed=False)
        session.add_claim("Claim with evidence.", [sid], "high", evidence="Evidence text")
        seen_prompts = []

        def fake_model(prompt, *args, **kwargs):
            seen_prompts.append(prompt)
            return "# Spider Research Report\n\n## Executive Summary\n\nDone."

        with patch.object(spider, "verify_claim_ledger"), patch.object(spider, "call_nanbeige", side_effect=fake_model):
            spider.generate_report(session)

        self.assertIn("## Methodology / Research Path", seen_prompts[0])
        self.assertIn("## Practical Checklist", seen_prompts[0])
        self.assertIn("## Claim Verification / Confidence", seen_prompts[0])

    def test_clarify_policy_auto_stops_and_writes_question_artifacts(self):
        session = spider.SpiderSession(question="research this", max_sources=5, max_reads=3)
        brief = json.dumps(
            {
                "question": "research this",
                "scope": "unclear",
                "exclusions": [],
                "target_audience": "Abe",
                "assumptions": [],
                "output_style": "report",
                "needs_clarification": True,
                "clarifying_questions": ["What topic should Spider research?"],
            }
        )
        with patch.object(spider, "call_controller_model", return_value=brief):
            may_run = spider.handle_workflow_gates(session, clarify_policy="auto", answers_path=None, plan_only=False, yes=False, approve_plan_path=None)
        self.assertFalse(may_run)
        self.assertEqual(session.status, "needs_clarification")
        self.assertTrue((spider.SESSION_DIR / "research_brief.json").exists())
        self.assertTrue((spider.SESSION_DIR / "clarifying_questions.md").exists())
        self.assertTrue((spider.SESSION_DIR / "clarifying_questions.json").exists())

    def test_clarify_policy_never_does_not_stop_for_ambiguous_intake(self):
        session = spider.SpiderSession(question="research this", max_sources=5, max_reads=3)
        brief = json.dumps({"question": "research this", "needs_clarification": True, "clarifying_questions": ["Clarify?"], "scope": "unclear"})
        plan = json.dumps({"scope": "Proceed with supplied question.", "search_strategy": ["search"], "source_priorities": ["official"], "verification_plan": ["verify"], "expected_output_sections": ["Answer"]})
        with patch.object(spider, "call_controller_model", side_effect=[brief, plan]):
            may_run = spider.handle_workflow_gates(session, clarify_policy="never", answers_path=None, plan_only=False, yes=True, approve_plan_path=None)
        self.assertTrue(may_run)
        self.assertTrue(session.plan_approved)
        self.assertFalse((spider.SESSION_DIR / "clarifying_questions.json").exists())

    def test_answers_file_leads_to_plan_artifacts(self):
        session = spider.SpiderSession(question="research this", max_sources=5, max_reads=3)
        answers = Path(self.tmp.name) / "answers.md"
        answers.write_text("Scope: compare Nanbeige and VibeThinker for local research.", encoding="utf-8")
        brief = json.dumps({"question": "research this", "needs_clarification": True, "clarifying_questions": ["Scope?"], "scope": "unclear"})
        plan = json.dumps({"scope": "Compare two local research models.", "search_strategy": ["read model cards"], "source_priorities": ["HF"], "verification_plan": ["claim ledger"], "expected_output_sections": ["Comparison"]})
        with patch.object(spider, "call_controller_model", side_effect=[brief, plan]):
            may_run = spider.handle_workflow_gates(session, clarify_policy="auto", answers_path=str(answers), plan_only=True, yes=False, approve_plan_path=None)
        self.assertFalse(may_run)
        self.assertIn("VibeThinker", session.clarifying_answers)
        self.assertEqual(session.status, "awaiting_plan_approval")
        self.assertTrue((spider.SESSION_DIR / "research_plan.md").exists())
        self.assertTrue((spider.SESSION_DIR / "research_plan.json").exists())

    def test_plan_approval_allows_run(self):
        session = spider.SpiderSession(question="approved research", max_sources=5, max_reads=3)
        plan_path = Path(self.tmp.name) / "research_plan.json"
        plan_path.write_text(json.dumps({"scope": "Approved scope", "search_strategy": ["search"]}), encoding="utf-8")
        with patch.object(spider, "call_controller_model", return_value=json.dumps({"question": "approved research", "needs_clarification": False})):
            may_run = spider.handle_workflow_gates(session, clarify_policy="never", answers_path=None, plan_only=False, yes=False, approve_plan_path=str(plan_path))
        self.assertTrue(may_run)
        self.assertTrue(session.plan_approved)
        self.assertEqual(session.research_plan["scope"], "Approved scope")

    def test_approve_plan_bypasses_clarify_policy(self):
        session = spider.SpiderSession(question="approved research", max_sources=5, max_reads=3)
        plan_path = Path(self.tmp.name) / "research_plan.json"
        plan_path.write_text(json.dumps({"scope": "Approved despite clarify policy", "search_strategy": ["search"]}), encoding="utf-8")
        brief = json.dumps({"question": "approved research", "needs_clarification": True, "clarifying_questions": ["Clarify?"]})
        with patch.object(spider, "call_controller_model", return_value=brief):
            may_run = spider.handle_workflow_gates(session, clarify_policy="always", answers_path=None, plan_only=False, yes=False, approve_plan_path=str(plan_path))
        self.assertTrue(may_run)
        self.assertTrue(session.plan_approved)
        self.assertFalse((spider.SESSION_DIR / "clarifying_questions.json").exists())

    def test_spider_lock_fails_fast_and_releases(self):
        lock_path = Path(self.tmp.name) / "spider.lock"
        with spider.SpiderRunLock(lock_path, wait=False):
            with self.assertRaises(spider.SpiderBusyError):
                with spider.SpiderRunLock(lock_path, wait=False):
                    pass
        with spider.SpiderRunLock(lock_path, wait=False):
            self.assertTrue(lock_path.exists())

    def test_read_file_action_summarizes_specific_file(self):
        session = spider.SpiderSession(question="file action", max_sources=5, max_reads=3)
        path = Path(self.tmp.name) / "source.md"
        path.write_text("# Source\n\nThis technical report contains target topic evidence, benchmark evaluation, implementation details, and migration guidance.", encoding="utf-8")
        model_json = json.dumps({"relevance": "relevant", "summary": "File evidence.", "claims": [{"claim": "File has target topic evidence.", "confidence": "high", "evidence": "target topic evidence"}], "contradictions": []})
        with patch.object(spider, "min_extracted_source_score", return_value=0), patch.object(spider, "call_nanbeige", return_value=model_json):
            observation = spider.execute_action(session, {"tool": "read_file", "path": str(path)})
        self.assertEqual(observation["source_id"], "S1")
        self.assertEqual(session.read_count, 1)
        self.assertEqual(session.claims[0]["claim"], "File has target topic evidence.")

    def test_morning_brief_does_not_post_unless_flag_is_set(self):
        result = {"markdown": "# brief", "candidates": []}
        with patch.object(morning_brief, "generate_agenda", return_value=result), patch.object(morning_brief, "post_watercooler") as post:
            with patch.object(sys, "argv", ["morning_brief.py", "--categories", "ai"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    morning_brief.main()
        post.assert_not_called()

    def test_morning_brief_post_path_uses_watercooler_and_sender(self):
        with patch.object(morning_brief.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                morning_brief.post_watercooler("# brief")
        cmd = run.call_args.args[0]
        self.assertIn("chat:watercooler", cmd)
        self.assertNotIn("chat:synchronous", cmd)
        from_index = cmd.index("from")
        self.assertEqual(cmd[from_index + 1], "MorningBrief")


@unittest.skipUnless(os.environ.get("SPIDER_RUN_INTEGRATION_TESTS") == "1", "set SPIDER_RUN_INTEGRATION_TESTS=1 to run live integration tests")
class SpiderIntegrationTests(unittest.TestCase):
    def test_live_brave_search_returns_normalized_results(self):
        if not os.environ.get("SPIDER_BRAVE_API_KEY"):
            self.skipTest("SPIDER_BRAVE_API_KEY is not set")
        session = spider.SpiderSession(question="CPython free-threading C extension HOWTO", max_sources=5, max_reads=3)
        with patch.dict(os.environ, {"SPIDER_SEARCH_BACKENDS": "brave", "SPIDER_SEARCH_CACHE_TTL_HOURS": "0"}):
            results, meta = spider.fetch_search(session, "CPython free-threading C extension HOWTO", max_results=3)
        self.assertEqual(meta.get("backend"), "brave")
        self.assertTrue(results)
        self.assertTrue(any("docs.python.org" in item.get("url", "") for item in results))

    def test_live_morning_brief_ai_dry_run(self):
        result = morning_brief.generate_agenda(["ai"], output_json=True, min_score=20)
        self.assertEqual(result.get("channel"), "chat:watercooler")
        self.assertNotIn("chat:synchronous", json.dumps(result))
        self.assertNotIn("<think", json.dumps(result).lower())


if __name__ == "__main__":
    unittest.main()
