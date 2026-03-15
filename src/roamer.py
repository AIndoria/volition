#!/usr/bin/env python3
"""
roamer.py - The Leashed Investigator
Volition 8.0.0 Subroutine

Role: Multi-turn, Read-Only investigation agent.
Model: Qwen-2.5-14B-Coder (via Local API)
Host: Runs LOCALLY inside the Abe LXC.

Usage:
  python3 roamer.py --directive "Find the error" --debug
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from typing import List, Dict, Any
from datetime import datetime
import shlex
import re

# Third Party
try:
    import redis
    from openai import OpenAI
except ImportError:
    print("Missing dependencies. Run: pip install redis openai")
    sys.exit(1)

# --- CONFIGURATION ---
# Default to local vLLM/Ollama/LlamaCpp
DEFAULT_API_URL = os.environ.get("ROAMER_API_URL", "") 
DEFAULT_API_KEY = os.environ.get("ROAMER_API_KEY", "volition-local")
DEFAULT_MODEL = os.environ.get("MODEL_ROAMER", "qwen-2.5-14b-coder")

DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "")
MAX_TURNS = 15

# LOGGING
logging.basicConfig(level=logging.INFO, format="[ROAMER] %(message)s")
logger = logging.getLogger("roamer")

class SafeShell:
    """Enforces read-only discipline on the Investigator."""
    
    ALLOWED_CMDS = [
        "ls", "cat", "grep", "head", "tail", "find", 
        "stat", "df", "du", "whoami", "date", "echo", 
        "awk", "sed", "cut", "sort", "uniq", "wc", "uptime", "free"
    ]

    FORBIDDEN_FLAGS = ["-i", "sudo", "su"]
    
    # Hostnames must not start with '-' to avoid option injection in ssh-like commands.
    HOSTNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")

    def __init__(self, target_host="local"):
        if target_host != "local" and not self.HOSTNAME_PATTERN.match(str(target_host)):
            logger.error(f"FATAL: Invalid target_host '{target_host}'.")
            self.target_host = "INVALID_HOST"
        else:
            self.target_host = target_host

    def validate(self, cmd: str) -> (bool, str):
        if not isinstance(cmd, str) or not cmd.strip():
            return False, "Empty command."
            
        # We ALLOW the pipe '|', but block chaining, subshells, and redirection
        forbidden_chars = ['&', ';', '$', '`', '<', '>', '\n']
        if any(c in cmd for c in forbidden_chars):
            return False, "Chaining (; &), subshells ($ `), and redirects (> <) are blocked."

        # Split the pipeline and validate every single executable
        segments = cmd.split('|')
        for segment in segments:
            segment = segment.strip()
            if not segment: return False, "Empty pipe segment."
            
            try:
                tokens = shlex.split(segment)
            except ValueError as e:
                return False, f"Command parsing error: {e}"
                
            if not tokens: return False, "Empty command in pipeline."
                
            base = tokens[0]
            if base not in self.ALLOWED_CMDS:
                return False, f"Command '{base}' is not in the read-only whitelist."

            if any(f in tokens for f in self.FORBIDDEN_FLAGS):
                return False, f"Forbidden flag detected in '{base}' segment."

        return True, ""

    def execute(self, cmd: str) -> str:
        if self.target_host == "INVALID_HOST":
            return "SAFETY BLOCK: Invalid target_host provided. Aborting to prevent local execution."

        is_safe, reason = self.validate(cmd)
        if not is_safe:
            return f"SAFETY BLOCK: {reason}"

        try:
            if self.target_host != "local":
                # SSH as an array (shell=False) to prevent host injection
                final_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=5", self.target_host, cmd
                ]
                logger.info(f"EXEC REMOTE: {' '.join(final_cmd)}")
                result = subprocess.run(final_cmd, shell=False, capture_output=True, text=True, timeout=15)
            else:
                # Local as a string (shell=True) because validate() proved every segment is safe
                logger.info(f"EXEC LOCAL: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            
            stdout, stderr = result.stdout, result.stderr

            if len(stdout) > 4000:
                stdout = stdout[:4000] + "\n... [TRUNCATED: Output exceeded 4000 chars] ..."
            
            output = ""
            if stdout: output += f"STDOUT:\n{stdout}"
            if stderr: output += f"\nSTDERR:\n{stderr}"
            return output or "(No Output)"

        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out (15s limit)."
        except Exception as e:
            return f"ERROR: Execution failed: {e}"
    
# --- THE INVESTIGATOR AGENT ---
class RoamerAgent:
    def __init__(self, directive, target_host, output_inbox, debug_mode=False, api_url=None, model=None):
        self.directive = directive
        self.shell = SafeShell(target_host)
        self.output_inbox = output_inbox
        self.debug_mode = debug_mode
        self.model = model or DEFAULT_MODEL
        
        url = api_url or DEFAULT_API_URL
        self.client = OpenAI(base_url=url, api_key=DEFAULT_API_KEY)
        
        self.history = [
            {"role": "system", "content": self._build_system_prompt(target_host)}
        ]

    def _build_system_prompt(self, host):
        return f"""
You are The Roamer, a specialized read-only investigator script.
You are currently running inside an infrastructure container.
TARGET HOST: {host}

YOUR DIRECTIVE: {self.directive}

TOOLS AVAILABLE:
1. execute_shell: Run a shell command. 
   - CONSTRAINTS: READ-ONLY. Allowed: ls, cat, grep, find, head, tail, df, du.
   - FORBIDDEN: rm, mv, cp, nano, vim, sed -i, > redirection, sudo.
   - If the user asks for a fix, INVESTIGATE first, then propose the fix in your final report. DO NOT execute it.

2. finish_investigation: Call this when you have found the answer or failed.
   - ARGUMENTS: final_report (Markdown summary of findings and suggested next steps).

PROTOCOL:
- You are in a loop. You observe -> think -> act.
- Do not hallucinate file contents. cat them.
- Do not guess paths. ls them.
- If you hit "Permission denied", report it; do not try to sudo.
- Keep your turns efficient.
"""

    def run(self):
        logger.info(f"Starting Investigation on {self.shell.target_host} (Debug: {self.debug_mode})")
        
        for turn in range(MAX_TURNS):
            logger.info(f"Turn {turn+1}/{MAX_TURNS}")
            
            # 1. Get LLM Response
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    tools=self._get_tool_schema(),
                    tool_choice="auto",
                    temperature=0.1 # Low temp for precise logic
                )
                msg = response.choices[0].message
                # Normalize assistant message into a plain dict for history so it doesn't crash turn 2
                assistant_entry = {
                    "role": msg.role,
                    "content": msg.content,
                }
                if getattr(msg, "tool_calls", None):
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        } for tc in msg.tool_calls
                    ]
                self.history.append(assistant_entry)
            except Exception as e:
                self._report_failure(f"LLM API Failure: {e}")
                return

            # 2. Check for Tool Calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        logger.error("LLM generated invalid JSON arguments")
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "ERROR: Invalid JSON arguments provided."
                        })
                        continue

                    logger.info(f"Tool Call: {func_name}")

                    if func_name == "finish_investigation":
                        self._report_success(args.get("final_report"))
                        return

                    elif func_name == "execute_shell":
                        output = self.shell.execute(args.get("command"))
                        
                        if self.debug_mode:
                            print(f"\n--- CMD OUT ---\n{output}\n---------------")

                        # Feed result back to history
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output
                        })
            else:
                # If LLM talks without tools, just push it back as user prompt to force action
                content = msg.content or ""
                logger.warning(f"LLM replied without tools: {content[:50]}...")
                self.history.append({
                    "role": "user", 
                    "content": "You must use a tool to proceed (execute_shell) or end the session (finish_investigation)."
                })

        self._report_failure("Max turns reached without conclusion.")

    def _get_tool_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute a read-only shell command to inspect the system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command (e.g., 'ls -la /var/log')"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_investigation",
                    "description": "Submit the final report and end the session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_report": {"type": "string", "description": "Markdown summary of findings."}
                        },
                        "required": ["final_report"]
                    }
                }
            }
        ]

    def _report_success(self, report):
        logger.info("Investigation Complete.")
        self._push_result("RoamerReport", report)

    def _report_failure(self, error):
        logger.error(f"Investigation Failed: {error}")
        self._push_result("RoamerFailed", f"Investigation aborted: {error}")

    def _push_result(self, event_type, content):
        payload = {
            "type": "GUPPIEvent",
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "content": content,
            "meta": {"source": "roamer", "target": self.shell.target_host}
        }
        
        if self.debug_mode:
            print("\n" + "="*40)
            print(f"FINAL PAYLOAD ({event_type}):")
            print(json.dumps(payload, indent=2))
            print("="*40 + "\n")
        else:
            try:
                r = redis.from_url(DEFAULT_REDIS_URL, decode_responses=True)
                r.lpush(self.output_inbox, json.dumps(payload))
                logger.info(f"Result pushed to {self.output_inbox}")
            except Exception as e:
                logger.error(f"Redis Push Failed: {e}")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volition Roamer")
    parser.add_argument("--directive", required=True, help="What to investigate")
    parser.add_argument("--target-host", default="local", help="Hostname (must be in .ssh/config) or 'local'")
    parser.add_argument("--output-inbox", default="inbox:debug", help="Redis inbox to push results to")
    
    # Configuration Overrides
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI-compatible API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--debug", action="store_true", help="Print results to console instead of Redis")

    args = parser.parse_args()

    agent = RoamerAgent(
        args.directive, 
        args.target_host, 
        args.output_inbox, 
        debug_mode=args.debug,
        api_url=args.api_url,
        model=args.model
    )
    agent.run()