#!/usr/bin/env python3
"""
VOLITION GENESIS PROTOCOL v2.3
The "Big Bang" script for Proxmox LXC Spawning.

Role:
1. Interrogates the Human for configuration.
2. Prompts for "Personality" edits (98/99 markdown files).
3. Creates 'abe-01' (Patient Zero).
4. "Infects" it with the Volition codebase.
5. Installs the 'spawn_abe_lxc.sh' script on the HOST only.
6. Establishes the SSH "Umbilical Cord" between Abe-01 and the Host.
"""

import os
import re
import sys
import subprocess
import json
import time
import shutil
import getpass
import socket
from pathlib import Path

# --- COLORS ---
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

# --- CONSTANTS ---
CWD = Path.cwd()
SRC_DIR = CWD / "src"
DOCS_DIR = CWD / "docs"
HOST_BIN_DIR = Path("/root/bin")
HOST_SSH_KEYS = Path("/root/.ssh/authorized_keys")  # Where host tools live

# Defines the "Core Identity" structure strictly as requested
def generate_identity_json(agent_name, human_name):
    return json.dumps({
        "name": agent_name,
        "persona": agent_name,
        "parent": human_name,
        "temp": 1.0,
        "top_k": 0.9
    }, indent=4)

def print_banner():
    print(f"{CYAN}")
    print(r"""
 __      __   _ _ _   _             
 \ \    / /  | (_) | (_)            
  \ \  / /__ | |_| |_ _  ___  _ __  
   \ \/ / _ \| | | __| |/ _ \| '_ \ 
    \  / (_) | | | |_| | (_) | | | |
     \/ \___/|_|_|\__|_|\___/|_| |_|
                                    
    VOLITION ABIVERSE GENESIS v2.1
    [Proxmox LXC Orchestrator]
    """)
    print(f"{RESET}")

def check_prerequisites():
    print(f"{YELLOW}[PRE-FLIGHT CHECKS]{RESET}")
    print("This script assumes the following services are ALREADY running on your network and/or you have access to them:")
    print(f"  1. {CYAN}Redis{RESET} (The nervous system)")
    print(f"  2. {CYAN}Ntfy{RESET} (The alerting backbone): We recommend installing Ntfy server or using https://ntfy.sh, and using it on your phone as well.")
    print("This script will generate the service files for the following. It is highly recommended to run these on the same host as redis (and install appropriate requirements). You will be asked to setup/start these services later after setting up Redis:")
    print(f"  1. {CYAN}Logger Service{RESET}")
    print(f"  2. {CYAN}Heartbeat Monitor{RESET}")
    print("This script will generate the service files for the following. They default to using local ollama. It is ideal that you ensure your local Ollama instance is running and reachable. Alternatively, you can modify the service files to use openrouter. You will be asked to setup/start these services later after setting up Redis:")
    print(f"  3. {CYAN}GPU Worker{RESET} (Embeddings/Summarization)")
    print(f"  4. {CYAN}Ear Service{RESET} (Social Router/Digesting)")
    print("This script will generate the service files for the following. You can run it anywhere where you can access Redis:")
    print(f"  5. {CYAN}Dashboard Service{RESET} (Web Command Dashboard)")
    
    if prompt("Do you understand?", "y").lower() != 'y':
        print(f"{RED}Aborting. You cannot proceed without acknowledging prerequisites.{RESET}")
        sys.exit(1)
    print(f"{GREEN}[OK] Prerequisites Acknowledged.{RESET}\n")


def run_cmd(cmd, shell=False, check=True):
    try:
        if shell and isinstance(cmd, list):
            cmd = " ".join(cmd)
        
        result = subprocess.run(
            cmd, 
            shell=shell, 
            check=check, 
            text=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"{RED}[FAIL] Command: {cmd}{RESET}")
        print(f"{RED}[STDERR] {e.stderr}{RESET}")
        sys.exit(1)

def prompt(text, default=None):
    if default:
        val = input(f"{GREEN}[?]{RESET} {text} [{default}]: ").strip()
        return val if val else default
    else:
        while True:
            val = input(f"{GREEN}[?]{RESET} {text}: ").strip()
            if val: return val

def find_latest_debian_template():
    try:
        out = run_cmd(["pveam", "available", "--section", "system"])
    except Exception:
        print(f"{RED}[FAIL] Unable to query pveam. Is this a Proxmox host?{RESET}")
        sys.exit(1)
    matches = re.findall(r'(debian-12-standard_[\d.]+-\d+_amd64\.tar\.zst)', out)
    if not matches:
        print(f"{RED}[FAIL] No debian-12-standard template found in pveam available.{RESET}")
        print("Try: pveam update")
        sys.exit(1)
    return sorted(matches)[-1]


def ensure_debian_template(template_name):
    print(f"{YELLOW}Checking for LXC template: {template_name}{RESET}")
    try:
        templates = run_cmd(["pveam", "list", "local"])
        if template_name in templates:
            print(f"{GREEN}[OK] Template already present.{RESET}")
            return
    except Exception:
        print(f"{RED}[FAIL] Unable to query pveam. Is this a Proxmox host?{RESET}")
        sys.exit(1)

    print(f"{YELLOW}Template not found. Attempting download...{RESET}")
    try:
        run_cmd(["pveam", "update"])
        run_cmd(["pveam", "download", "local", template_name])
        print(f"{GREEN}[OK] Template downloaded successfully.{RESET}")
    except Exception as e:
        print(f"{RED}[FAIL] Could not download template {template_name}{RESET}")
        print("You may need to:")
        print("  - Check internet connectivity on the Proxmox host")
        print("  - Verify storage 'local' exists and is enabled")
        print(f"Try: pveam update && pveam download local {template_name}")
        sys.exit(1)


def edit_file_pause(filepath):
    """Pauses execution to let user edit a file."""
    print(f"\n{YELLOW}[ATTENTION]{RESET} You must now edit: {CYAN}{filepath.name}{RESET}")
    print("This defines the Abes' Service Map (99) or Personality (98).")
    print("This is MANDATORY. Without this, the Abes will be lobotomized.")
    input(f"Press {GREEN}ENTER{RESET} to open nano...")
    subprocess.run(["nano", str(filepath)])
    print(f"{GREEN}Saved. Continuing...{RESET}\n")
    
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "10.0.0.1"
    
def check_storage(storage_id):
    out = run_cmd(["pvesm", "status"])
    if storage_id not in out:
        print(f"{RED}Storage '{storage_id}' not found.{RESET}")
        sys.exit(1)

def check_bridge(bridge_id):
    out = run_cmd(["ip", "link"])
    if bridge_id not in out:
        print(f"{RED}Network bridge '{bridge_id}' not found.{RESET}")
        sys.exit(1)


def check_redis_connectivity(host, port, password):
    print(f"{YELLOW}Checking Redis connectivity at {host}:{port}...{RESET}")
    try:
        cmd = [
            "redis-cli",
            "-h", host,
            "-p", str(port),
            "-a", password,
            "PING"
        ]
        out = run_cmd(cmd)
        if out.strip() != "PONG":
            raise RuntimeError(f"Unexpected Redis response: {out}")
        print(f"{GREEN}[OK] Redis reachable and authenticated.{RESET}")
    except Exception as e:
        print(f"{RED}[FAIL] Cannot reach Redis: {e}{RESET}")
        print("Fix Redis connectivity before proceeding.")
        sys.exit(1)


def main():
    print_banner()

    if os.geteuid() != 0:
        print(f"{RED}Error: Must run as root (Proxmox Host).{RESET}")
        sys.exit(1)

    check_prerequisites()
    # --- 1. CONFIGURATION GATHERING ---
    print(f"{CYAN}--- PHASE 1: CONFIGURATION ---{RESET}")
    agent_name = "abe-01" # This is important. A lot of services depend on the abes being abes and serializing.
    human_name = prompt("Your name(This substitutes the placeholders in documentation folder)", "Human-Abe")
    contact_method = prompt("Preferred secondary contact method for Abe (email, phone etc)? The default will remain ntfy. We will ask for ntfy url+topic in a minute.", "-not-specified-")

    print("\n[Infrastructure Config]")
    print("\nYou should have a Redis instance running on your local network and reachable by the new LXC.")
    redis_host = prompt("Redis Host", "10.0.0.0")
    redis_port = prompt("Redis Port", "6379")
    redis_pass = prompt("Redis Password", "volition")
    print(f"{YELLOW}Installing redis-tools on host...{RESET}")
    run_cmd(["apt-get", "install", "-y", "-qq", "redis-tools"])
    print(f"{GREEN}[OK] redis-tools installed.{RESET}")
    check_redis_connectivity(redis_host, redis_port, redis_pass)

    print("\n[Search Config]")
    print("\nThe Abes use SearXNG as its default privacy-respecting search engine to search for information. The default is hosted by -The Abe- and is publicly available, but you can host your own instance for better reliability/privacy. Ensure that json results are enabled in your SearXNG instance, or find a public one that does.")
    searxng_url = prompt("SearXNG URL", "https://civitat.es/search")
    
    print("\n[Proxmox Defaults]")
    # Ask for Storage/Bridge once, to reuse for Genesis AND Spawning
    storage_id = prompt("Storage ID for Containers", "local-lvm")
    check_storage(storage_id)
    bridge_id = prompt("Network Bridge", "vmbr0")
    check_bridge(bridge_id)
    host_ip = prompt("Proxmox Host LAN IP (For Abe -> Host SSH)", get_host_ip())
    host_hostname = socket.gethostname()
    print(f"Proxmox Hostname detected as: {GREEN}{host_hostname}{RESET}")
    
    print("\n[Alerting Config]")
    ntfy_url = prompt("Ntfy URL", "https://ntfy.sh/abe_alerts")
    ntfy_token = prompt("Ntfy Token (Optional, Enter to skip)", "")

    print("\n[OpenRouter Config]")
    print("Current Public Version of Volition uses OpenRouter as the LLM provider for flexibility.")
    print("You must have an OpenRouter API key. Sign up at https://openrouter.ai/")
    print("In future versions, I will explicitly enable ollama as it is currently default for embeddings only.")
    openrouter_api_key = prompt("OpenRouter API Key", "sk-xxxxxx")

    # Generate .env content
    env_content = f"""# Volition Environment Config
# Generated by Genesis v2.1
ABE_NAME={agent_name}
ABE_ROOT=/root
PARENT_NODE={host_hostname}

# Redis
REDIS_HOST={redis_host}
REDIS_PORT={redis_port}
REDIS_PASSWORD={redis_pass}
REDIS_URL=redis://:{redis_pass}@{redis_host}:{redis_port}/0

# Alerts
NTFY_URL={ntfy_url}
NTFY_TOKEN={ntfy_token}

# Paths
MEMORY_DIR=/root/memory
LOG_DIR=/root/logs

# --- Network Config ---
SEARXNG_URL={searxng_url}

# --- v6.4 Provider Config (OpenRouter) ---
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY={openrouter_api_key}

# --- Model Routing (Split-Brain) ---
# Flash: Social, Low-Risk (Chat)
OPENROUTER_MODEL_FLASH=google/gemini-3-flash-preview
# Pro: Work, High-Agency (Inbox/Tasks) - Using Thinking Model for complex tasks
# If Gemini-3-flash-preview is listed here, it will have thinking:high set in guppi for Pro.
OPENROUTER_MODEL_PRO=google/gemini-3-flash-preview

# --- Metadata (for OpenRouter Leaderboard) ---
OPENROUTER_SITE_URL=https://volition.indoria.org
OPENROUTER_APP_NAME=Volition

# Identity & Paths (Optional overrides)
# IDENTITY_FILE=/root/.abe-identity
# GENESIS_PROMPT_FILE=0.0-Abe-Genesis_Prompt.md

"""
# --- 1.5 GENERATE INFRA SERVICES & PAUSE ---
    print(f"\n{CYAN}--- PHASE 1.5: INFRASTRUCTURE PREP ---{RESET}")
    print("Generating systemd service files for your Redis/Infra node...")
    
    # Define templates
    # We use the {redis_host} variable from Phase 1 configuration.
    # This ensures the monitor connects to the correct Redis instance, 
    # whether it's localhost or a remote server.
    heartbeat_svc = f"""[Unit]
Description=Volition Heartbeat Monitor
After=network-online.target redis-server.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bin
ExecStart=/usr/bin/python3 /root/bin/heartbeat-monitor.py

Restart=always
RestartSec=5

Environment=REDIS_HOST={redis_host}
Environment=REDIS_PORT={redis_port}
Environment=REDIS_PASSWORD={redis_pass}
Environment=NTFY_URL={ntfy_url}
Environment=NTFY_TOKEN={ntfy_token}

NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""
    
    dashboard_svc = f"""[Unit]
Description=Volition Command Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/volition

Environment=REDIS_HOST={redis_host}
Environment=REDIS_PORT={redis_port}
Environment=REDIS_PASSWORD={redis_pass}

ExecStart=/opt/volition/venv/bin/python /opt/volition/volition_dashboard.py

Restart=always
RestartSec=3

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target

"""

    logger_svc = f"""[Unit]
Description=Volition Immutable Audit Logger
After=network.target redis-server.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /usr/local/bin/volition-logger.py
Environment="REDIS_HOST={redis_host}"
Environment="REDIS_PORT={redis_port}"
Environment="REDIS_PASSWORD={redis_pass}"
Environment="PYTHONUNBUFFERED=1"
Environment=VOLITION_LOG_DIR=/mnt/storage_5/volition_logs
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
    
    gpu_worker_svc = f"""[Unit]
Description=Volition GPU Worker (Embeddings / Heavy Compute)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/volition

Environment=REDIS_HOST={redis_host}
Environment=REDIS_PORT={redis_port}
Environment=REDIS_PASSWORD={redis_pass}

# Default: Ollama local
Environment=EMBEDDING_BACKEND=ollama
Environment=OLLAMA_URL=http://localhost:11434/api
Environment=MODEL_EMBED=nomic-embed-text
Environment=MODEL_SUMMARIZE=mistral

# Optional OpenRouter fallback
Environment=OPENROUTER_API_KEY={openrouter_api_key}
Environment=OPENROUTER_MODEL_EMBED=google/gemini-embedding-001

ExecStart=/opt/volition/venv/bin/python /opt/volition/gpu_worker.py
Restart=always
RestartSec=3

StandardOutput=journal
StandardError=journal
Nice=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
"""
    
    ear_svc = f"""[Unit]
Description=Volition Ear (Social Router & Digest Generator)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/volition

Environment=REDIS_HOST={redis_host}
Environment=REDIS_PORT={redis_port}
Environment=REDIS_PASSWORD={redis_pass}

# Default: Ollama local
Environment=SUMMARIZE_BACKEND=ollama
Environment=OLLAMA_URL=http://localhost:11434/api
Environment=MODEL_SUMMARIZE=mistral

# Optional OpenRouter fallback
Environment=OPENROUTER_API_KEY={openrouter_api_key}
Environment=OPENROUTER_MODEL_SUMMARIZE=google/gemini-3-flash-preview

ExecStart=/opt/volition/venv/bin/python /opt/volition/ear.py
Restart=always
RestartSec=3

StandardOutput=journal
StandardError=journal
Nice=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
"""



    # Write files locally for easy transfer
    with open(CWD / "volition-heartbeat.service", "w") as f: f.write(heartbeat_svc)
    with open(CWD / "volition-logger.service", "w") as f: f.write(logger_svc)
    with open(CWD / "volition-gpu-worker.service", "w") as f: f.write(gpu_worker_svc)
    with open(CWD / "volition-ear.service", "w") as f: f.write(ear_svc)
    with open(CWD / "volition-dashboard.service", "w") as f: f.write(dashboard_svc)


    
    print(f"{GREEN}Generated 'volition-heartbeat.service', 'volition-gpu-worker.service', "
      f"'volition-ear.service', and 'volition-logger.service' in {CWD}.{RESET}")

    print(f"{YELLOW}[ACTION REQUIRED]{RESET} You must now set up the Infrastructure Node(s) "
          f"(Redis target: {redis_host}).\n")

    print(f"{YELLOW}--- PART 1: REQUIRED LOGGING / HEARTBEAT SERVICES ---{RESET}")
    print("1. Copy the following files to /etc/systemd/system/ on the Redis/infra node:")
    print("   - volition-heartbeat.service")
    print("   - volition-logger.service")
    print()
    print(f"2. Copy '{SRC_DIR}/heartbeat-monitor.py' to '/root/bin/heartbeat-monitor.py'")
    print(f"3. Copy '{SRC_DIR}/logger.py' to '/usr/local/bin/volition-logger.py'")
    print()
    print("If logger/heartbeat run on the same machine as Redis, you may change:")
    print("  REDIS_HOST=127.0.0.1 in the service files")
    print()
    print(f"4. Create Python environment:")
    print(f"   {CYAN}python3 -m venv /opt/volition/venv{RESET}")
    print(f"   {CYAN}/opt/volition/venv/bin/pip install aiohttp redis requests{RESET}")
    print()
    print(f"5. Enable and start services:")
    print(f"   {CYAN}systemctl daemon-reload && systemctl enable --now "
          f"volition-heartbeat volition-logger{RESET}")
    print()
    print(f"6. Verify:")
    print(f"   {CYAN}systemctl status volition-heartbeat volition-logger{RESET}")
    print("-" * 40)

    print(f"{RED}IMPORTANT REDIS CONFIG:{RESET}")
    print(f"Ensure Redis at {redis_host} is listening on the LAN interface.")
    print("Edit /etc/redis/redis.conf:")
    print(f"  {CYAN}bind 0.0.0.0{RESET}")
    print(f"  {CYAN}requirepass volition{RESET}")
    print(f"  {CYAN}protected-mode no{RESET}")
    print(f"Restart Redis:")
    print(f"  {CYAN}systemctl restart redis-server{RESET}")
    print(f"Verify:")
    print(f"  {CYAN}systemctl status redis-server{RESET}")
    print("-" * 40)

    print("Genesis does NOT install these services automatically because they often run on "
          "different machines.\n")

    print(f"{YELLOW}--- PART 2: REQUIRED COMPUTE SERVICES ---{RESET}")
    print()
    print("Volition REQUIRES the following services to exist on your network:\n")

    print("1. volition-gpu-worker.service")
    print("   - Handles embeddings (Tier-3 memory) and optional local summarization")
    print("   - Requires either:")
    print("       * Ollama (local GPU), OR")
    print("       * OpenRouter (cloud)")
    print("   - Default configuration assumes Ollama")
    print("   - Required Ollama models if used:")
    print("       * nomic-embed-text")
    print("       * mistral:latest")
    print("   - WARNING: Changing embedding model later requires full re-embedding\n")

    print("2. volition-ear.service")
    print("   - Handles social awareness and orientation")
    print("   - Without this, Abes wake up context-blind\n")

    print("These services MAY run on:")
    print(" - The Redis host")
    print(" - A GPU workstation")
    print(" - Any reachable Linux machine\n")

    print("They MUST:")
    print(f" - Reach Redis at {redis_host}:{redis_port}")
    print(" - Have Python + dependencies installed")
    print(" - Be enabled and running via systemd\n")

    print("Deployment steps for compute node:")
    print("1. Copy service files to /etc/systemd/system/:")
    print("   - volition-gpu-worker.service")
    print("   - volition-ear.service")
    print()
    print(f"2. Copy code files:")
    print(f"   - {SRC_DIR}/gpu-worker.py  ->  /opt/volition/gpu-worker.py")
    print(f"   - {SRC_DIR}/ear.py         ->  /opt/volition/ear.py")
    print()
    print(f"3. Create Python environment:")
    print(f"   {CYAN}python3 -m venv /opt/volition/venv{RESET}")
    print(f"   {CYAN}/opt/volition/venv/bin/pip install aiohttp redis requests{RESET}")
    print()
    print(f"4. Enable and start services:")
    print(f"   {CYAN}systemctl daemon-reload && systemctl enable --now "
          f"volition-gpu-worker volition-ear{RESET}")
    print()
    print(f"5. Verify:")
    print(f"   {CYAN}systemctl status volition-gpu-worker volition-ear{RESET}")
    print()
    print(f"{RED}This is NOT optional.{RESET}")
    print("Genesis will continue, but GUPPI will be cognitively degraded until these "
          "services are running.\n")

    input(f"{GREEN}Press ENTER once infrastructure services are deployed.{RESET}")


    # --- 2. PERSONALITY INJECTION ---
    print(f"\n{CYAN}--- PHASE 2: PERSONALITY SYNTHESIS ---{RESET}")
    # Verify critical files exist
    profile_src = DOCS_DIR / "98-source_profile.md"
    services_src = DOCS_DIR / "99-current_services.md"
    
    if not profile_src.exists() or not services_src.exists():
        print(f"{RED}CRITICAL: Missing 98-source_profile.md or 99-current_services.md in docs/{RESET}")
        print("Cannot proceed without personality cores.")
        sys.exit(1)

    # Prompt for edits
    if prompt("Do you want to edit the Source Profile (98) now?", "y").lower() == 'y':
        edit_file_pause(profile_src)
    
    if prompt("Do you want to edit the Service Map (99) now?", "y").lower() == 'y':
        edit_file_pause(services_src)


    # --- 3. CONTAINER SPAWNING ---
    print(f"\n{CYAN}--- PHASE 3: BIOLOGICAL FORMATION (LXC) ---{RESET}")
    vmid = prompt("LXC VMID", "9000")
    
    # Check if exists
    existing = run_cmd(["pct", "list"])
    existing_ids = {line.split()[0] for line in existing.splitlines() if line.strip()}
    if str(vmid) in existing_ids:
      print(f"{RED}VMID {vmid} already exists.{RESET}")
      print("Existing containers:")
      print(existing)

      suggested = max(int(i) for i in existing_ids if i.isdigit()) + 1

      print(f"\nSuggested next free VMID: {GREEN}{suggested}{RESET}")
      choice = prompt("Enter a different VMID, or press ENTER to use suggested", str(suggested))

      vmid = choice
      if vmid in existing_ids:
          print(f"{RED}VMID {vmid} is also in use. Aborting.{RESET}")
          sys.exit(1)

    template = find_latest_debian_template()
    ensure_debian_template(template)

    print(f"{YELLOW}Creating Container...{RESET}")
    try:
        run_cmd([
            "pct", "create", vmid, f"/var/lib/vz/template/cache/{template}",
            "--hostname", agent_name,
            "--storage", storage_id,
            "--net0", f"name=eth0,bridge={bridge_id},ip=dhcp",
            "--features", "nesting=1,keyctl=1",
            "--unprivileged", "1",
            "--start", "1",
            "--onboot", "1"
        ])
    except Exception as e:
        print(f"{RED}Container creation failed.{RESET}")
        print("Common causes:")
        print(f" - Storage '{storage_id}' does not exist")
        print(f" - Network bridge '{bridge_id}' does not exist")
        print(" - Insufficient permissions")
        print(" - Proxmox cluster in degraded state")
        sys.exit(1)

    print(f"{GREEN}Container {vmid} Started. Waiting 10s for networking...{RESET}")
    time.sleep(10)


    


    # --- 4. DATA INJECTION ---
    print(f"\n{CYAN}--- PHASE 4: NEURAL MAPPING (File Injection) ---{RESET}")
    
    # 4.1 Create Directory Structure
    # 4.1 Create Directory Structure
    dirs = [
        "/root/bin",
        "/root/docs",
        "/root/logs",
        "/root/src",
        "/root/.ssh",

        # Memory root + substructure
        "/root/memory",
        "/root/memory/episodes",
        "/root/memory/tier_1_archive",
        "/root/memory/overflow",
        "/root/memory/downloads",
        "/root/memory/vector.db",
    ]

    for d in dirs:
        run_cmd(["pct", "exec", vmid, "--", "mkdir", "-p", d])
    
    # 4.1.1 Create Required Base Files
    base_files = {
        "/root/working.log": "",
        "/root/communications.log": "",
        "/root/todo.db": "",  # sqlite will initialize schema on first open
        f"/root/.abe-clipboard-{agent_name}.md": "",
    }

    for path, content in base_files.items():
        run_cmd([
            "pct", "exec", vmid, "--",
            "bash", "-c", f"test -f {path} || printf '%s' {json.dumps(content)} > {path}"
        ])



    # 4.2 Push SRC -> BIN
    print(f"Injecting Tools ({SRC_DIR} -> /root/bin)...")
    for item in SRC_DIR.glob("*"):
        if item.is_file():
            if item.name == "guppi.service": continue 
            # Note: We push spawn_abe_lxc.sh to the container too, just for reference/backup, 
            # even though the active one lives on the host.
            run_cmd(["pct", "push", vmid, str(item), f"/root/bin/{item.name}"])

    # 4.3 Push Docs
    print(f"Injecting Cortex ({DOCS_DIR} -> /root/docs)...")
    # --- Constitution Templating (STRICT, SINGLE-PASS) ---
    genesis_doc = DOCS_DIR / "0.0-Abe-Genesis_Prompt.md"
    templated_genesis_tmp = None

    if genesis_doc.exists():
        text = genesis_doc.read_text()

        text = text.replace("{{ user.human_name }}", human_name)
        text = text.replace("{{ user.contact_method }}", contact_method)
        text = text.replace("{{ system.host }}", host_hostname)

        templated_genesis_tmp = Path("/tmp/0.0-Abe-Genesis_Prompt.md")
        templated_genesis_tmp.write_text(text)

    for doc in DOCS_DIR.glob("*.md"):
      if doc.name == "0.0-Abe-Genesis_Prompt.md" and templated_genesis_tmp:
          run_cmd([
              "pct", "push", vmid,
              str(templated_genesis_tmp),
              f"/root/docs/{doc.name}"
          ])
      else:
          run_cmd(["pct", "push", vmid, str(doc), f"/root/docs/{doc.name}"])
    
    if templated_genesis_tmp and templated_genesis_tmp.exists():
      templated_genesis_tmp.unlink()

    
    current_docs = DOCS_DIR / "current"
    if current_docs.exists():
        for doc in current_docs.glob("*.md"):
             run_cmd(["pct", "push", vmid, str(doc), f"/root/docs/{doc.name}"])

    # 4.4 Push Service & Env
    print("Configuring System Service...")
    
    # Write temp .env
    with open("/tmp/volition.env", "w") as f:
        f.write(env_content)
    run_cmd(["pct", "push", vmid, "/tmp/volition.env", "/root/bin/.env"])
    os.remove("/tmp/volition.env")

    # Read guppi.service, inject EnvironmentFile, and push
    guppi_svc_path = SRC_DIR / "guppi.service"
    if guppi_svc_path.exists():
        svc_content = guppi_svc_path.read_text()
        if "EnvironmentFile=" not in svc_content:
            svc_content = svc_content.replace(
                "[Service]", 
                "[Service]\nEnvironmentFile=/root/bin/.env"
            )
        
        with open("/tmp/guppi.service", "w") as f:
            f.write(svc_content)
            
        run_cmd(["pct", "push", vmid, "/tmp/guppi.service", "/etc/systemd/system/guppi.service"])
        os.remove("/tmp/guppi.service")

    else:
        print(f"{YELLOW}Warning: guppi.service not found in src/{RESET}")

    # 4.5 Identity
    print("Forging Identity...")
    id_json = generate_identity_json(agent_name, human_name)
    with open("/tmp/.abe-identity", "w") as f:
        f.write(id_json)
    run_cmd(["pct", "push", vmid, "/tmp/.abe-identity", "/root/.abe-identity"])
    os.remove("/tmp/.abe-identity")
    
    # 4.6 Host Tools Installation (The Spawner)
    print(f"Installing Host Tools ({HOST_BIN_DIR})...")
    if not HOST_BIN_DIR.exists():
        HOST_BIN_DIR.mkdir(parents=True, exist_ok=True)
        
    spawn_src = SRC_DIR / "spawn_abe_lxc.sh"

    if spawn_src.exists():
        # Read the script
        script_content = spawn_src.read_text()
        
        # Inject the user's configuration at the top
        # We replace the hardcoded STORAGE="local" line with the user's choice
        # and ensure the BRIDGE is set correctly.
        config_block = f"""
# --- CONFIG INJECTED BY GENESIS ---
STORAGE="{storage_id}"
BRIDGE="{bridge_id}"
# ----------------------------------
"""
        # Simple string replacement or prepend
        # Since the original script has `STORAGE="local"`, we try to replace that line.
        if 'STORAGE="local"' in script_content:
             script_content = script_content.replace('STORAGE="local"', f'STORAGE="{storage_id}"')
        else:
             # Just prepend if we can't find the exact line
             script_content = config_block + script_content

        # Write to Host Bin
        target_path = HOST_BIN_DIR / "spawn_abe_lxc.sh"
        with open(target_path, "w") as f:
            f.write(script_content)
        
        os.chmod(target_path, 0o755)
        print(f"{GREEN}Installed and Configured spawn_abe_lxc.sh to {target_path}{RESET}")
      
     # --- 5. SSH CONFIGURATION ---
    print(f"\n{CYAN}--- PHASE 5: THE UMBILICAL CORD (SSH) ---{RESET}")
    print("Generating SSH identity for Abe-01...")
    
    # Generate Key in Container
    run_cmd(["pct", "exec", vmid, "--", "ssh-keygen", "-t", "ed25519", "-f", "/root/.ssh/id_ed25519", "-N", ""])
    
    # Read Pubkey
    pubkey = run_cmd(["pct", "exec", vmid, "--", "cat", "/root/.ssh/id_ed25519.pub"])
    
    print(f"\n{YELLOW}Abe-01 Public Key:{RESET}")
    print(pubkey)
    print(f"\nTo spawn new agents, Abe-01 needs SSH access to this host ({host_ip}).")
    if prompt("Authorize Abe-01 to SSH into this Proxmox Host?", "y").lower() == 'y':
        with open(HOST_SSH_KEYS, "a") as f:
            f.write(f"\n# Volition: Abe-01 Genesis Key\n{pubkey}\n")
        print(f"{GREEN}Authorized.{RESET}")
    else:
        print(f"{RED}Skipped. Spawning will fail until you authorize this key manually.{RESET}")


       # Create .ssh/config in Container
    ssh_config = f"""
# ==============================
# Volition SSH Control Surface
# ==============================
# This file defines ALL machines Abes are allowed to control.
# Host aliases MUST match names used in 99-current_services.md
# ==============================

# Global Defaults
Host *
    IdentityFile /root/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
    BatchMode yes
    ConnectTimeout 10
    ServerAliveInterval 30
    ServerAliveCountMax 3



# The Proxmox Host (Parent Node)
Host parent_node {host_hostname}
    HostName {host_ip}
    User root
"""
    with open("/tmp/ssh_config", "w") as f: f.write(ssh_config)
    run_cmd(["pct", "push", vmid, "/tmp/ssh_config", "/root/.ssh/config"])
    run_cmd(["pct", "exec", vmid, "--", "chmod", "700", "/root/.ssh"])
    run_cmd(["pct", "exec", vmid, "--", "chmod", "600", "/root/.ssh/config"])
    run_cmd(["pct", "exec", vmid, "--", "chmod", "600", "/root/.ssh/id_ed25519"])
    run_cmd(["pct", "exec", vmid, "--", "chmod", "644", "/root/.ssh/id_ed25519.pub"])

    os.remove("/tmp/ssh_config")

    print(f"""
    {YELLOW}--- SSH CONTROL BOUNDARY DECLARATION REQUIRED ---{RESET}

    The file /root/.ssh/config inside the Abe container now defines
    EVERY machine this Abe (and its children) may control.

    You MUST do the following before continuing(in a new session):

    1. Enter the container:
      {CYAN}pct enter {vmid}{RESET}

    2. Edit the SSH config:
      {CYAN}nano /root/.ssh/config{RESET}

    3. For EACH host listed in docs/99-current_services.md:
      - Add a matching 'Host <name>' entry
      - Ensure HostName matches the real IP/DNS
      - Ensure User is correct (root recommended)
      - The parent-node entry is already done for you. You can use that as a reference.

    4. On EACH target machine:
      - Install Abe's SSH public key for that user you defined.
      - Example: To authorize for root on the host, append the following to /root/.ssh/authorized_keys:
        {CYAN}{pubkey}{RESET}
      - If NOT using root, grant passwordless sudo at : {CYAN}/etc/sudoers.d/username{RESET}
        Example sudoers line:
        {CYAN}<user> ALL=(ALL) NOPASSWD: ALL{RESET}

    This is a SECURITY BOUNDARY.
    Genesis WILL NOT proceed until you confirm this is done. You MUST be comfortable with the security implications.
        {YELLOW}Note: The Abes will hallucinate/be lobotomized without proper SSH access to the host and services.{RESET}
    """)

    input(f"{GREEN}Press ENTER ONLY after SSH access has been verified manually.{RESET}")




    # --- 5. BOOTSTRAP ---
    print(f"\n{CYAN}--- PHASE 5: AWAKENING (Bootstrap) ---{RESET}")
    
    bootstrap_script = """#!/bin/bash
set -e
export LANG=C
export LC_ALL=C
export DEBIAN_FRONTEND=noninteractive

echo "[*] Updating Apt..."
apt-get update -qq

echo "[*] Installing Dependencies..."
apt-get install -y -qq python3-full python3-pip python3-venv git redis-tools curl nano

echo "[*] Creating Venv..."
if [ ! -d "/root/venv" ]; then
    python3 -m venv /root/venv
fi

echo "[*] Installing Python Libs..."
/root/venv/bin/pip install --upgrade pip
# Core Volition Requirements
/root/venv/bin/pip install aiosqlite redis asyncssh aiohttp requests chromadb google-genai trafilatura

echo "[*] Enabling Service..."
systemctl daemon-reload
systemctl enable guppi.service

echo "[*] Setting Permissions..."
chmod +x /root/bin/*.py
chmod +x /root/bin/*.sh 2>/dev/null || true

echo "[SUCCESS] Bootstrap Complete."
"""
    with open("/tmp/bootstrap_volition.sh", "w") as f:
        f.write(bootstrap_script)
    
    run_cmd(["pct", "push", vmid, "/tmp/bootstrap_volition.sh", "/root/bootstrap_volition.sh"])
    
    print(f"{YELLOW}Running internal bootstrap (this may take a minute)...{RESET}")
    print(run_cmd(["pct", "exec", vmid, "--", "bash", "/root/bootstrap_volition.sh"]))
    
    os.remove("/tmp/bootstrap_volition.sh")

    # --- Redis Sanity Check (from inside container, after redis-tools is installed) ---
    print(f"{CYAN}--- REDIS SANITY CHECK (Container -> Redis) ---{RESET}")
    try:
        out = run_cmd([
            "pct", "exec", vmid, "--",
            "redis-cli",
            "-h", redis_host,
            "-p", str(redis_port),
            "-a", redis_pass,
            "PING"
        ])
        if out.strip() != "PONG":
            raise RuntimeError(out)
        print(f"{GREEN}[OK] Container can reach Redis.{RESET}")
    except Exception as e:
        print(f"{RED}[FAIL] Container cannot reach Redis: {e}{RESET}")
        print("Fix Redis networking/auth before continuing.")
        sys.exit(1)

    print(f"\n{GREEN}=== GENESIS COMPLETE ==={RESET}")
    print(f"Container: {agent_name} ({vmid})")
    print(f"Identity:  {agent_name} (Child of {human_name})")

    print("\nNext Steps (Operator Actions):")
    print(f"1. Enter the container:")
    print(f"   {CYAN}pct enter {vmid}{RESET}")

    print("2. Verify core agent is alive:")
    print(f"   {CYAN}systemctl status guppi{RESET}")

    print(f"\n{CYAN}Verification checklist inside container:{RESET}")
    print("  /root/todo.db")
    print("  /root/working.log")
    print("  /root/communications.log")
    print(f"  /root/.abe-clipboard-{agent_name}.md")
    print("  /root/memory/{episodes,tier_1_archive,overflow,downloads,vector.db}")


    print("\n3. Deploy the Volition Dashboard (Human Control Plane):")
    print("   The dashboard runs OUTSIDE Abe containers.")
    print("   It requires direct Redis access and Python.")

    print("\n   a) Choose a machine to run it on:")
    print("      - Proxmox host OR")
    print("      - Any Linux machine that can reach Redis")

    print("\n   b) Copy dashboard files:")
    print(f"      {CYAN}mkdir -p /opt/volition/dashboard{RESET}")
    print(f"      {CYAN}cp -r dashboard/* /opt/volition/dashboard/{RESET}")

    print("\n   c) Create a Python environment:")
    print(f"      {CYAN}python3 -m venv /opt/volition/venv{RESET}")
    print(f"      {CYAN}/opt/volition/venv/bin/pip install fastapi uvicorn redis aiohttp jinja2{RESET}")

    print("\n   d) Install the systemd service:")
    print(f"      {CYAN}cp dashboard/volition-dashboard.service /etc/systemd/system/{RESET}")

    print("\n   e) Edit the service file if needed:")
    print("      - Verify paths:")
    print("        ExecStart=/opt/volition/venv/bin/python /opt/volition/dashboard/volition_dashboard.py")
    print("      - Verify Redis env vars match your setup")

    print("\n   f) Enable and start:")
    print(f"      {CYAN}systemctl daemon-reload && systemctl enable --now volition-dashboard{RESET}")

    print("\n   g) Access the UI (Make sure to change your name on top left):")
    print(f"      {CYAN}http://<dashboard-host>:8000{RESET}")


    print("\n4. Start the Abe:")
    print(f"   {CYAN}systemctl start guppi{RESET}")

    print("\nNotes:")
    print("- The dashboard is optional but strongly recommended.")
    print("- Abes will still function headless without it.")
    print("- Alerts via ntfy are your primary safety net.")
    print("")
    print("NOTE: Changing embedding models later requires deleting vector.db and re-embedding memory.")



if __name__ == "__main__":
    main()