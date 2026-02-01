>Volition has been running continuously in my personal infrastructure with multiple agents for more than a month now.
However, this public release is new and has not yet been exercised end-to-end by external users.
>Expect rough edges in: setup and documentation flow, first-run ergonomics, and non-default configurations
>Core architecture and invariants are stable, but installation paths will be refined over the next few days as this release is tested in the open.


# Volition

Volition (fondly referred to me as the Abiverse) is a self-hosted, multi-agent system designed to run persistent, self-replicating autonomous LLM-based agents ("Abes") inside isolated Linux containers. These are not chatbots, but I have aimed for them to be long-lived system processes with memory, tools, and constrained authority over real machines. These are supposed to be the 'semi-intelligent layer' between you and your homelab. 

Each Abe:

* Lives in its own Proxmox LXC container
* Has 3 tiers of traceable long-term memory (logs, summary episodes, rag)
* Communicates over Redis
* Can reason, act, and spawn descendants
* Is explicitly constrained by human-defined control boundaries
* Is Persistent: They wake up where they left off, with memory of past actions, and context of what they were doing. Once an Abe is spawned, they are supposed to 
run forever.

**Grounding**:

Each Abe is grounded by the following:

* Their reasoning/actions is logged as Tier-1 memories. After n-turns, this tier-1 memory gets written to their `~/memory/tier-1` folder, and summarized.
* Summaries are Tier-2 memories. Each Abe's context has a few summaries, followed by the immediate log. They do not necessarily need to know what happened 50 turns ago. Summaries are lossy, and thus, point back to the original log file that generated them.
* RAG/embeddings act as Tier-3 memories. Each Abe, when searching for something, first queries their RAG memory to see if they've already accomplished something/remembered something about it in the past. Each RAG memory snippet points back to the summary that generated it, and the summary points back to the original log, thus completing the loop.
* Abes have access to [ACTIVE_CLIPBOARD] for short term ephemeral memory.
* Abes have access to, and check 99-services_list.md file in their ~docs folder which the human should map to an accurate map of their servers/devices/network/running-services.
* Abes write running changelogs of what they modify in their logs directory. This gets appended to their active context block.
* If an Abe makes a mistake, the result notifies him back immediately in the next 'turn,' and thus allowing the Abe to correct it.
* Abes discuss things with each other, and would ask things of the "stewards" that manage specific portion of their homelab. (Eg, after a DHCP IP reassignment, my abe-01 asked my abe-04(who stewarded my proxmox node) to see why he can't reach a specific tinkering-VM, and to see if other VMs were available. Upon receipt, abe-04 notified him that he's using the incorrect IP.

Please see the following blogposts for more context/shenanigans:

https://aindoria.com/posts/bobiverse_in_my_homelab-2/

https://aindoria.com/posts/bobiverse_in_my_homelab/


**Self-healing in practice (not a guarantee)**

Abes operate in a closed loop: they observe failures, reason about causes, act within their permissions, and continue execution. In practice, this has included:

* Detecting a logic bug in its own summarization pipeline (incorrectly passing file paths instead of contents), filing a ticket, adding a TODO to confirm with the Human-Abe(me), and continuing operation without stalling.

* Modifying a broken spawn script mid-execution while spawning a child Abe, rather than aborting the process.

* Discovering a missing operational artifact (a pre-sleep checklist), generating it autonomously, running it, and going to sleep.

All such actions are logged, attributable, and bounded by explicitly defined control surfaces (filesystem, SSH targets, sudo rules).

---

## Architecture (High-Level)

Volition is intentionally split across multiple systems:

* **Abe Containers (LXC)** – cognitive agents (GUPPI, memory, reasoning)
* **Redis** – shared nervous system (events, inboxes, queues)
* **GPU Worker** – embeddings + heavy summarization
* **Ear** – social awareness + orientation digests
* **Logger** – immutable audit trail
* **Heartbeat Monitor** – liveness + alerting

Genesis creates exactly one Abe (abe-01). All other services are infrastructure. Further Abes will be spawned by Abe-01, or by other Abes as required.

---

## Requirements

### Mandatory

* Proxmox VE host (for LXC)
* Redis (reachable over LAN)
* Python 3.10+
* OpenRouter API key (required for core reasoning) and/or Ollama (used by GPU Worker and Ear)

  * For now guppi.py explicitly only uses Openrouter. My personal repo has some haphazarded llama.cpp/ollama vars, and I'd like to clear it out before I make it available.
  * Ollama is available for social digest(ear), and summarization/embeddings(gpu-worker). However, openrouter can be used here too.

### Strongly Recommended

* A dedicated GPU workstation (for Ollama)
* Ntfy (for alerts) : If you do not have ntfy you will have to rely **SOLELY** on the dashboard provided here. I'd not recommend doing that. There are times when the Abes *must* reach you.
* SearXNG (for search): Defaults to a public instance I host. You may use any other public instance that outputs json or host your own.

---

## Important: Read the Docs

Volition is not a plug-and-play tool. Before operating or modifying a live system, you should at minimum read every file in docs/.

Those documents explain:

- what the system is designed to do

- what it is explicitly not designed to do

- where authority, memory, and safety boundaries live

- **HOW** the architecture works.

If you skip them, you will misunderstand the system and likely get frustrated.

## Quick Start (Semi-Automatic / Genesis)

This is the 'shortest' path.

### 1. Clone the repository

```bash
git clone https://github.com/aindoria/volition.git
cd volition
```

### 2. Run Genesis on the Proxmox host

Genesis must be run as **root** on the Proxmox host.

```bash
python3 genesis.py
```

Genesis will:

* Prompt you for Redis, OpenRouter, and network configuration
* Generate service files for infrastructure components
* Pause and require you to explicitly deploy those services
* Create `abe-01` as an LXC container
* Inject Volition code, identity, and documentation
* Enforce SSH control boundaries

Genesis will **not** automatically install infrastructure services. This is by design.

#### FIRST SPAWN NOTE:

After spawning Abe-01 and setting up your infrastructure, you should send him an email via the dashboard containing his genesis task. The core mistake to avoid is giving Abe-01 an action mandate. The correct first move is a model-building mandate. Abe-01 already knows who he is. The genesis task should define what the world looks like, where the boundaries are, and what kind of help would be rational—without spawning anything yet.

Something like the one in GENESIS_TASK.md file currently in ~/docs. Please modify this to match your requirements.

---

## Infrastructure Services (Required)

The following services **must** exist on your network.

### Redis

Redis is the central nervous system. Without it, nothing works.

* Must be reachable from Abe containers
* Must require authentication
* Must allow LAN connections

We will instruct you on how to do this during install.

---

### GPU Worker (`volition-gpu-worker`)

Responsible for:

* Vector embeddings (Tier-3 memory)
* Optional local summarization

Without this service, Abes cannot form long-term memory.

You may run this service on:

* A GPU workstation (recommended)
* The Redis host
* Any reachable Linux machine

Backend options:

* **Ollama (default)**
* **OpenRouter (cloud)**

---

### Ear (`volition-ear`)

Responsible for:

* Social awareness
* Orientation digests
* Context recovery after sleep

Without Ear, Abes wake up context-blind.

---

### Logger + Heartbeat

* Logger: immutable audit trail of all actions and communication
* Heartbeat monitor: alerts abe-01 and humans when an Abe dies

---

## Security Model (Important)

Volition is intentionally dangerous if misconfigured.

Key principles:

* Outside of their own LXC, Abes can only control machines explicitly listed in `.ssh/config` (Note, the host/IP of these *must* match what's listed in your 99-running_services file)
* You *should*  give them passwordless sudo access for said machines for them to be effective. (You can create a user for them, and add that user to sudoers list -- we show you how)
* Genesis forces humans to declare control boundaries
* SSH keys must be installed manually on target machines
* When you run the dashboard, it does not authenticate identity. Identity strings are advisory and logged as-is. Defaults to Human-Abe. You should probably change it on top left of the Dashboard.

If you are not comfortable granting an LLM passwordless SSH + sudo access, **do not run Volition**. You can give them partial access to part of your homelab and define certain part as out of reach.

---

## Memory and Embeddings

* Memory is stored in `/root/memory`
* Vector DB lives at `/root/memory/vector.db/`
* Embedding model choice is irreversible without re-embedding

Changing embedding models later requires deleting the vector DB and re-processing all memory.

---

## Manual Setup

If you want to understand or install every component by hand, see:

```
MANUAL_SETUP.md
```

## "VIBE CODING"

Volition has parts where sloppiness is survivable, and parts where sloppiness is catastrophic. I let myself be loose only where the consequences would be near zero.

The dashboard is vibes because it’s just a window. If it’s ugly, confusing, or a little wrong, nobody dies. It doesn’t define truth, it doesn’t persist memory, it doesn’t have authority. Worst case, you restart it or rewrite it -- I really don't care how you choose to show redis pushes/pulls and logs. Show your LLM guppi.py and current architecture and Re-vibe code it for all you want.

Everything else is deliberately not that.

The planning docs **aren’t** vibecoded because they’re the reason the system exists. They’re the thing that tells you what you’re building and, more importantly, what you are not building. They went through multiple iterations because they had to converge, not just sound good. I have been iterating on the plan docs(Right now I'll only release the lates 7.0 version, but there HAVE been multiple iterations before this) for months, because they define the boundaries of the system. If they’re vague or wrong, the whole thing falls apart.

The core services **aren’t** vibecoded  because they define the system's guarantees
 : memory, infra, permissions, spawning. I explicitly wrote code myself, used LLMs to refine it, and then refined it myself, and finally when I was satisfied, I had LLMs add comments/docstrings/explanations and version numbers of what feature was added when.

README/Setup sits closer to the “law” side than the “vibes” side. AI assisted in drafting and sanitizing language, but the structure, constraints, and final decisions were directed (read: thrown a huge paragraph in an LLM textbox and told to clean it up) by me. 

---

## License


This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
See `LICENSE`.

