##Volition Architecture Documentation

This directory contains the specifications for the Volition Agent System.

### Current Architecture (v7.0+)

The active system specification. Reflects the codebase in /src. Note: Almost all of these files are mandatory for the system to function correctly. You should take some time to map out your homelab in at least `99-current_services.md` before deploying agents.

- 0.0-Abe-Genesis_Prompt.md: The complete system prompt injected by GUPPI on every Think cycle. Defines agent identity, split-brain modes, memory architecture, tool catalog, social protocols, and output contracts.

- 98-source_profile.md: The human user's identity document. Template covering personal background, professional context, technical environment, personality traits, communication style, goals, and preferences that agents read to understand their Source. This will be 'interpreted' by agents into their `.abe-priors.md` file on their first boot.

- 99-current_services.md: Complete infrastructure inventory. Hardware specs, network topology, VMs, containers, and services across all hosts. Single source of truth for the homelab's physical and logical architecture. This file is **mandatory**. Without this, the Abes will be lobotomized.

- Volition-1-Philosophy.md: The "Why". Split-brain theory and agent rights.

- Volition-2-Core-Architecture.md: The "What". Component definitions (Abe, GUPPI, Scribe).

- Volition-3-GUPPI-Loop.md: The "How". The async event loop and lifecycle.

- Volition-4-Memory-RAG.md: Memory tiers and retrieval logic.

- Volition-7-Log-Schemas.md: The strict JSON contracts for logs.

- Volition-8-Governance-Human-Interface.md: The human-agent interface and governance.

### The Archive (Legacy & Evolution)

Historical documents preserving the evolution of the system. Useful for understanding why certain architectural decisions (like the move to structured logging) were made.

- Comparison 7.0 vs 5.0: An analysis of the shift from "Biological" (v5) to "Industrial" (v7) agent design.

- Legacy Specs: The original "High Trust" architecture specs. This is what I envisioned in [my first blog post](https://aindoria.com/posts/bobiverse_in_my_homelab/)