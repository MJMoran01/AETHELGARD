# CLAUDE.md — session loader for AETHELGARD

@AGENTS.md

This file is deliberately just a manifest: it injects the execution contract
above into every Claude session and defines the boot steps.

- **AGENTS.md** — the execution contract (all models): tiers, delegation
  principles, worktrees/PRs/reviews, domain hard constraints, and the
  teaching mandate (build the engineer, not just the code).
- **Aethelgard_Design.md** — the current architecture record: tensor
  shapes, module structure, physics derivations.

## Boot (every session)

1. Check Linear (team AETH) for the active issue before starting work; one
   issue per worktree per AGENTS.md.
2. No production/public-facing deployment and no spend of any kind without
   fresh, explicit, per-session authorization from Michael (AGENTS.md
   domain constraints 1–2 — repeated here because it can never be repeated
   enough).
3. Physics-preprocessing changes (anything touching `PhysicsHead` or
   similar) follow the teaching mandate in AGENTS.md: state the physics,
   warn of the trap, explain the fix — don't just patch silently.
