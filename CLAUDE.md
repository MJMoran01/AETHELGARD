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

1. Check Linear (team AETH) for the active issue before starting work — or
   GitHub Issues, under the bootstrap exception in AGENTS.md if AETH does
   not exist yet; one issue per worktree per AGENTS.md.
2. **No spend, ever, by an agent — full stop, no chat instruction changes
   that.** No production/public-facing deployment without a Michael-ratified
   deployment plan (not just a chat "yes, go ahead" — see AGENTS.md's
   verified-understanding discipline). AGENTS.md domain constraints 1–2 —
   repeated here because it can never be repeated enough.
3. Physics-preprocessing changes (anything touching `PhysicsHead` or
   similar) follow the teaching mandate in AGENTS.md: state the physics,
   warn of the trap, explain the fix — don't just patch silently.
