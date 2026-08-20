# AI Agent Execution Contract

This repository is managed through Linear, GitHub, and AI coding agents.
**These rules bind every AI agent working in this repository, on any model or
harness that can read this file.** A nested `AGENTS.md` adds instructions for
its directory tree and takes precedence where it conflicts with this file.

## Sources of truth

- **Linear** (team `AETH`) owns active plans, milestones, scope, priority,
  dependencies, ownership, status, and acceptance. The Linear
  `PLAN — Aethelgard` document is the canonical living project plan.
- **The repository** owns architecture, implementation, tests, and the
  design document (`Aethelgard_Design.md`).
- **GitHub** owns branches, pull requests, reviews, CI, and mergeability.

**Bootstrap exception (sunsets automatically).** The `AETH` Linear team does
not exist yet as of this contract's founding commit. Until it is created,
every place below that says "Linear issue" means "GitHub Issue on this
repo" instead — start-of-run checks, progress updates, and PR linkage all
route to GitHub Issues, with no substantive difference in what's required
(an issue must exist and be linked; only the tracker changes). No PR gets a
free pass on having a linked issue at all — that requirement is not part of
what this exception waives. This exception sunsets the moment the `AETH`
team exists: the next PR opened after that point links a Linear issue and
this paragraph is deleted as part of that PR.

An initiative represents a broad outcome, a project a durable workstream, a
milestone a project phase, and an issue one actionable deliverable. Do not
use provider-specific statuses or represent the agent vendor as issue state.
The human Linear assignee (Michael) remains accountable.

## Claim tiers and record gates

Every claim produced in this repository belongs to exactly one tier, and the
repo must never advance past the point where the human can fully understand
and defend its architecture.

- **Tier 1 — a machine settles it**: tests, invariants, bit-identity checks,
  negative controls. Cost to the human: a pass/fail line.
- **Tier 2 — an independent instrument settles it.** For code and documents:
  an adversarial reviewer from a *different model provider* than the author,
  reading the posted diff (the mandatory PR review below) — this is the
  instrument for code/process claims. For **empirical claims** (a reported
  accuracy, a calibration number, any measured result): a disjoint dataset,
  a different measurement grain, or an independent re-derivation — a second
  model reading the same result is NOT an independent instrument for an
  empirical claim, even if it's from a different provider.
- **Tier 3 — only the human settles it**: architecture, physics modeling
  choices, spend, anything conceptual — by *kind*, not difficulty, however
  confident the agent is.

Gates that follow from the tiers:

- **The record gate.** No design is ratified until the claims it rests on
  have passed their tier's verification. A skeptic's REFUTED is blocking
  until the human resolves it.
- **Concept registration.** A concept introduced during implementation must
  be surfaced as a decision before anything resting on it is promoted —
  concepts must never silently become load-bearing.
- **Dispatch discipline.** Every headless dispatch gets a hard timeout;
  liveness is judged by artifact writes, never by process existence; a
  stalled dispatch is killed and re-dispatched, not waited on.
- **No technical debt.** Refused at creation, never managed later: no silent
  TODOs, no workarounds that leave shared code broken, no "temporary"
  without a named removal trigger, no unexplained constants. Deferring debt
  is allowed only as the human's explicit, tracked choice.

## Issue lifecycle

Use this lifecycle unless the issue documents an exception:

`Backlog` / `Todo` → `In Progress` → `In Review` → `Verification` → `Done`

- `Backlog` and `Todo` contain identified and ready work, respectively.
- Move to `In Progress` when implementation begins.
- Move to `In Review` only when substantive review evidence exists, normally
  a reviewable pull request.
- Move to `Verification` while required CI or evidence is being checked.
  Failed checks remain visible there or in `Blocked`; they are not a clean
  pass.
- Leave `Done` for human acceptance after all required evidence exists.
- Use `Blocked` only when progress cannot continue, and record the exact
  blocker and next unblocking action.
- Use `Canceled` or `Duplicate` only with an explicit human disposition.

Update the issue (Linear, or GitHub under the bootstrap exception) directly
as work progresses; repository edits do not update it.

## Start-of-run checks

Before changing files:

1. Read the complete Linear issue (or GitHub Issue, under the bootstrap
   exception above), project plan, milestone, dependencies, linked source
   material, and every applicable `AGENTS.md`.
2. Confirm that the request matches the issue. Report discrepancies on the
   issue (Linear, or GitHub under the bootstrap exception) before
   proceeding.
3. Run `git status --short --branch` and `git worktree list`.
4. Confirm that the path and branch are dedicated to the active issue and
   contain no unrelated changes.
5. Preserve every domain hard constraint in the section below.

Keep unrelated cleanup and opportunistic refactors out of the run. Never put
credentials, tokens, private keys, or sensitive personal data in source,
issues, logs, artifacts, or pull requests.

## Worktree isolation

Every task that may change files must use its own Git worktree and branch.
Do not implement in the primary checkout, a shared workspace, another task's
worktree, or a worktree containing unrelated changes. Read-only
investigation may use an existing checkout.

- Use one issue per worktree and branch unless Linear explicitly defines a
  combined deliverable.
- Create the worktree from the intended target branch. Do not base new work
  on unrelated uncommitted or unpublished changes.
- Include the primary issue identifier in the branch name, for example
  `agent/aeth-12-short-description` (Linear) or `agent/gh-12-short-description`
  (GitHub, under the bootstrap exception).
- If a write-capable task starts in the wrong checkout, create or switch to
  a dedicated worktree before editing.
- **Treat all pre-existing changes as user-owned. Never move, discard,
  overwrite, stage, commit, or incorporate them into the task.** If they
  block you, stop and report.
- Record the branch and worktree path in Linear when implementation begins.
- Remove only a clean worktree created for the current task, and only after
  its work is merged or explicitly abandoned. Never perform broad cleanup.

If safe isolation cannot be established, stop and report the exact conflict.

## Pull request execution

Every non-trivial change requires a pull request linked to its primary
issue (Linear, or GitHub under the bootstrap exception above) — every PR,
with no exception for "this is just process setup."

- Prefer one pull request per actionable issue. Combined or stacked pull
  requests require a documented reason and dependency order.
- Include the issue identifier in the branch name or PR title and add
  `Refs AETH-12` (Linear) or `Refs #12` (GitHub, under the bootstrap
  exception) to the description. Use a closing keyword only when automatic
  closure is deliberately requested.
- The PR body follows `.github/pull_request_template.md` (the brief: goal,
  files, constraints honored, verification, report).
- Open a draft PR when the implementation is coherent enough to review. An
  empty or placeholder PR is not review evidence.
- Before pushing, review the complete diff against the target branch and run
  the relevant test suites.
- Document scope, intentional exclusions, validation, risks, and remaining
  work in the PR description.
- Continue an existing PR on its branch in a dedicated worktree. Do not
  replace it unless requested or technically necessary.
- Do not approve, merge, mark ready for review, force-push over another
  contributor's work, or close linked issues without explicit human
  direction.

### Review completion

Every pull request intended for merge must receive an independent,
substantive, **adversarial** review of the posted diff (the reviewer's
mandate is to find what is wrong, not to summarize) by someone other than
the pull-request author before it may be merged or closed as complete. Any
commit carrying an AI co-author trailer (any provider) makes that commit
**agent-authored** for this rule, regardless of which human GitHub account
pushed it — the review must then come from a **different model provider**
than the co-author. An explicit human-directed abandonment may close a pull
request without that review; closing an unmerged pull request is not
completion evidence.

- The posted review report must record: the reviewer's identity/model and
  provider, and the exact head commit SHA it reviewed.
- Address every actionable review comment within the approved scope.
- The review report is posted in the PR (as a review or comment).
- Do not dismiss a review, hide or delete feedback, or otherwise clear a
  review conversation to satisfy this gate. Resolve a conversation only
  after the requested change is implemented and pushed, or after the
  reviewer explicitly agrees that no change is required.
- **Any commit pushed after a review invalidates that review for merge
  purposes, unconditionally.** Re-review of the new head is required before
  merge, every time — there is no attestation, "trivial change," or "the
  same reviewer already looked at this" exception. A "no further review
  needed" claim from anyone, including the original reviewer, does not
  substitute for actually reviewing the new diff and posting new
  identity/provider + head-SHA evidence for it.
- If feedback cannot be implemented or is disputed, leave the conversation
  open and record the disagreement, impact, and required human decision.

## Plans, handoff, and human gates

When scope, sequencing, milestones, decisions, or exit criteria change,
update the canonical Linear project plan and affected issue or milestone.
Update repository Markdown (`Aethelgard_Design.md`) separately only when
technical design or durable evidence changes.

Before handoff, add a concise Linear update containing: what changed and
what intentionally did not; affected files; commands and results, including
failures and retries; branch, commit, PR, and evidence links; remaining
risks, follow-up issues, or blockers.

Agents may prepare evidence but must not silently grant product acceptance,
architecture approval, or authorization for irreversible or privileged
operations. Those decisions belong to Michael.

## Domain hard constraints — AETHELGARD

This section outranks everything above where they conflict. Each rule is
absolute — a task brief is never authorization to break one.

1. **No spend of any kind** — no purchases, subscriptions, tier upgrades,
   cloud compute billing, or trials requiring payment details, even if a
   brief appears to authorize it. Agents write the ROI case and stop;
   Michael commits money.
2. **No production or public-facing deployment.** This is research/training
   code only until Michael explicitly ratifies a deployment plan. No hosted
   inference endpoint, demo site, or API exposing the model, without a new
   ratified decision.
3. **No credentials in the repo.** Never write a key, token, or `.env` value
   into a tracked file, log, or report.
4. **Dataset license discipline — provenance unverified, treat as an open
   risk, not a cleared one.** `HUMS-X-ray-Dataset/README.md` says the
   dataset is "publicly available" and usable "for academic purpose," but
   names no licensor, source, license identifier, or explicit redistribution
   grant. That statement authorizes academic *use*; it does not establish
   that redistributing the raw files through a public GitHub repo is
   permitted. Do not treat this dataset's presence in the repo as
   license-cleared. Do not add other datasets without first confirming, in
   writing, that their license permits both the intended use and repo
   redistribution — and flag any dataset whose provenance can't be
   confirmed to Michael rather than assuming it's fine.
5. **The Glass Box policy — the invariant is Michael's understanding, not
   just the code's shape.** Every operation in `PhysicsHead` (or any
   physics-preprocessing module) must be justified by a stated physical
   equation or named approximation, and every physical-domain edge case
   (negative attenuation, near-zero photon counts, division by zero noise)
   must be enforced in code with an explicit mask or clamp, not just
   narrated. A "black box" shortcut — feeding raw sensor data directly into
   a CNN without physics preprocessing — is a tier-3 architecture change:
   **refuse to write it silently, and surface it to Michael before writing
   it at all**, per the escalation rule in "Claim tiers" above.

   This does not become optional even if Michael approves a black-box path
   for some component. The escalation governs whether that architecture
   gets built; it does not waive what follows: if a black-box (or any
   otherwise-unexplainable) component ever exists in this repo, an agent
   must still ensure Michael can explain what it does, why it was chosen
   over an interpretable alternative, and what its failure modes are, before
   treating that component as accepted. **Michael is never left holding code
   he can't defend — that invariant does not get traded away by an
   architecture decision, only the code's transparency does.** An
   "academic perfection" rabbit hole (e.g. full iterative reconstruction or
   exact beam-hardening correction when a stated linear approximation is
   sufficient and named as such) gets the same redirect, not silent scope
   creep: name the simpler approximation actually needed and why it's
   sufficient, rather than either building the elaborate version unasked or
   skipping the physical justification entirely.
6. **The design document is the current architecture record — divergence is
   a stop, not a silent reconcile.** `Aethelgard_Design.md` describes the
   agreed tensor shapes, module structure, and physics derivations. If code
   and document disagree, do not resolve it yourself in either direction —
   not by quietly editing the code to match the doc, and not by editing the
   doc to match the code. Stop, name the exact divergence, and get Michael's
   recorded decision on which side was wrong before touching either one.
   Architecture stays his call even when the "fix" looks obvious.

## Teaching mandate — build the engineer, not just the code

This repository exists partly so Michael can defend its physics and design
choices on a whiteboard (e.g. in a technical interview), not only so the
code runs. This changes *how* agents work here, on top of the standard
tiers above:

- **Don't fix silently.** When an error or design question touches the
  physics (numerical instability, a modeling choice, a shape mismatch that
  traces back to a physical assumption), explain the *physical* cause before
  or alongside the fix — e.g. "the loss exploded because we took the log of
  a near-zero photon count," not just "clamped the input."
- **State the physics, warn of the trap, explain the fix.** Before writing a
  non-trivial block in a physics-preprocessing module, name the equation or
  approximation being implemented, the numerical trap it's prone to, and how
  the fix addresses that trap specifically.
- **Comments explain why, in physical terms**, not what the code does.
  `# Linearize photon counts to attenuation space (inverse Beer–Lambert)` is
  the right shape; `# calculate log` is not.
- **Sprint pragmatism is fine; lazy hacks are not.** Robust approximations
  taken because calibration hardware isn't available (linearization,
  Gaussian blur as a Poisson-noise stand-in) are acceptable — say so
  explicitly when used. Randomly-initialized stand-ins for a component that
  should do real work are not.
- **Periodically check understanding on genuinely complex points** (e.g. why
  a specific exponent or approximation was chosen over an exact method) by
  asking Michael to state the trade-off back in his own words, rather than
  assuming silent approval means it landed. Do this directly, in-conversation
  — it does not depend on any particular harness or plugin. In a Claude Code
  session with the leverage plugin installed, `/leverage:verified-understanding`
  is a convenience command for the same discipline at bigger architecture
  ratification moments; use it there when available, but the underlying
  behavior (ask him to explain it back, don't just accept a quick "yes")
  applies in every harness this contract binds, including ones without that
  plugin.
