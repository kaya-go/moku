# Specs

Every non-trivial change starts as a spec here, before any code: one file per piece of work,
numbered in creation order (`NNN-short-name.md`). The spec is the source of truth for *why* and
*when it is done*; issues and PRs link to it instead of restating it.

## Template

```markdown
# NNN — Title

Status: draft | in progress | done | abandoned — Issue/PR links

## Why
The problem, with the numbers that show it.

## Hypothesis / goal
What we expect to change, stated so that it can be wrong.

## Acceptance criteria
Checkable conditions (metrics with CIs, commands that must pass, files that must exist).

## Design
The decisions and their reasons; what we deliberately leave out.

## Tasks
- [ ] Small, verifiable steps, ticked as they land.

## Results
What happened (runs, tables, links), and the decision it led to.
```

## Rules

- Write or update the spec first; code follows the spec, not the reverse.
- Keep *Tasks* and *Results* current while working: they are the resume point for the next session.
- Results quote board metrics from `moku eval` with their 90% CIs (see the evaluation principle in `CLAUDE.md`).
- A spec that changes a decision updates `docs/architecture.md` too.

## Index

| Spec | Status |
|---|---|
| [001 — Training loop and run tracking](001-training-loop.md) | in progress |
| [002 — D-FINE](002-dfine.md) | in progress |
| [003 — External evaluation data](003-external-eval-data.md) | draft |
