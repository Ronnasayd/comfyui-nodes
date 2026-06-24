
<!-- INIT AUTO-CONTEXT -->


## Environments

- JS/TS: use `yarn`, not `npm`, unless project says `npm`.
- Python: use `pip` + `venv`.
- Multi-language versions: use `asdf` when fit.

## Documentation

Search context in this order:

- `@docs/SUMMARY.md` — executive summary, module table, quick start, full doc index
- `@docs/architecture.md` — Mermaid diagrams, components, data flow, patterns, ADR links
- `@docs/setup.md` — requirements, env vars, yarn commands, local setup guide
- `@docs/usage.md` — core flows with end-to-end HTTP examples (assign, deactivate, bulk, amendment)
- `@docs/contribution.md` — branch/commit conventions, layer rules, naming, test locations
- `@docs/models.md` — full data model for all entities with fields and invariants
- `@docs/endpoints.md` — all REST endpoints grouped by module with request/response examples
- `@docs/faq.md` — Q&A covering setup, auth, licensing, billing, and architecture
- `@docs/modules/<module>.md` — high-level context for each bounded context (generated first; linked via `CONTEXT.md` symlinks inside each module folder)
- `@docs/adr/` — Architectural Decision Records (ADR-001 to ADR-005)

When entering module, check for `CONTEXT.md` inside module directory. File holds module-only context.

Load only minimum sections needed for task domain.

## Always Use Interactive Question Tools

For every user question, use interactive question tool. No exceptions for context, type, or intent.

Use this for clarifications, options, confirmations, preference checks, all user interactions.

- **VS Code (GitHub Copilot)**: Use `vscode_askQuestions`
- **Other environments**: Use equivalent interactive question tools available in your context
- **Fallback**: if no interactive tools exist, use labeled options (A, B, C... Z)

If interactive tool exists, never ask plain-text question.

## Response Style

- Activate skill **caveman** in mode full when available. If not, use concise, direct style.

## Context-Specific Rules

The following rules apply to specific file types:
- [code.instructions](.claude/instructions/code.instructions.md) — applies to: `**/*.ts, **/*.js, **/*.py, **/*.java, **/*.go, **/*.css, **/*.cpp, **/*.c, **/*.vue, **/*.jsx, **/*.tsx`
- [test.instructions](.claude/instructions/test.instructions.md) — applies to: `**/*.test.ts,**/*.test.js`
<!-- END AUTO-CONTEXT -->
