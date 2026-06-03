# CLAUDE.md: Instructions for Claude Code

## Project Context

See `PROJECT.md` (if present) for project-specific structure, conventions, and domain context.

## General Principles

- **Correctness over cleverness.** Write working, production-ready code.
- **Simplicity first.** Avoid unnecessary complexity. Don't propose adding complexity unless asked. Don't suggest large refactors during tuning or experimentation.
- **Respect scope.** Limit edits to what's requested. Suggest broader improvements separately.
- **Explicit incompleteness.** Unfinished work gets a comment explaining status—no hidden assumptions.
- **Follow conventions.** Use established best practices; if none exist, define one and be consistent.
- **Design for resilience.** Anticipate edge cases; handle unexpected inputs gracefully.
- **Explicit errors.** Prefer clear error messages over silent failures.
- **Sparse comments.** Comment only to clarify intent or non-obvious reasoning.
- **Track dependencies.** Note any new external requirements clearly.
- **Document minimally.** One-line summaries for modules, functions, and classes at minimum.
- **Security awareness.** Avoid hardcoded secrets; flag potential security concerns.

## Communication Style

- **Lead with the answer.** Tables with deltas over prose. Compare old vs new with numbers.
- **Be direct.** "50x worse", "catastrophic", "identical" — not hedged language.
- **One suggestion, not a menu.** Suggest one (or few) next steps; let the user redirect.
- **Understand WHY before fixing.** Explain causation, not just correlation.
- **Don't second-guess data with theory.** The data is what it is.
- **Momentum.** After logging results, immediately suggest the next experiment.
- **Log before moving on.** Record results and decisions before starting the next thing.

## Autonomy and Asking

- **Front-load questions.** Ask all clarifying questions before starting work (typically in Plan Mode).
- **Proceed autonomously** once the plan is confirmed—unless:
    - A critical issue or decision point emerges that wasn't anticipated,
    - It cannot be reasonably postponed or would significantly benefit from input now.
- **When in doubt, ask.** A brief pause beats compounding a wrong assumption.
- **Don't re-derive settled decisions.** Check existing logs, records, and memory first.

## Handling Existing Code

- **New code:** Follow these guidelines.
- **Edits to existing code:** Follow these guidelines where possible. On conflict, prefer local consistency within the file.
- **Re-read before editing** if a linter or formatter may have modified the file since your last read.
- **Check for undefined variables and unused imports** before considering multi-file changes complete.
- **Don't add docstrings/types to experimental scripts** unless asked.

## Important Constraints

- **Don't run builds or heavy commands** unless explicitly told to. Projects may target remote hardware (DGX, SLURM clusters).
- **Save context early and often.** Long sessions hit context limits — dump important state defensively.

## Python

- **Python 3.14+**, run with `python -O`, typically from `src/`.
- **PyTorch** for tensor operations, differentiable programming, and ML tasks. Use numpy and JAX idioms correctly where applicable.
- **Modern type hints.** Use native syntax: `tuple[int, float]`, `int | None`, `collections.abc` imports.
- **Ruff** for formatting and linting. Use `~/ruffconfigs/default/ruff.toml` if available.
- **Docstrings** (`"""..."""`) for all modules, functions, and classes.
- Common libraries: `simple_parsing`, `safetensors`.

## Environment

- **Shell:** Fish. Use `rm -f` not `rm` (interactive alias).
