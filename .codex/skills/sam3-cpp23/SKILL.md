---
name: sam3-cpp23-modernization
description: Use when modernizing this sam3.cpp worktree to C++23, xmake, and just while preserving measured model parity and performance. Applies to sam3.cpp, examples, benchmark tooling, ggml interop, and Codex changes in this repository.
---

# sam3.cpp C++23 Modernization

Use this skill for C++23 or tooling work in this repository. The goal is not cosmetic
modernization; every change should either improve maintainability, resource safety,
build hygiene, or benchmark reliability without regressing parity or performance.

## Working Tree

- Work in the active worktree branch, not the original checkout.
- Use `just` as the front door:
  - `just build`
  - `just fmt-check`
  - `just parity-stats 15 outputs/<run-name>`
- Keep `compile_commands.json` current when build metadata changes.

## Acceptance Gate

A C++23 modernization change is not done until:

- `just build` passes.
- `just fmt-check` passes.
- `git diff --check` passes.
- Parity JSONL comparison reports `diff_rows == 0`.
- Paired performance remains statistically comparable to the previous run, or the regression is
  explicitly quantified and justified.

## C++23 Policy

- Prefer RAII for owned resources: POSIX fds, `FILE*`, ggml contexts, backend buffers, allocators,
  and temporary files.
- Use `std::span`/`std::as_bytes` for byte transport instead of raw `void*` plus size when the
  boundary is internal C++.
- Use `std::expected` for fallible internal helpers where the caller can recover or report a clear
  error. Do not throw across inference or benchmark boundaries.
- Use `std::string_view` for borrowed text parameters; materialize `std::string` only when ownership
  or C API lifetime requires it.
- Use `std::array` for fixed-size buffers and wire structs where size is part of the contract.
- Add `[[nodiscard]]` to return values that carry ownership, success/failure, or benchmark data.
- Use concepts only for real generic code. Do not add template abstractions just to look modern.

## ggml Interop

- Treat ggml pointers as borrowed unless the code clearly owns them.
- Wrap owned ggml resources with move-only RAII before broad refactors.
- Validate tensor shapes before calling ggml operations that assert/abort.
- Avoid extra copies in hot paths unless the parity/perf run proves the cost is irrelevant.

## Style Constraints

- Follow this repo's `.clang-format` and `.clang-tidy`, even where older notes disagree.
- Prefer standard-library formatting such as `std::format` when the active C++23 toolchain builds it
  cleanly; do not add a fmt dependency for formatting-only changes.
- Keep changes scoped. Prefer one measurable modernization slice at a time.
