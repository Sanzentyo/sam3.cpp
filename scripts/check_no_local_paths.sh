#!/usr/bin/env bash
set -euo pipefail

if ! command -v rg >/dev/null 2>&1; then
    echo "rg is required" >&2
    exit 2
fi

paths=(
    README.md
    docs
    scripts
    .codex
)

patterns=()

if [[ -n "${USER:-}" ]]; then
    patterns+=("$(printf '%s' "$USER" | sed 's/[][\\.^$*+?{}|()]/\\&/g')")
fi

if [[ -n "${HOME:-}" ]]; then
    patterns+=("$(printf '%s' "$HOME" | sed 's/[][\\.^$*+?{}|()]/\\&/g')")
fi

slash="/"
patterns+=("${slash}home${slash}[A-Za-z0-9._-]+${slash}")
patterns+=("${slash}Users${slash}[A-Za-z0-9._-]+${slash}")

pattern="$(IFS='|'; echo "${patterns[*]}")"

rg -n --hidden --glob '!scripts/check_no_local_paths.sh' "$pattern" "${paths[@]}" && {
    echo "local user/home path leaked into tracked docs or scripts" >&2
    exit 1
}

exit 0
