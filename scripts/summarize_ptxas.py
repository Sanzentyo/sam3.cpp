#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


COMPILE_RE = re.compile(r"Compiling entry function '(?P<name>[^']+)'")
STACK_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<spill_stores>\d+) bytes spill stores, (?P<spill_loads>\d+) bytes spill loads"
)
REG_RE = re.compile(r"Used (?P<registers>\d+) registers")
MMQ_X_RE = re.compile(
    r"_Z9mul_mat_qIL9ggml_type(?P<ggml_type>\d+)ELi(?P<mmq_x>\d+)ELb(?P<need_check>[01])"
    r"ELb(?P<plain_writeback>[01])(?:EL14mmq_activation(?P<activation>\d+))?E"
)
FIXUP_RE = re.compile(
    r"_Z24mul_mat_q_stream_k_fixupIL9ggml_type(?P<ggml_type>\d+)ELi(?P<mmq_x>\d+)ELb(?P<need_check>[01])"
    r"ELb(?P<plain_writeback>[01])(?:EL14mmq_activation(?P<activation>\d+))?E"
)


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := COMPILE_RE.search(line):
            if current:
                rows.append(current)
            name = match.group("name")
            current = {"function": name, "source": str(path)}
            if mmq := MMQ_X_RE.search(name):
                current.update(
                    {
                        "kernel": "mul_mat_q",
                        "ggml_type": int(mmq.group("ggml_type")),
                        "mmq_x": int(mmq.group("mmq_x")),
                        "need_check": bool(int(mmq.group("need_check"))),
                        "plain_writeback": bool(int(mmq.group("plain_writeback"))),
                        "activation": int(mmq.group("activation") or 0),
                    }
                )
            elif fixup := FIXUP_RE.search(name):
                current.update(
                    {
                        "kernel": "mul_mat_q_stream_k_fixup",
                        "ggml_type": int(fixup.group("ggml_type")),
                        "mmq_x": int(fixup.group("mmq_x")),
                        "need_check": bool(int(fixup.group("need_check"))),
                        "plain_writeback": bool(int(fixup.group("plain_writeback"))),
                        "activation": int(fixup.group("activation") or 0),
                    }
                )
            else:
                current["kernel"] = "other"
            continue

        if current is None:
            continue

        if match := STACK_RE.search(line):
            current["stack_bytes"] = int(match.group("stack"))
            current["spill_store_bytes"] = int(match.group("spill_stores"))
            current["spill_load_bytes"] = int(match.group("spill_loads"))
        if match := REG_RE.search(line):
            current["registers"] = int(match.group("registers"))

    if current:
        rows.append(current)
    return rows


def summarize(rows: list[dict[str, Any]], ggml_type: int | None = None) -> dict[str, Any]:
    mmq_rows = [row for row in rows if row.get("kernel") in {"mul_mat_q", "mul_mat_q_stream_k_fixup"}]
    if ggml_type is not None:
        mmq_rows = [row for row in mmq_rows if row.get("ggml_type") == ggml_type]
    hot = sorted(
        mmq_rows,
        key=lambda row: (
            -(row.get("registers") or 0),
            -(row.get("spill_store_bytes") or 0) - (row.get("spill_load_bytes") or 0),
            row.get("kernel", ""),
            row.get("mmq_x", 0),
        ),
    )
    by_mmq_x: dict[int, dict[str, Any]] = {}
    for row in mmq_rows:
        mmq_x = int(row.get("mmq_x", -1))
        group = by_mmq_x.setdefault(
            mmq_x,
            {
                "mmq_x": mmq_x,
                "rows": 0,
                "max_registers": 0,
                "max_spill_store_bytes": 0,
                "max_spill_load_bytes": 0,
                "kernels": set(),
                "ggml_types": set(),
                "activations": set(),
            },
        )
        group["rows"] += 1
        group["max_registers"] = max(group["max_registers"], row.get("registers") or 0)
        group["max_spill_store_bytes"] = max(group["max_spill_store_bytes"], row.get("spill_store_bytes") or 0)
        group["max_spill_load_bytes"] = max(group["max_spill_load_bytes"], row.get("spill_load_bytes") or 0)
        group["kernels"].add(row.get("kernel"))
        if "ggml_type" in row:
            group["ggml_types"].add(row["ggml_type"])
        if "activation" in row:
            group["activations"].add(row["activation"])

    by_mmq_x_rows = []
    for group in by_mmq_x.values():
        group = dict(group)
        group["kernels"] = sorted(group["kernels"])
        group["ggml_types"] = sorted(group["ggml_types"])
        group["activations"] = sorted(group["activations"])
        by_mmq_x_rows.append(group)
    by_mmq_x_rows.sort(key=lambda group: group["mmq_x"], reverse=True)

    return {
        "rows": len(rows),
        "filter_ggml_type": ggml_type,
        "mmq_rows": len(mmq_rows),
        "by_mmq_x": by_mmq_x_rows,
        "top_register_rows": hot[:32],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize nvcc ptxas verbose output.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ggml-type", type=int, help="Only summarize MMQ rows for this ggml_type value.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.logs:
        rows.extend(parse_log(path))

    summary = summarize(rows, args.ggml_type)
    summary["sources"] = [str(path) for path in args.logs]
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
