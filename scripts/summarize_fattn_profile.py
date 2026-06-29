# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FATTN_RE = re.compile(
    r"GGML_CUDA_PROFILE_FATTN name=(?P<name>\S+) "
    r"Q_convert_ms=(?P<Q>[0-9.]+) K_convert_ms=(?P<K>[0-9.]+) "
    r"V_convert_ms=(?P<V>[0-9.]+) kernel_ms=(?P<kernel>[0-9.]+) "
    r"fixup_ms=(?P<fixup>[0-9.]+) compute_ms=(?P<compute>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+).*? Q=\[(?P<shape>[^\]]+)\].*? types=(?P<types>\S+)"
)
FATTN_DETAIL_RE = re.compile(
    r"ncols1=(?P<ncols1>\d+) ncols2=(?P<ncols2>\d+) nbatch_fa=(?P<nbatch_fa>\d+) "
    r"parallel_blocks=(?P<parallel_blocks>\d+).*?schedule=(?P<schedule>\S+) bpt=(?P<bpt>\d+)"
)


def classify(name: str, shape: str) -> str:
    if name.startswith("sam3_vit_block") and "_window_" in name:
        return "sam3-vit-window-head64"
    if name.startswith("sam3_vit_block") and "_global_" in name:
        return "sam3-vit-global-head64"
    if shape.startswith("32,5184"):
        return "other-head32-long"
    return "other"


def summarize(path: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "types": Counter(),
            "ncols1": Counter(),
            "schedule": Counter(),
            "rows": [],
            "Q_convert_ms": 0.0,
            "K_convert_ms": 0.0,
            "V_convert_ms": 0.0,
            "kernel_ms": 0.0,
            "fixup_ms": 0.0,
            "compute_ms": 0.0,
            "total_ms": 0.0,
        }
    )

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FATTN_RE.search(line)
        if not match:
            continue
        detail = FATTN_DETAIL_RE.search(line)
        row = groups[classify(match.group("name"), match.group("shape"))]
        row["count"] += 1
        row["types"][match.group("types")] += 1
        if detail:
            row["ncols1"][detail.group("ncols1")] += 1
            row["schedule"][detail.group("schedule")] += 1
        values: dict[str, float] = {}
        for src, dst in (
            ("Q", "Q_convert_ms"),
            ("K", "K_convert_ms"),
            ("V", "V_convert_ms"),
            ("kernel", "kernel_ms"),
            ("fixup", "fixup_ms"),
            ("compute", "compute_ms"),
            ("total", "total_ms"),
        ):
            values[dst] = float(match.group(src))
            row[dst] += values[dst]
        row["rows"].append(values)

    result_groups: dict[str, dict[str, Any]] = {}
    for name, row in sorted(groups.items()):
        per_metric: dict[str, Any] = {}
        for metric in (
            "Q_convert_ms",
            "K_convert_ms",
            "V_convert_ms",
            "kernel_ms",
            "fixup_ms",
            "compute_ms",
            "total_ms",
        ):
            values = [sample[metric] for sample in row["rows"]]
            steady_values = values[1:] if len(values) > 1 else values
            per_metric[metric] = {
                "sum": row[metric],
                "sum_drop_first": sum(steady_values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "steady_mean": statistics.fmean(steady_values),
                "steady_median": statistics.median(steady_values),
                "min": min(values),
                "max": max(values),
            }
        result_groups[name] = {
            "count": row["count"],
            "types": dict(sorted(row["types"].items())),
            "ncols1": dict(sorted(row["ncols1"].items())),
            "schedule": dict(sorted(row["schedule"].items())),
            "Q_convert_ms": row["Q_convert_ms"],
            "K_convert_ms": row["K_convert_ms"],
            "V_convert_ms": row["V_convert_ms"],
            "KV_convert_ms": row["K_convert_ms"] + row["V_convert_ms"],
            "QKV_convert_ms": row["Q_convert_ms"] + row["K_convert_ms"] + row["V_convert_ms"],
            "kernel_ms": row["kernel_ms"],
            "fixup_ms": row["fixup_ms"],
            "compute_ms": row["compute_ms"],
            "total_ms": row["total_ms"],
            "metrics": per_metric,
        }

    return {
        "source": str(path),
        "note": "GGML_CUDA_PROFILE_FATTN uses CUDA events and synchronizes; compare groups, not wall time.",
        "groups": result_groups,
    }


def fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def fmt_counter(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}:{count}" for key, count in sorted(value.items()))


def steady_metric(group: dict[str, Any], metric: str) -> float | None:
    metrics = group.get("metrics")
    if not isinstance(metrics, dict):
        return None
    row = metrics.get(metric)
    if not isinstance(row, dict):
        return None
    value = row.get("steady_mean")
    return float(value) if isinstance(value, int | float) else None


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# FATTN Profile Summary",
        "",
        "CUDA event profiling synchronizes the stream, so use these rows for "
        "attribution rather than normal wall-clock speed.",
        "",
        "| group | calls | types | ncols1 | schedule | total steady ms | kernel steady ms | QKV convert steady ms | V convert steady ms | fixup steady ms |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups = result.get("groups")
    if isinstance(groups, dict):
        for name, group in groups.items():
            if not isinstance(group, dict):
                continue
            qkv = sum(
                steady_metric(group, metric) or 0.0
                for metric in ("Q_convert_ms", "K_convert_ms", "V_convert_ms")
            )
            lines.append(
                f"| `{name}` | {group.get('count', '-')} | "
                f"`{fmt_counter(group.get('types'))}` | "
                f"`{fmt_counter(group.get('ncols1'))}` | "
                f"`{fmt_counter(group.get('schedule'))}` | "
                f"{fmt_ms(steady_metric(group, 'total_ms'))} | "
                f"{fmt_ms(steady_metric(group, 'kernel_ms'))} | "
                f"{fmt_ms(qkv)} | "
                f"{fmt_ms(steady_metric(group, 'V_convert_ms'))} | "
                f"{fmt_ms(steady_metric(group, 'fixup_ms'))} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    result = summarize(args.log)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown(result), encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
