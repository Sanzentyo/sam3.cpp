# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROFILE_RE = re.compile(
    r"GGML_CUDA_PROFILE_MMQ "
    r"quant_ms=(?P<quant>[0-9.]+) "
    r"mmq_ms=(?P<mmq>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+) "
    r"stream_k=(?P<stream_k>[01]) "
    r"native_fp4=(?P<native_fp4>[01]) "
    r"(?:src1_q8_cache=(?P<src1_q8_cache>[01]) )?"
    r"src0_type=(?P<src0_type>\S+) "
    r"src0_ne=(?P<src0_ne>[0-9,]+) "
    r"src1_ne=(?P<src1_ne>[0-9,]+) "
    r"dst_ne=(?P<dst_ne>[0-9,]+) "
    r"(?:(?:src0_name=(?P<src0_name>\S+) )?"
    r"(?:src1_name=(?P<src1_name>\S+) )?"
    r"(?:src1_ptr=(?P<src1_ptr>\S+) )?)?"
    r"name=(?P<name>.*)$"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
    return ordered[idx]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "sum": sum(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            {
                "path": str(path),
                "quant_ms": float(groups["quant"]),
                "mmq_ms": float(groups["mmq"]),
                "total_ms": float(groups["total"]),
                "stream_k": groups["stream_k"] == "1",
                "native_fp4": groups["native_fp4"] == "1",
                "src1_q8_cache": groups.get("src1_q8_cache") == "1",
                "src0_type": groups["src0_type"],
                "src0_ne": groups["src0_ne"],
                "src1_ne": groups["src1_ne"],
                "dst_ne": groups["dst_ne"],
                "src0_name": groups.get("src0_name") or "",
                "src1_name": groups.get("src1_name") or "",
                "src1_ptr": groups.get("src1_ptr") or "",
                "name": groups["name"],
            }
        )
    return rows


def shape_key(row: dict[str, Any]) -> str:
    return (
        f"src0={row['src0_type']}[{row['src0_ne']}] "
        f"src1=f32[{row['src1_ne']}] dst=f32[{row['dst_ne']}] "
        f"stream_k={int(row['stream_k'])} native_fp4={int(row['native_fp4'])}"
    )


def summarize(rows: list[dict[str, Any]], drop_first_per_shape: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[shape_key(row)].append(row)

    effective: list[dict[str, Any]] = []
    by_shape: list[dict[str, Any]] = []
    for key, shape_rows in groups.items():
        shape_effective = shape_rows[1:] if drop_first_per_shape and len(shape_rows) > 1 else shape_rows
        effective.extend(shape_effective)
        by_shape.append(
            {
                "shape": key,
                "count": len(shape_rows),
                "effective_count": len(shape_effective),
                "quant_ms": stats([float(row["quant_ms"]) for row in shape_effective]),
                "mmq_ms": stats([float(row["mmq_ms"]) for row in shape_effective]),
                "total_ms": stats([float(row["total_ms"]) for row in shape_effective]),
            }
        )

    by_shape.sort(key=lambda row: row["total_ms"]["sum"], reverse=True)
    return {
        "rows": len(rows),
        "effective_rows": len(effective),
        "drop_first_per_shape": drop_first_per_shape,
        "totals": {
            "quant_ms": stats([float(row["quant_ms"]) for row in effective]),
            "mmq_ms": stats([float(row["mmq_ms"]) for row in effective]),
            "total_ms": stats([float(row["total_ms"]) for row in effective]),
        },
        "by_shape": by_shape,
        "src1_reuse": summarize_src1_reuse(effective),
        "src1_name_reuse": summarize_src1_name_reuse(effective),
        "producer_consumer_q8_candidates": summarize_producer_consumer_candidates(effective),
        "src1_q8_cache": summarize_src1_q8_cache(effective),
    }


def summarize_src1_q8_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [row for row in rows if row.get("src1_q8_cache")]
    misses = [row for row in rows if not row.get("src1_q8_cache")]
    return {
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": len(hits) / len(rows) if rows else 0.0,
        "hit_quant_ms": stats([float(row["quant_ms"]) for row in hits]),
        "miss_quant_ms": stats([float(row["quant_ms"]) for row in misses]),
    }


def summarize_src1_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed_rows = [row for row in rows if row.get("src1_ptr")]
    by_ptr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keyed_rows:
        by_ptr[str(row["src1_ptr"])].append(row)

    reusable_groups = []
    for ptr, ptr_rows in by_ptr.items():
        if len(ptr_rows) <= 1:
            continue
        reusable_groups.append(
            {
                "src1_ptr": ptr,
                "src1_name": ptr_rows[0].get("src1_name", ""),
                "src1_ne": ptr_rows[0].get("src1_ne", ""),
                "count": len(ptr_rows),
                "unique_shapes": len({shape_key(row) for row in ptr_rows}),
                "quant_ms": stats([float(row["quant_ms"]) for row in ptr_rows]),
                "total_ms": stats([float(row["total_ms"]) for row in ptr_rows]),
                "dst_names": sorted({str(row["name"]) for row in ptr_rows})[:12],
            }
        )

    reusable_groups.sort(key=lambda row: row["quant_ms"]["sum"], reverse=True)
    return {
        "note": "Only populated for logs that include src1_ptr. Reuse means multiple MMQ nodes saw the same source activation pointer in the profiled run.",
        "rows_with_src1_ptr": len(keyed_rows),
        "unique_src1_ptrs": len(by_ptr),
        "reused_src1_ptrs": len(reusable_groups),
        "reused_quant_ms": stats(
            [
                float(row["quant_ms"])
                for ptr_rows in by_ptr.values()
                if len(ptr_rows) > 1
                for row in ptr_rows
            ]
        ),
        "top_reused_src1": reusable_groups[:20],
    }


def summarize_src1_name_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed_rows = [row for row in rows if row.get("src1_name")]
    by_name: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in keyed_rows:
        key = (
            str(row["src1_name"]),
            str(row["src1_ne"]),
            str(row["src0_type"]),
            str(row["dst_ne"]),
        )
        by_name[key].append(row)

    reusable_groups = []
    for (name, src1_ne, src0_type, dst_ne), name_rows in by_name.items():
        if len(name_rows) <= 1:
            continue
        quant_values = [float(row["quant_ms"]) for row in name_rows]
        reusable_groups.append(
            {
                "src1_name": name,
                "src1_ne": src1_ne,
                "src0_type": src0_type,
                "dst_ne": dst_ne,
                "count": len(name_rows),
                "quant_ms": stats(quant_values),
                "quant_ms_after_first": stats(quant_values[1:]),
                "dst_names": sorted({str(row["name"]) for row in name_rows})[:12],
            }
        )

    reusable_groups.sort(key=lambda row: row["quant_ms"]["sum"], reverse=True)
    return {
        "note": (
            "Groups rows by src1 tensor name, shape, src0 type, and dst shape. This is a conservative "
            "upper-bound signal for graph-identity reuse, but repeated names can still come from repeated "
            "graph executions over different frames and must not be treated as cache hits without a graph "
            "compute/liveness boundary."
        ),
        "rows_with_src1_name": len(keyed_rows),
        "unique_src1_name_shape_groups": len(by_name),
        "reused_src1_name_shape_groups": len(reusable_groups),
        "reused_quant_ms": stats(
            [
                float(row["quant_ms"])
                for name_rows in by_name.values()
                if len(name_rows) > 1
                for row in name_rows
            ]
        ),
        "reused_quant_ms_after_first": stats(
            [
                float(row["quant_ms"])
                for name_rows in by_name.values()
                if len(name_rows) > 1
                for row in name_rows[1:]
            ]
        ),
        "top_reused_src1_name": reusable_groups[:20],
    }


def summarize_producer_consumer_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    producers: dict[str, dict[str, Any]] = {
        str(row["name"]): row
        for row in rows
        if row.get("name")
    }

    candidates: list[dict[str, Any]] = []
    for row in rows:
        producer = producers.get(str(row.get("src1_name", "")))
        if producer is None:
            continue
        if producer.get("dst_ne") != row.get("src1_ne"):
            continue
        candidates.append(
            {
                "producer_name": producer["name"],
                "producer_src0_name": producer.get("src0_name", ""),
                "producer_shape": shape_key(producer),
                "consumer_name": row["name"],
                "consumer_src0_name": row.get("src0_name", ""),
                "consumer_shape": shape_key(row),
                "consumer_quant_ms": row["quant_ms"],
                "consumer_total_ms": row["total_ms"],
            }
        )

    by_consumer_weight: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_consumer_weight[str(row["consumer_src0_name"])].append(row)

    grouped = []
    for consumer_weight, group_rows in by_consumer_weight.items():
        grouped.append(
            {
                "consumer_src0_name": consumer_weight,
                "count": len(group_rows),
                "consumer_quant_ms": stats([float(row["consumer_quant_ms"]) for row in group_rows]),
                "consumer_total_ms": stats([float(row["consumer_total_ms"]) for row in group_rows]),
                "producer_src0_names": sorted({str(row["producer_src0_name"]) for row in group_rows})[:12],
                "examples": group_rows[:6],
            }
        )
    grouped.sort(key=lambda row: row["consumer_quant_ms"]["sum"], reverse=True)

    return {
        "note": (
            "Rows where an MMQ output tensor is later consumed as src1 by another MMQ with matching shape. "
            "These are direct producer-side q8_1 side-output candidates; the consumer quant_ms is the "
            "upper bound for removing the later activation quantize launch/read."
        ),
        "count": len(candidates),
        "consumer_quant_ms": stats([float(row["consumer_quant_ms"]) for row in candidates]),
        "consumer_total_ms": stats([float(row["consumer_total_ms"]) for row in candidates]),
        "by_consumer_src0_name": grouped[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--drop-first-per-shape", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.logs:
        rows.extend(parse_log(path))

    result = {
        "sources": [str(path) for path in args.logs],
        "note": "GGML_CUDA_PROFILE_MMQ synchronizes CUDA events around each MMQ node; use for attribution, not normal runtime.",
        **summarize(rows, args.drop_first_per_shape),
    }
    result["by_shape"] = result["by_shape"][: args.limit]

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
