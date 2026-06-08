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


MMQ_RE = re.compile(
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
    r"(?:src1_name=(?P<src1_name>.*?) )?"
    r"(?:src1_ptr=(?P<src1_ptr>\S+) )?)?"
    r"name=(?P<name>.*)$"
)

NODE_RE = re.compile(
    r"GGML_CUDA_PROFILE_NODE fused=(?P<fused>[01]) skipped=(?P<skipped>\d+) "
    r"op=(?P<op>\S+) name=(?P<name>\S*) ms=(?P<ms>[0-9.]+) "
    r"dst_type=(?P<dst_type>\S+) dst_ne=(?P<dst_ne>[0-9,]+)"
)

NODE_NAME_RE = re.compile(r"^node_(\d+)$")


def parse_node_id(name: str) -> int | None:
    match = NODE_NAME_RE.match(name)
    return int(match.group(1)) if match else None


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


def parse_log(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    mmq_rows: list[dict[str, Any]] = []
    fused_nodes: dict[int, dict[str, Any]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := MMQ_RE.search(line):
            groups = match.groupdict()
            mmq_rows.append(
                {
                    "source": str(path),
                    "quant_ms": float(groups["quant"]),
                    "mmq_ms": float(groups["mmq"]),
                    "total_ms": float(groups["total"]),
                    "stream_k": groups["stream_k"] == "1",
                    "native_fp4": groups["native_fp4"] == "1",
                    "src0_type": groups["src0_type"],
                    "src0_ne": groups["src0_ne"],
                    "src1_ne": groups["src1_ne"],
                    "dst_ne": groups["dst_ne"],
                    "src0_name": groups.get("src0_name") or "",
                    "src1_name": (groups.get("src1_name") or "").strip(),
                    "name": groups["name"],
                }
            )
            continue

        if match := NODE_RE.search(line):
            groups = match.groupdict()
            node_id = parse_node_id(groups["name"])
            if node_id is None:
                continue
            fused_nodes[node_id] = {
                "fused": groups["fused"] == "1",
                "skipped": int(groups["skipped"]),
                "op": groups["op"],
                "name": groups["name"],
                "ms": float(groups["ms"]),
                "dst_type": groups["dst_type"],
                "dst_ne": groups["dst_ne"],
            }

    return mmq_rows, fused_nodes


def producer_fused_node(row: dict[str, Any], fused_nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    row_id = parse_node_id(str(row["name"]))
    if row_id is None:
        return None
    for source_id in (row_id, row_id - 1, row_id - 2):
        node = fused_nodes.get(source_id)
        if node and node["op"] == "MUL_MAT" and node["fused"] and node["skipped"] >= 2:
            return node
    return None


def consumer_fused_node(row: dict[str, Any], fused_nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    row_id = parse_node_id(str(row["name"]))
    if row_id is None:
        return None
    for source_id in (row_id, row_id - 1):
        node = fused_nodes.get(source_id)
        if node and node["op"] == "MUL_MAT" and node["fused"]:
            return node
    return None


def parse_ne(ne: str) -> list[int]:
    return [int(part) for part in ne.split(",")]


def element_count(ne: str) -> int:
    total = 1
    for dim in parse_ne(ne):
        total *= dim
    return total


def summarize(paths: list[Path], drop_first_per_consumer_shape: bool) -> dict[str, Any]:
    all_mmq_rows: list[dict[str, Any]] = []
    all_fused_nodes: dict[int, dict[str, Any]] = {}
    for path in paths:
        mmq_rows, fused_nodes = parse_log(path)
        all_mmq_rows.extend(mmq_rows)
        all_fused_nodes.update(fused_nodes)

    have_node_profile = bool(all_fused_nodes)

    producers = {}
    for row in all_mmq_rows:
        if row["src0_type"] != "q4_1" or ".mlp.fc1.weight" not in str(row["src0_name"]):
            continue
        if have_node_profile and producer_fused_node(row, all_fused_nodes) is None:
            continue
        producers[str(row["name"])] = row

    candidates: list[dict[str, Any]] = []
    for row in all_mmq_rows:
        producer = producers.get(str(row["src1_name"]))
        if producer is None:
            continue
        if row["src0_type"] != "q4_1" or ".mlp.fc2.weight" not in str(row["src0_name"]):
            continue
        if row["src1_ne"] != producer["dst_ne"]:
            continue
        if have_node_profile and consumer_fused_node(row, all_fused_nodes) is None:
            continue
        dst_elements = element_count(producer["dst_ne"])
        candidates.append(
            {
                "producer_name": producer["name"],
                "producer_src0_name": producer["src0_name"],
                "producer_dst_ne": producer["dst_ne"],
                "consumer_name": row["name"],
                "consumer_src0_name": row["src0_name"],
                "consumer_src0_ne": row["src0_ne"],
                "consumer_src1_ne": row["src1_ne"],
                "consumer_dst_ne": row["dst_ne"],
                "consumer_quant_ms": row["quant_ms"],
                "consumer_mmq_ms": row["mmq_ms"],
                "consumer_total_ms": row["total_ms"],
                "producer_f32_write_mib": dst_elements * 4 / (1024 * 1024),
                "stream_k": row["stream_k"],
            }
        )

    grouped_candidates = candidates
    if drop_first_per_consumer_shape:
        by_shape: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_shape[(row["consumer_src0_ne"], row["consumer_src1_ne"], row["consumer_dst_ne"])].append(row)
        grouped_candidates = [
            row
            for rows in by_shape.values()
            for row in (rows[1:] if len(rows) > 1 else rows)
        ]

    by_consumer_shape: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_candidates:
        groups[(row["consumer_src0_ne"], row["consumer_src1_ne"], row["consumer_dst_ne"])].append(row)
    for (src0_ne, src1_ne, dst_ne), rows in groups.items():
        by_consumer_shape.append(
            {
                "consumer_src0_ne": src0_ne,
                "consumer_src1_ne": src1_ne,
                "consumer_dst_ne": dst_ne,
                "count": len(rows),
                "consumer_quant_ms": stats([float(row["consumer_quant_ms"]) for row in rows]),
                "consumer_total_ms": stats([float(row["consumer_total_ms"]) for row in rows]),
                "producer_f32_write_mib": stats([float(row["producer_f32_write_mib"]) for row in rows]),
                "examples": rows[:6],
            }
        )
    by_consumer_shape.sort(key=lambda row: row["consumer_quant_ms"]["sum"], reverse=True)

    return {
        "sources": [str(path) for path in paths],
        "drop_first_per_consumer_shape": drop_first_per_consumer_shape,
        "node_profile_available": have_node_profile,
        "note": (
            "Counts q4_1 Hiera MLP fc1 MMQ+bias+GELU outputs that are consumed by "
            "a following q4_1 fc2 MMQ. consumer_quant_ms is the upper bound for a "
            "producer-side q8_1 DS4 side-output fusion that lets the consumer skip "
            "standalone activation quantization. producer_f32_write_mib is the "
            "intermediate f32 write volume that a q8-only producer fusion could elide "
            "only if graph analysis proves the fc1 output has no other real consumer."
        ),
        "candidate_count": len(grouped_candidates),
        "consumer_quant_ms": stats([float(row["consumer_quant_ms"]) for row in grouped_candidates]),
        "consumer_total_ms": stats([float(row["consumer_total_ms"]) for row in grouped_candidates]),
        "producer_f32_write_mib": stats([float(row["producer_f32_write_mib"]) for row in grouped_candidates]),
        "by_consumer_shape": by_consumer_shape,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--drop-first-per-consumer-shape", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args.logs, args.drop_first_per_consumer_shape)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
