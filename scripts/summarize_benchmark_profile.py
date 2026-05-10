# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TABLE_ROW_RE = re.compile(
    r"^\s*\d+\s*\|\s*(?P<model>\S+)\s+\|[^\n]*?\|\s*(?P<backend>CUDA|CPU)\s*\|"
    r"\s*(?P<load>[0-9.]+)\s*\|\s*(?P<init>[0-9.]+)\s*\|"
    r"\s*(?P<track>[0-9.]+)\s*\|\s*(?P<p50>[0-9.]+)\s*\|"
    r"\s*(?P<p95>[0-9.]+)\s*\|\s*(?P<total>[0-9.]+)",
    re.M,
)
ENCODE_RE = re.compile(r"sam2_encode_image_hiera: SAM2 image encoded in ([0-9.]+) ms")
COMPUTE_RE = re.compile(
    r"SAM3_PROFILE compute (?:label=(\S+) )?backend=(\S+) nodes=(\d+) ms=([0-9.]+)"
)
TENSOR_SET_RE = re.compile(r"SAM3_PROFILE tensor_set name=\S+ bytes=\d+ offset=\d+ ms=([0-9.]+)")
TENSOR_GET_RE = re.compile(r"SAM3_PROFILE tensor_get name=\S+ bytes=\d+ offset=\d+ ms=([0-9.]+)")
CPU_SPAN_RE = re.compile(r"SAM3_PROFILE cpu name=(\S+) ms=([0-9.]+)")
HIERA_CUT_RE = re.compile(
    r"SAM3_PROFILE_HIERA_CUT label=(\S+) nodes=(\d+) mean_ms=([0-9.]+) warmup=(\d+) iter=(\d+)"
)
HIERA_CUT_OPS_RE = re.compile(
    r"SAM3_PROFILE_HIERA_CUT_OPS label=(\S+) op=(\S+) count=(\d+) out_elements=(\d+)"
)
HIERA_CUT_MATMUL_RE = re.compile(
    r"SAM3_PROFILE_HIERA_CUT_MATMUL label=(\S+) "
    r"src0_type=(\S+) src0_ne=([0-9,]+) "
    r"src1_type=(\S+) src1_ne=([0-9,]+) "
    r"dst_type=(\S+) dst_ne=([0-9,]+)"
)
GRAPH_OPS_RE = re.compile(
    r"SAM3_PROFILE_GRAPH_OPS label=(\S+) op=(\S+) count=(\d+) out_elements=(\d+)"
)
GRAPH_MATMUL_RE = re.compile(
    r"SAM3_PROFILE_GRAPH_MATMUL label=(\S+) "
    r"src0_type=(\S+) src0_ne=([0-9,]+) "
    r"src1_type=(\S+) src1_ne=([0-9,]+) "
    r"dst_type=(\S+) dst_ne=([0-9,]+)"
)
CUDA_NODE_RE = re.compile(
    r"GGML_CUDA_PROFILE_NODE fused=(\d+) skipped=(\d+) op=(\S+) name=(\S*) ms=([0-9.]+) "
    r"dst_type=(\S+) dst_ne=([0-9,]+)"
    r"(?: src0_type=(\S+) src0_ne=([0-9,]+) src1_type=(\S+) src1_ne=([0-9,]+))?"
)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = TABLE_ROW_RE.search(text)
    encodes = [float(value) for value in ENCODE_RE.findall(text)]
    computes = [
        (label or "unlabeled", backend, int(nodes), float(ms))
        for label, backend, nodes, ms in COMPUTE_RE.findall(text)
    ]
    tensor_sets = [float(value) for value in TENSOR_SET_RE.findall(text)]
    tensor_gets = [float(value) for value in TENSOR_GET_RE.findall(text)]
    cpu_spans: dict[str, list[float]] = {}
    for name, ms in CPU_SPAN_RE.findall(text):
        cpu_spans.setdefault(name, []).append(float(ms))
    hiera_cuts: dict[str, list[dict[str, Any]]] = {}
    for label, nodes, ms, warmup, iters in HIERA_CUT_RE.findall(text):
        hiera_cuts.setdefault(label, []).append(
            {
                "nodes": int(nodes),
                "mean_ms": float(ms),
                "warmup": int(warmup),
                "iter": int(iters),
            }
        )
    hiera_cut_ops: dict[str, dict[str, dict[str, int | float]]] = {}
    for label, op, count, elements in HIERA_CUT_OPS_RE.findall(text):
        op_stats = hiera_cut_ops.setdefault(label, {}).setdefault(
            op, {"samples": 0, "count": 0, "out_elements": 0}
        )
        op_stats["samples"] += 1
        op_stats["count"] += int(count)
        op_stats["out_elements"] += int(elements)
    for ops in hiera_cut_ops.values():
        for op_stats in ops.values():
            samples = int(op_stats["samples"])
            op_stats["mean_count"] = op_stats["count"] / samples
            op_stats["mean_out_elements"] = op_stats["out_elements"] / samples
    hiera_cut_matmuls: dict[str, dict[str, dict[str, int | float | str]]] = {}
    for label, src0_type, src0_ne, src1_type, src1_ne, dst_type, dst_ne in HIERA_CUT_MATMUL_RE.findall(
        text
    ):
        key = (
            f"src0={src0_type}[{src0_ne}] src1={src1_type}[{src1_ne}] "
            f"dst={dst_type}[{dst_ne}]"
        )
        stats = hiera_cut_matmuls.setdefault(label, {}).setdefault(
            key,
            {
                "count": 0,
                "src0_type": src0_type,
                "src0_ne": src0_ne,
                "src1_type": src1_type,
                "src1_ne": src1_ne,
                "dst_type": dst_type,
                "dst_ne": dst_ne,
            },
        )
        stats["count"] += 1
    for label, matmuls in hiera_cut_matmuls.items():
        label_samples = max(1, len(hiera_cuts.get(label, [])))
        for stats in matmuls.values():
            stats["samples"] = label_samples
            stats["mean_count"] = stats["count"] / label_samples
    graph_ops: dict[str, dict[str, dict[str, int | float]]] = {}
    for label, op, count, elements in GRAPH_OPS_RE.findall(text):
        op_stats = graph_ops.setdefault(label, {}).setdefault(
            op, {"samples": 0, "count": 0, "out_elements": 0}
        )
        op_stats["samples"] += 1
        op_stats["count"] += int(count)
        op_stats["out_elements"] += int(elements)
    for ops in graph_ops.values():
        for op_stats in ops.values():
            samples = int(op_stats["samples"])
            op_stats["mean_count"] = op_stats["count"] / samples
            op_stats["mean_out_elements"] = op_stats["out_elements"] / samples
    graph_matmuls: dict[str, dict[str, dict[str, int | float | str]]] = {}
    for label, src0_type, src0_ne, src1_type, src1_ne, dst_type, dst_ne in GRAPH_MATMUL_RE.findall(
        text
    ):
        key = (
            f"src0={src0_type}[{src0_ne}] src1={src1_type}[{src1_ne}] "
            f"dst={dst_type}[{dst_ne}]"
        )
        stats = graph_matmuls.setdefault(label, {}).setdefault(
            key,
            {
                "count": 0,
                "src0_type": src0_type,
                "src0_ne": src0_ne,
                "src1_type": src1_type,
                "src1_ne": src1_ne,
                "dst_type": dst_type,
                "dst_ne": dst_ne,
            },
        )
        stats["count"] += 1
    for label, matmuls in graph_matmuls.items():
        # Estimate samples from GRAPH_OPS when available; otherwise keep raw counts.
        label_samples = max(
            1,
            max((int(stats["samples"]) for stats in graph_ops.get(label, {}).values()), default=1),
        )
        for stats in matmuls.values():
            stats["samples"] = label_samples
            stats["mean_count"] = stats["count"] / label_samples

    cuda_nodes: list[dict[str, Any]] = []
    cuda_nodes_by_op: dict[str, dict[str, Any]] = {}
    cuda_nodes_by_signature: dict[str, dict[str, Any]] = {}
    cuda_nodes_by_label_op: dict[str, dict[str, dict[str, Any]]] = {}
    cuda_nodes_by_label_signature: dict[str, dict[str, dict[str, Any]]] = {}

    def make_cuda_node(match: tuple[str, ...]) -> dict[str, Any]:
        fused, skipped, op, name, ms, dst_type, dst_ne, src0_type, src0_ne, src1_type, src1_ne = match
        return {
            "fused": bool(int(fused)),
            "skipped": int(skipped),
            "op": op,
            "name": name,
            "ms": float(ms),
            "dst_type": dst_type,
            "dst_ne": dst_ne,
            "src0_type": src0_type or "none",
            "src0_ne": src0_ne or "0,0,0,0",
            "src1_type": src1_type or "none",
            "src1_ne": src1_ne or "0,0,0,0",
        }

    def cuda_node_signature(node_row: dict[str, Any]) -> str:
        return (
            f"op={node_row['op']} dst={node_row['dst_type']}[{node_row['dst_ne']}] "
            f"src0={node_row['src0_type']}[{node_row['src0_ne']}] "
            f"src1={node_row['src1_type']}[{node_row['src1_ne']}]"
        )

    def add_cuda_node_to(
        node_row: dict[str, Any],
        by_op: dict[str, dict[str, Any]],
        by_signature: dict[str, dict[str, Any]],
    ) -> None:
        op = node_row["op"]
        op_stats = by_op.setdefault(op, {"count": 0, "sum_ms": 0.0, "min_ms": None, "max_ms": 0.0})
        op_stats["count"] += 1
        op_stats["sum_ms"] += node_row["ms"]
        op_stats["min_ms"] = (
            node_row["ms"] if op_stats["min_ms"] is None else min(op_stats["min_ms"], node_row["ms"])
        )
        op_stats["max_ms"] = max(op_stats["max_ms"], node_row["ms"])
        signature = cuda_node_signature(node_row)
        sig_stats = by_signature.setdefault(
            signature,
            {
                "op": op,
                "dst_type": node_row["dst_type"],
                "dst_ne": node_row["dst_ne"],
                "src0_type": node_row["src0_type"],
                "src0_ne": node_row["src0_ne"],
                "src1_type": node_row["src1_type"],
                "src1_ne": node_row["src1_ne"],
                "count": 0,
                "sum_ms": 0.0,
                "min_ms": None,
                "max_ms": 0.0,
            },
        )
        sig_stats["count"] += 1
        sig_stats["sum_ms"] += node_row["ms"]
        sig_stats["min_ms"] = (
            node_row["ms"]
            if sig_stats["min_ms"] is None
            else min(sig_stats["min_ms"], node_row["ms"])
        )
        sig_stats["max_ms"] = max(sig_stats["max_ms"], node_row["ms"])

    for match in CUDA_NODE_RE.findall(text):
        node_row = make_cuda_node(match)
        cuda_nodes.append(node_row)
        add_cuda_node_to(node_row, cuda_nodes_by_op, cuda_nodes_by_signature)

    pending_nodes: list[dict[str, Any]] = []
    for line in text.splitlines():
        node_match = CUDA_NODE_RE.search(line)
        if node_match:
            pending_nodes.append(make_cuda_node(node_match.groups()))
            continue
        compute_match = COMPUTE_RE.search(line)
        if compute_match:
            label = compute_match.group(1) or "unlabeled"
            by_op = cuda_nodes_by_label_op.setdefault(label, {})
            by_signature = cuda_nodes_by_label_signature.setdefault(label, {})
            for node_row in pending_nodes:
                add_cuda_node_to(node_row, by_op, by_signature)
            pending_nodes = []

    all_stats = list(cuda_nodes_by_op.values()) + list(cuda_nodes_by_signature.values())
    for by_op in cuda_nodes_by_label_op.values():
        all_stats.extend(by_op.values())
    for by_signature in cuda_nodes_by_label_signature.values():
        all_stats.extend(by_signature.values())
    for stats in all_stats:
        stats["mean_ms"] = stats["sum_ms"] / max(1, stats["count"])
        if stats["count"] > 1:
            stats["sum_ms_drop_max"] = stats["sum_ms"] - stats["max_ms"]
            stats["mean_ms_drop_max"] = stats["sum_ms_drop_max"] / (stats["count"] - 1)
            stats["max_fraction_of_sum"] = stats["max_ms"] / stats["sum_ms"] if stats["sum_ms"] else 0.0

    result: dict[str, Any] = {
        "path": str(path),
        "encode_ms": encodes,
        "encode_mean_ms": mean(encodes),
        "profile_compute_count": len(computes),
        "profile_compute_sum_ms": sum(ms for _, _, _, ms in computes),
        "profile_compute_top": [
            {"label": label, "backend": backend, "nodes": nodes, "ms": ms}
            for label, backend, nodes, ms in sorted(computes, key=lambda item: item[3], reverse=True)[:5]
        ],
        "profile_compute_by_label": {
            label: {
                "count": len(values),
                "sum_ms": sum(values),
                "mean_ms": mean(values),
                "values_ms": values,
            }
            for label, values in sorted(
                {
                    label: [ms for item_label, _, _, ms in computes if item_label == label]
                    for label in {item_label for item_label, _, _, _ in computes}
                }.items()
            )
        },
        "tensor_set_sum_ms": sum(tensor_sets),
        "tensor_get_sum_ms": sum(tensor_gets),
        "tensor_get_count": len(tensor_gets),
        "cpu_spans": {
            name: {
                "count": len(values),
                "sum_ms": sum(values),
                "mean_ms": mean(values),
                "values_ms": values,
            }
            for name, values in sorted(cpu_spans.items())
        },
        "hiera_cuts": {
            label: {
                "count": len(values),
                "mean_ms": mean([row["mean_ms"] for row in values]),
                "values": values,
            }
            for label, values in sorted(hiera_cuts.items())
        },
        "hiera_cut_ops": {
            label: {
                op: stats
                for op, stats in sorted(ops.items(), key=lambda item: item[0])
            }
            for label, ops in sorted(hiera_cut_ops.items(), key=lambda item: item[0])
        },
        "hiera_cut_matmuls": {
            label: {
                key: stats
                for key, stats in sorted(
                    matmuls.items(), key=lambda item: (-float(item[1]["mean_count"]), item[0])
                )
            }
            for label, matmuls in sorted(hiera_cut_matmuls.items(), key=lambda item: item[0])
        },
        "graph_ops": {
            label: {
                op: stats
                for op, stats in sorted(ops.items(), key=lambda item: item[0])
            }
            for label, ops in sorted(graph_ops.items(), key=lambda item: item[0])
        },
        "graph_matmuls": {
            label: {
                key: stats
                for key, stats in sorted(
                    matmuls.items(), key=lambda item: (-float(item[1]["mean_count"]), item[0])
                )
            }
            for label, matmuls in sorted(graph_matmuls.items(), key=lambda item: item[0])
        },
        "cuda_profile_nodes_top": sorted(cuda_nodes, key=lambda item: item["ms"], reverse=True)[:20],
        "cuda_profile_nodes_by_op": {
            op: stats
            for op, stats in sorted(
                cuda_nodes_by_op.items(), key=lambda item: (-float(item[1]["sum_ms"]), item[0])
            )
        },
        "cuda_profile_nodes_by_signature": {
            signature: stats
            for signature, stats in sorted(
                cuda_nodes_by_signature.items(),
                key=lambda item: (-float(item[1]["sum_ms"]), item[0]),
            )[:50]
        },
        "cuda_profile_nodes_by_label_op": {
            label: {
                op: stats
                for op, stats in sorted(
                    by_op.items(), key=lambda item: (-float(item[1]["sum_ms"]), item[0])
                )
            }
            for label, by_op in sorted(cuda_nodes_by_label_op.items())
        },
        "cuda_profile_nodes_by_label_signature": {
            label: {
                signature: stats
                for signature, stats in sorted(
                    by_signature.items(),
                    key=lambda item: (-float(item[1]["sum_ms"]), item[0]),
                )[:50]
            }
            for label, by_signature in sorted(cuda_nodes_by_label_signature.items())
        },
    }
    if row:
        result.update(
            {
                "model": row.group("model"),
                "backend": row.group("backend"),
                "load_ms": float(row.group("load")),
                "init_ms": float(row.group("init")),
                "track_ms": float(row.group("track")),
                "p50_ms": float(row.group("p50")),
                "p95_ms": float(row.group("p95")),
                "total_ms": float(row.group("total")),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {"logs": [summarize(path) for path in args.logs]}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
