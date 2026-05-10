# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


SAM2_MAGIC = 0x73616D32
SAM3_MAGIC = 0x73616D33


TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q4_0 = 2
TYPE_Q4_1 = 3
TYPE_Q8_0 = 8


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: int
    dims: tuple[int, ...]
    header: bytes
    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.header) + len(self.data)


@dataclass(frozen=True)
class ModelRecords:
    header: bytes
    tensors: dict[str, TensorRecord]
    order: list[str]


def product(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def tensor_data_size(dtype: int, dims: tuple[int, ...]) -> int:
    n_el = product(dims)
    if dtype == TYPE_F32:
        return n_el * 4
    if dtype == TYPE_F16:
        return n_el * 2

    ne0 = dims[0]
    n_rows = n_el // ne0
    if dtype == TYPE_Q4_0:
        return n_rows * ((ne0 // 32) * 18)
    if dtype == TYPE_Q4_1:
        return n_rows * ((ne0 // 32) * 20)
    if dtype == TYPE_Q8_0:
        return n_rows * ((ne0 // 32) * 34)
    raise ValueError(f"unsupported tensor dtype {dtype}")


def read_model(path: Path) -> ModelRecords:
    blob = path.read_bytes()
    magic, version, _ftype, n_tensors = struct.unpack_from("<IIII", blob, 0)
    offset = 16

    if magic == SAM2_MAGIC and version == 1:
        fields = list(struct.unpack_from("<57i", blob, offset))
        offset += 57 * 4
        backbone_type = fields[1]
        if backbone_type == 2:
            offset += 19 * 4
    elif magic == SAM3_MAGIC and version == 3:
        n_global_attn = struct.unpack_from("<i", blob, offset + 7 * 4)[0]
        offset += (8 + min(n_global_attn, 4) + 28) * 4
    else:
        raise ValueError(f"{path}: unsupported magic/version 0x{magic:08x}/{version}")

    header = blob[:offset]
    tensors: dict[str, TensorRecord] = {}
    order: list[str] = []

    for _ in range(n_tensors):
        record_start = offset
        n_dims, name_len, dtype = struct.unpack_from("<iii", blob, offset)
        offset += 12
        dims = struct.unpack_from("<" + "i" * n_dims, blob, offset)
        offset += 4 * n_dims
        name = blob[offset : offset + name_len].decode("utf-8")
        offset += name_len
        pad = (32 - offset % 32) % 32
        offset += pad
        data_size = tensor_data_size(dtype, dims)
        data_start = offset
        offset += data_size
        header_bytes = blob[record_start:data_start]
        data = blob[data_start:offset]
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        tensors[name] = TensorRecord(name=name, dtype=dtype, dims=dims, header=header_bytes, data=data)
        order.append(name)

    return ModelRecords(header=header, tensors=tensors, order=order)


def replace_from_donor(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Model that provides the default tensors")
    parser.add_argument("--donor", type=Path, required=True, help="Model that provides selected tensors")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=[], help="Tensor prefix to copy from donor")
    args = parser.parse_args()

    prefixes = args.prefix or ["sam_pe.", "sam_dec.", "obj_ptr_proj."]
    base = read_model(args.base)
    donor = read_model(args.donor)
    if base.order != donor.order:
        raise ValueError("base and donor tensor order differ")

    replaced: list[str] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fout:
        fout.write(base.header)
        for name in base.order:
            source = donor.tensors[name] if replace_from_donor(name, prefixes) else base.tensors[name]
            if source is donor.tensors[name]:
                replaced.append(name)
            fout.write(source.header)
            fout.write(source.data)

    summary = {
        "base": str(args.base),
        "donor": str(args.donor),
        "out": str(args.out),
        "prefixes": prefixes,
        "tensors_total": len(base.order),
        "tensors_replaced": len(replaced),
        "bytes": args.out.stat().st_size,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
