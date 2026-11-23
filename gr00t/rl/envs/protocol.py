"""Wire format shared by env servers and clients (msgpack + numpy, no torch)."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy as mnp


def pack(data: Any) -> bytes:
    return msgpack.packb(data, default=mnp.encode, use_bin_type=True)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False, strict_map_key=False)
