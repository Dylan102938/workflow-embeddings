"""Random-access reader for split ZIP archives (images.z01 ... images.zNN + images.zip) without merging.

Python's zipfile refuses multi-disk archives, and merging needs a second copy of the data (~200 GB) that
the NAS quota cannot hold. Offsets in a split archive are relative to the start of the part (disk) that
holds them, so we map (disk, offset) onto a virtual concatenation of the parts.
"""
from __future__ import annotations

import bisect
import os
import struct
import zlib
from pathlib import Path


def split_parts(zip_path: Path) -> list[Path]:
    """images.zip -> [images.z01, ..., images.zNN, images.zip] (disk order)."""
    zip_path = Path(zip_path)
    parts = sorted(zip_path.parent.glob(zip_path.stem + ".z[0-9][0-9]"))
    return parts + [zip_path]


class SplitZip:
    def __init__(self, zip_path: Path):
        self.parts = split_parts(zip_path)
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = [0]
        for s in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + s)
        self._fh: dict[int, object] = {}
        self.members = self._read_central_directory()

    # ---- virtual concatenation ----
    def _handle(self, i):
        fh = self._fh.get(i)
        if fh is None:
            fh = self._fh[i] = open(self.parts[i], "rb")
        return fh

    def _read(self, pos: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            i = bisect.bisect_right(self.starts, pos) - 1
            fh = self._handle(i)
            off = pos - self.starts[i]
            fh.seek(off)
            chunk = fh.read(min(n, self.sizes[i] - off))
            if not chunk:
                raise EOFError(f"read past end of {self.parts[i]}")
            out += chunk
            pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def _abs(self, disk: int, offset: int) -> int:
        return self.starts[disk] + offset

    # ---- central directory ----
    def _read_central_directory(self):
        last = len(self.parts) - 1
        tail_len = min(self.sizes[last], 65536 + 22)
        tail = self._read(self.starts[last] + self.sizes[last] - tail_len, tail_len)
        e = tail.rfind(b"PK\x05\x06")
        if e < 0:
            raise ValueError("no end-of-central-directory record")
        (_, disk_no, cd_disk, n_this, n_total, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[e:e + 22])
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        if loc >= 0:  # ZIP64
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
            rec = self._read(self._abs(z64_disk, z64_off), 56)
            (sig, _, _, _, disk_no, cd_disk, n_this, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec)
            assert sig == 0x06064B50, "bad zip64 end-of-central-directory record"
        cd = self._read(self._abs(cd_disk, cd_off), cd_size)
        members, p = {}, 0
        while p < len(cd):
            (sig, _, _, flags, method, _, _, _, csize, usize, nlen, xlen, clen, disk, _, _, lho) = \
                struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            assert sig == 0x02014B50, f"bad central directory entry at {p}"
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    z = extra[q + 4:q + 4 + hlen]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack("<I", z[k:k + 4])[0]; k += 4
                q += 4 + hlen
            if not name.endswith("/"):
                members[name] = (disk, lho, csize, method, flags)
            p += 46 + nlen + xlen + clen
        return members

    def read(self, name: str) -> bytes:
        disk, lho, csize, method, flags = self.members[name]
        pos = self._abs(disk, lho)
        hdr = self._read(pos, 30)
        sig, _, _, _, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", hdr)
        assert sig == 0x04034B50, f"bad local header for {name}"
        if flags & 0x1:
            raise ValueError(f"{name} is encrypted")
        data = self._read(pos + 30 + nlen + xlen, csize)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompressobj(-15).decompress(data)
        raise ValueError(f"unsupported compression method {method} for {name}")

    def basename_index(self) -> dict[str, str]:
        return {os.path.basename(n): n for n in self.members}

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
