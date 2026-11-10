"""Random-access reader for split ZIP archives (images.z01 ... images.zNN + images.zip), read in place.

Python's zipfile refuses multi-disk archives; why they are not merged instead: docs/DATA_PREP.md.
"""

from __future__ import annotations

import bisect
import itertools
import os
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, NamedTuple

END_RECORD = struct.Struct("<IHHHHIIH")
END_RECORD_SIGNATURE = b"PK\x05\x06"
END_RECORD_SEARCH_LENGTH = 65536 + END_RECORD.size  # the record is followed by a comment of at most 64 KiB
ZIP64_LOCATOR = struct.Struct("<IIQI")
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_END_RECORD = struct.Struct("<IQHHIIQQQQ")
ZIP64_END_RECORD_SIGNATURE = 0x06064B50
DIRECTORY_ENTRY = struct.Struct("<IHHHHHHIIIHHHHHII")
DIRECTORY_ENTRY_SIGNATURE = 0x02014B50
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_HEADER_SIGNATURE = 0x04034B50

EXTRA_BLOCK_HEADER = struct.Struct("<HH")
ZIP64_EXTRA_BLOCK_ID = 0x0001
# A ZIP64 extra block holds the real value of each saturated directory field, in this order.
ZIP64_FIELDS = (
    ("uncompressed_size", 0xFFFFFFFF, struct.Struct("<Q")),
    ("compressed_size", 0xFFFFFFFF, struct.Struct("<Q")),
    ("header_offset", 0xFFFFFFFF, struct.Struct("<Q")),
    ("disk", 0xFFFF, struct.Struct("<I")),
)

ENCRYPTED_FLAG = 0x1
STORED, DEFLATED = 0, 8


class Member(NamedTuple):
    disk: int
    header_offset: int  # of the local file header, from the start of its disk
    compressed_size: int
    method: int
    flags: int


def split_parts(zip_path: Path) -> list[Path]:
    """images.zip -> [images.z01, ..., images.zNN, images.zip] (disk order)."""
    zip_path = Path(zip_path)
    numbered_parts = sorted(zip_path.parent.glob(zip_path.stem + ".z[0-9][0-9]"))
    return numbered_parts + [zip_path]


class SplitZip:
    def __init__(self, zip_path: Path):
        self.parts = split_parts(zip_path)
        self.part_sizes = [part.stat().st_size for part in self.parts]
        self.part_starts = list(itertools.accumulate(self.part_sizes[:-1], initial=0))
        self._handles: dict[int, BinaryIO] = {}
        self.members = self._read_central_directory()

    def read(self, member: str) -> bytes:
        """The decompressed bytes of a member, named by its full path inside the archive."""
        disk, header_offset, compressed_size, method, flags = self.members[member]
        header_position = self._position(disk, header_offset)
        signature, *_, name_length, extra_length = LOCAL_HEADER.unpack(
            self._read_at(header_position, LOCAL_HEADER.size)
        )
        assert signature == LOCAL_HEADER_SIGNATURE, f"bad local header for {member}"
        if flags & ENCRYPTED_FLAG:
            raise ValueError(f"{member} is encrypted")
        data = self._read_at(header_position + LOCAL_HEADER.size + name_length + extra_length, compressed_size)
        if method == STORED:
            return data
        if method == DEFLATED:
            return zlib.decompressobj(-15).decompress(data)
        raise ValueError(f"unsupported compression method {method} for {member}")

    def basename_index(self) -> dict[str, str]:
        """File name -> full member path."""
        return {os.path.basename(member): member for member in self.members}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _position(self, disk: int, offset: int) -> int:
        """Offsets in a split archive count from the start of their disk; this is the offset over all parts."""
        return self.part_starts[disk] + offset

    def _read_at(self, position: int, length: int) -> bytes:
        """Reads from the parts as if they were one concatenated file (a read may span parts)."""
        data = bytearray()
        while length > 0:
            part = bisect.bisect_right(self.part_starts, position) - 1
            offset = position - self.part_starts[part]
            handle = self._handle(part)
            handle.seek(offset)
            chunk = handle.read(min(length, self.part_sizes[part] - offset))
            if not chunk:
                raise EOFError(f"read past end of {self.parts[part]}")
            data += chunk
            position += len(chunk)
            length -= len(chunk)
        return bytes(data)

    def _handle(self, part: int) -> BinaryIO:
        handle = self._handles.get(part)
        if handle is None:
            handle = self._handles[part] = open(self.parts[part], "rb")
        return handle

    def _read_central_directory(self) -> dict[str, Member]:
        disk, offset, size = self._locate_central_directory()
        directory = self._read_at(self._position(disk, offset), size)
        members = {}
        entry_start = 0
        while entry_start < len(directory):
            name, member, entry_length = _parse_directory_entry(directory, entry_start)
            if not name.endswith("/"):
                members[name] = member
            entry_start += entry_length
        return members

    def _locate_central_directory(self) -> tuple[int, int, int]:
        """(disk, offset, size) of the central directory, from the end record in the last part."""
        last = len(self.parts) - 1
        tail_length = min(self.part_sizes[last], END_RECORD_SEARCH_LENGTH)
        tail = self._read_at(self.part_starts[last] + self.part_sizes[last] - tail_length, tail_length)
        end_record_at = tail.rfind(END_RECORD_SIGNATURE)
        if end_record_at < 0:
            raise ValueError("no end-of-central-directory record")
        _, _, disk, _, _, size, offset, _ = END_RECORD.unpack_from(tail, end_record_at)

        locator_at = tail.rfind(ZIP64_LOCATOR_SIGNATURE, 0, end_record_at)
        if locator_at >= 0:
            _, record_disk, record_offset, _ = ZIP64_LOCATOR.unpack_from(tail, locator_at)
            record = self._read_at(self._position(record_disk, record_offset), ZIP64_END_RECORD.size)
            signature, _, _, _, _, disk, _, _, size, offset = ZIP64_END_RECORD.unpack(record)
            assert signature == ZIP64_END_RECORD_SIGNATURE, "bad zip64 end-of-central-directory record"
        return disk, offset, size


def _parse_directory_entry(directory: bytes, start: int) -> tuple[str, Member, int]:
    """(member name, member, length of the entry) of the central directory entry at start."""
    (
        signature,
        _,
        _,
        flags,
        method,
        _,
        _,
        _,
        compressed_size,
        uncompressed_size,
        name_length,
        extra_length,
        comment_length,
        disk,
        _,
        _,
        header_offset,
    ) = DIRECTORY_ENTRY.unpack_from(directory, start)
    assert signature == DIRECTORY_ENTRY_SIGNATURE, f"bad central directory entry at {start}"
    name_start = start + DIRECTORY_ENTRY.size
    extra_start = name_start + name_length
    name = directory[name_start:extra_start].decode("utf-8", "replace")
    fields = {
        "uncompressed_size": uncompressed_size,
        "compressed_size": compressed_size,
        "header_offset": header_offset,
        "disk": disk,
    }
    _apply_zip64_extra(directory[extra_start : extra_start + extra_length], fields)
    member = Member(fields["disk"], fields["header_offset"], fields["compressed_size"], method, flags)
    return name, member, DIRECTORY_ENTRY.size + name_length + extra_length + comment_length


def _apply_zip64_extra(extra: bytes, fields: dict[str, int]) -> None:
    """Replaces the saturated fields with their real values from the ZIP64 extra block, if there is one."""
    block_start = 0
    while block_start + EXTRA_BLOCK_HEADER.size <= len(extra):
        block_id, block_length = EXTRA_BLOCK_HEADER.unpack_from(extra, block_start)
        values_start = block_start + EXTRA_BLOCK_HEADER.size
        if block_id == ZIP64_EXTRA_BLOCK_ID:
            values = extra[values_start : values_start + block_length]
            value_offset = 0
            for name, saturated, value_format in ZIP64_FIELDS:
                if fields[name] == saturated:
                    fields[name] = value_format.unpack_from(values, value_offset)[0]
                    value_offset += value_format.size
        block_start = values_start + block_length
