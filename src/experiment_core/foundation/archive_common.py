"""共享归档压缩工具。

本模块只负责与具体业务无关的归档原语：
- 计算文件哈希
- 以 `tar.zst` 形式打包一组相对路径
- 安全解压 `tar.zst`
- 复制一组相对路径到临时发布目录
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import zstandard as zstd


@dataclass(frozen=True)
class PackedArchive:
    """描述一个已经写出的压缩包。"""

    archive_name: str
    archive_path: Path
    members: tuple[str, ...]
    original_size_bytes: int
    archive_size_bytes: int
    sha256_hex: str


def sha256_file(path: str | Path) -> str:
    """计算单个文件的 SHA-256。"""
    target = Path(path)
    digest = sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_tar_zst(
    *,
    source_root: str | Path,
    members: list[str],
    archive_path: str | Path,
    compression_level: int = 10,
) -> PackedArchive:
    """把一组相对路径写成 `tar.zst` 压缩包。"""
    root = Path(source_root)
    target = Path(archive_path)
    normalized_members = tuple(sorted(dict.fromkeys(_normalize_relative_path(item) for item in members)))
    target.parent.mkdir(parents=True, exist_ok=True)

    if not normalized_members:
        if target.exists():
            target.unlink()
        target.write_bytes(b"")
        return PackedArchive(
            archive_name=target.name,
            archive_path=target,
            members=(),
            original_size_bytes=0,
            archive_size_bytes=0,
            sha256_hex=sha256_file(target),
        )

    original_size_bytes = sum((root / member).stat().st_size for member in normalized_members)
    with target.open("wb") as raw_handle:
        cctx = zstd.ZstdCompressor(level=compression_level)
        with cctx.stream_writer(raw_handle) as compressed_handle:
            with tarfile.open(fileobj=compressed_handle, mode="w|") as tar_handle:
                for member in normalized_members:
                    source_path = root / member
                    if not source_path.exists():
                        raise FileNotFoundError(f"Archive member does not exist: {source_path}")
                    tar_handle.add(source_path, arcname=member, recursive=True)

    return PackedArchive(
        archive_name=target.name,
        archive_path=target,
        members=normalized_members,
        original_size_bytes=original_size_bytes,
        archive_size_bytes=target.stat().st_size,
        sha256_hex=sha256_file(target),
    )


def extract_tar_zst(
    *,
    archive_path: str | Path,
    target_root: str | Path,
) -> tuple[str, ...]:
    """安全解压 `tar.zst` 到目标目录。"""
    archive = Path(archive_path)
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)

    extracted_members: list[str] = []
    with archive.open("rb") as raw_handle:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw_handle) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar_handle:
                for member in tar_handle:
                    if member.name in {"", ".", "./"}:
                        continue
                    relative_path = _normalize_relative_path(member.name)
                    destination = root / relative_path
                    _ensure_within_root(root, destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    extracted = tar_handle.extractfile(member)
                    if extracted is None:
                        continue
                    with destination.open("wb") as output_handle:
                        shutil.copyfileobj(extracted, output_handle)
                    extracted_members.append(relative_path)
    return tuple(extracted_members)


def copy_relative_files(
    *,
    source_root: str | Path,
    target_root: str | Path,
    members: list[str],
) -> tuple[str, ...]:
    """把一组相对路径复制到目标根目录。"""
    source = Path(source_root)
    target = Path(target_root)
    copied: list[str] = []
    for member in sorted(dict.fromkeys(_normalize_relative_path(item) for item in members)):
        source_path = source / member
        target_path = target / member
        _ensure_within_root(target, target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append(member)
    return tuple(copied)


def write_text_file(path: str | Path, content: str) -> None:
    """统一写 UTF-8 文本文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _normalize_relative_path(value: str) -> str:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute():
        raise ValueError(f"Archive member must be relative: {value}")
    normalized = relative.as_posix().lstrip("./")
    if normalized in {"", "."}:
        raise ValueError("Archive member must not be empty.")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"Archive member must not escape root: {value}")
    return normalized


def _ensure_within_root(root: Path, candidate: Path) -> None:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - 防御性安全校验
        raise ValueError(f"Path escapes target root: {candidate}") from exc
