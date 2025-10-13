from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import requests


def is_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def ensure_local_path(source: str, cache_root: Path, force_dir: bool = False) -> Path:
    source = source.strip()
    if not source:
        raise ValueError("Empty data source")

    cache_root.mkdir(parents=True, exist_ok=True)
    candidate = Path(source)

    if candidate.exists():
        if force_dir and candidate.is_file():
            if candidate.suffix.lower() == ".zip":
                return _ensure_archive_extracted(candidate, cache_root / "archives")
            raise ValueError(f"Duong dan '{candidate}' la file. Can thu muc du lieu.")
        return candidate

    if not is_url(source):
        raise FileNotFoundError(f"Khong tim thay duong dan '{source}'")

    return _download_remote_source(source, cache_root / "remote", force_dir=force_dir)


def _ensure_archive_extracted(archive_path: Path, target_root: Path) -> Path:
    stat = archive_path.stat()
    cache_key = _hash_key(f"{archive_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
    extract_root = target_root / cache_key
    marker = extract_root / ".complete"
    if marker.exists():
        return _pick_extracted_root(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)
    marker.write_text("ok", encoding="utf-8")
    return _pick_extracted_root(extract_root)


def _pick_extracted_root(root: Path) -> Path:
    subdirs = [child for child in root.iterdir() if child.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return root


def _download_remote_source(url: str, target_root: Path, force_dir: bool) -> Path:
    cache_key = _hash_key(url)
    resource_root = target_root / cache_key
    download_path = resource_root / "download"
    resource_root.mkdir(parents=True, exist_ok=True)

    if not download_path.exists():
        temp_path = download_path.with_suffix(".tmp")
        _stream_download(url, temp_path)
        temp_path.rename(download_path)

    if force_dir:
        if download_path.is_file() and download_path.suffix.lower() == ".zip":
            extract_dir = resource_root / "extracted"
            marker = extract_dir / ".complete"
            if marker.exists():
                return _pick_extracted_root(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(download_path) as archive:
                archive.extractall(extract_dir)
            marker.write_text("ok", encoding="utf-8")
            return _pick_extracted_root(extract_dir)
        if download_path.is_dir():
            return download_path
        raise ValueError(f"Tep tai ve tu '{url}' khong phai thu muc hop le.")

    return download_path


def _stream_download(url: str, destination: Path) -> None:
    session = requests.Session()
    file_id = _extract_drive_file_id(url)
    if file_id:
        _download_google_drive_file(session, file_id, destination)
        return

    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()
    _write_response_content(response, destination)


def _download_google_drive_file(session: requests.Session, file_id: str, destination: Path) -> None:
    base_url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(base_url, params=params, stream=True, timeout=60)
    token = _extract_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(base_url, params=params, stream=True, timeout=60)
    response.raise_for_status()
    _write_response_content(response, destination)


def _extract_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _write_response_content(response: requests.Response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            file_obj.write(chunk)


def _extract_drive_file_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "docs.google.com"}:
        return None

    query_params = parse_qs(parsed.query)
    if "id" in query_params:
        return query_params["id"][0]

    match = re.search(r"/d/([^/]+)/", parsed.path)
    if match:
        return match.group(1)

    return None


def _hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


class DatasetRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.entries: List[str] = []
        self._load()
        self.prune_missing()

    def add(self, path: Path) -> bool:
        resolved = str(Path(path).resolve())
        if resolved not in self.entries:
            self.entries.append(resolved)
            self._save()
            return True
        return False

    def remove(self, path: Path) -> bool:
        resolved = str(Path(path).resolve())
        try:
            self.entries.remove(resolved)
        except ValueError:
            return False
        self._save()
        return True

    def prune_missing(self) -> None:
        initial_len = len(self.entries)
        self.entries = [entry for entry in self.entries if Path(entry).exists()]
        if len(self.entries) != initial_len:
            self._save()

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, list):
            self.entries = [str(item) for item in data if isinstance(item, str)]

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")


__all__ = ["ensure_local_path", "is_url", "DatasetRegistry"]
