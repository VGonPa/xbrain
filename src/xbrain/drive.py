"""Google Drive-backed library synchronization.

XBrain's core pipeline is deliberately file-oriented, so Drive support uses a
local working cache and synchronizes it with a user-selected Drive folder.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from xbrain.config import Config

_FOLDER_MIME = "application/vnd.google-apps.folder"
_SESSION_DEPTH: ContextVar[int] = ContextVar("xbrain_drive_session_depth", default=0)
_IGNORED_REMOTE_PATHS: set[tuple[str, ...]] = {
    ("data", "media"),
    ("vault", "x-knowledge"),
}


@dataclass(frozen=True)
class DriveFolder:
    id: str
    name: str
    modified_time: str | None = None


@dataclass(frozen=True)
class SyncReport:
    downloaded: int = 0
    uploaded: int = 0
    folders_created: int = 0
    trashed: int = 0


class DriveClient:
    """Small wrapper around Drive API v3 operations XBrain needs."""

    def __init__(self, service: Any):
        self.service = service

    def list_folders(self, *, page_size: int = 50) -> list[DriveFolder]:
        rows: list[DriveFolder] = []
        page_token: str | None = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"mimeType='{_FOLDER_MIME}' and trashed=false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, modifiedTime)",
                    pageSize=page_size,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            rows.extend(
                DriveFolder(
                    id=item["id"],
                    name=item["name"],
                    modified_time=item.get("modifiedTime"),
                )
                for item in response.get("files", [])
            )
            page_token = response.get("nextPageToken")
            if not page_token:
                return rows

    def create_folder(self, name: str, *, parent_id: str | None = None) -> DriveFolder:
        body: dict[str, Any] = {"name": name, "mimeType": _FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        item = (
            self.service.files()
            .create(
                body=body,
                fields="id, name, modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )
        return DriveFolder(id=item["id"], name=item["name"], modified_time=item.get("modifiedTime"))

    def sync_down(self, root_folder_id: str, cache_dir: Path) -> SyncReport:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return self._download_folder(root_folder_id, cache_dir, relative_path=())

    def sync_up(self, root_folder_id: str, cache_dir: Path) -> SyncReport:
        library_roots = (cache_dir / "data", cache_dir / "vault")
        has_library_file = cache_dir.exists() and any(
            path.is_file() for root in library_roots if root.is_dir() for path in root.rglob("*")
        )
        if not has_library_file:
            raise RuntimeError("refusing to mirror an empty Drive cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return self._upload_folder(root_folder_id, cache_dir, relative_path=())

    def remote_manifest(self, root_folder_id: str) -> str:
        """Return a stable fingerprint used to detect concurrent remote edits."""
        rows: list[tuple[str, str, str]] = []

        def walk(folder_id: str, relative_path: tuple[str, ...]) -> None:
            children = sorted(self._children(folder_id), key=lambda item: item["name"])
            for item in children:
                child_path = (*relative_path, item["name"])
                if _ignored_remote_path(child_path):
                    continue
                rows.append(
                    (
                        "/".join(child_path),
                        item["mimeType"],
                        item.get("md5Checksum", ""),
                    )
                )
                if item["mimeType"] == _FOLDER_MIME:
                    walk(item["id"], child_path)

        walk(root_folder_id, ())
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _download_folder(
        self, folder_id: str, destination: Path, *, relative_path: tuple[str, ...]
    ) -> SyncReport:
        report = SyncReport()
        remote_children = [
            item
            for item in self._children(folder_id)
            if not _ignored_remote_path((*relative_path, item["name"]))
        ]
        remote_names = {item["name"] for item in remote_children}
        for local in destination.iterdir() if destination.exists() else []:
            if local.name not in remote_names:
                if local.is_dir():
                    _remove_tree(local)
                else:
                    local.unlink()
        for item in remote_children:
            target = destination / item["name"]
            child_relative_path = (*relative_path, item["name"])
            if item["mimeType"] == _FOLDER_MIME:
                target.mkdir(parents=True, exist_ok=True)
                child = self._download_folder(item["id"], target, relative_path=child_relative_path)
                report = _merge_reports(report, child)
                continue
            if target.exists() and item.get("md5Checksum") == _md5(target):
                continue
            self._download_file(item["id"], target)
            report = _merge_reports(report, SyncReport(downloaded=1))
        return report

    def _upload_folder(
        self, folder_id: str, source: Path, *, relative_path: tuple[str, ...]
    ) -> SyncReport:
        report = SyncReport()
        remote_by_name = {item["name"]: item for item in self._children(folder_id)}
        local_names = {path.name for path in source.iterdir()} if source.exists() else set()
        for name, item in remote_by_name.items():
            if _ignored_remote_path((*relative_path, name)):
                continue
            if name not in local_names:
                report = _merge_reports(report, SyncReport(trashed=self._trash_tree(item)))
        for local in source.iterdir() if source.exists() else []:
            child_relative_path = (*relative_path, local.name)
            if _ignored_remote_path(child_relative_path):
                continue
            remote = remote_by_name.get(local.name)
            if local.is_dir():
                if remote is None or remote["mimeType"] != _FOLDER_MIME:
                    if remote is not None:
                        self._trash(remote["id"])
                    folder = self.create_folder(local.name, parent_id=folder_id)
                    report = _merge_reports(report, SyncReport(folders_created=1))
                    remote_id = folder.id
                else:
                    remote_id = remote["id"]
                child = self._upload_folder(remote_id, local, relative_path=child_relative_path)
                report = _merge_reports(report, child)
                continue
            if remote is not None and remote.get("md5Checksum") == _md5(local):
                continue
            self._upload_file(local, folder_id=folder_id, file_id=remote["id"] if remote else None)
            report = _merge_reports(report, SyncReport(uploaded=1))
        return report

    def _children(self, folder_id: str) -> list[dict[str, Any]]:
        query = f"'{folder_id}' in parents and trashed=false"
        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, md5Checksum)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            rows.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return rows

    def _download_file(self, file_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        with tmp.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        tmp.replace(destination)

    def _upload_file(self, path: Path, *, folder_id: str, file_id: str | None = None) -> None:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        if file_id:
            (
                self.service.files()
                .update(
                    fileId=file_id,
                    media_body=media,
                    fields="id, md5Checksum",
                    supportsAllDrives=True,
                )
                .execute()
            )
            return
        (
            self.service.files()
            .create(
                body={"name": path.name, "parents": [folder_id]},
                media_body=media,
                fields="id, md5Checksum",
                supportsAllDrives=True,
            )
            .execute()
        )

    def _trash(self, file_id: str) -> None:
        (
            self.service.files()
            .update(fileId=file_id, body={"trashed": True}, supportsAllDrives=True)
            .execute()
        )

    def _trash_tree(self, item: dict[str, Any]) -> int:
        if item["mimeType"] == _FOLDER_MIME:
            # Trashing a Drive folder also removes its descendants from normal
            # listings. Do it at the parent level so pruning snapshot folders
            # does not require one API write per tiny JSON/YAML file.
            self._trash(item["id"])
            return 1
        self._trash(item["id"])
        return 1


def authenticate(cfg: Config) -> DriveClient:
    creds = _credentials(cfg)
    service = build("drive", "v3", credentials=creds)
    return DriveClient(service)


def login(cfg: Config, *, port: int = 8766, open_browser: bool = False) -> None:
    _credentials(cfg, interactive=True, port=port, open_browser=open_browser)


def drive_sync_down(cfg: Config) -> SyncReport:
    if not cfg.drive_enabled:
        return SyncReport()
    return authenticate(cfg).sync_down(cfg.drive_root_folder_id, cfg.drive_cache_dir)


def drive_sync_up(cfg: Config) -> SyncReport:
    if not cfg.drive_enabled:
        return SyncReport()
    return authenticate(cfg).sync_up(cfg.drive_root_folder_id, cfg.drive_cache_dir)


@contextmanager
def drive_write_session(cfg: Config) -> Iterator[None]:
    """Sync around a successful write and reject concurrent remote changes."""
    if not cfg.drive_enabled:
        yield
        return
    depth = _SESSION_DEPTH.get()
    if depth:
        token = _SESSION_DEPTH.set(depth + 1)
        try:
            yield
        finally:
            _SESSION_DEPTH.reset(token)
        return
    token = _SESSION_DEPTH.set(1)
    try:
        client = authenticate(cfg)
        remote_before = client.remote_manifest(cfg.drive_root_folder_id)
        client.sync_down(cfg.drive_root_folder_id, cfg.drive_cache_dir)
        yield
        remote_after = client.remote_manifest(cfg.drive_root_folder_id)
        if remote_after != remote_before:
            raise RuntimeError(
                "Google Drive changed during this command; local changes were not uploaded"
            )
        client.sync_up(cfg.drive_root_folder_id, cfg.drive_cache_dir)
    finally:
        _SESSION_DEPTH.reset(token)


def _credentials(
    cfg: Config,
    *,
    interactive: bool = False,
    port: int = 8766,
    open_browser: bool = False,
) -> Credentials:
    creds = None
    if cfg.drive_token_path.exists():
        creds = Credentials.from_authorized_user_file(str(cfg.drive_token_path), cfg.drive_scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif interactive:
        if not cfg.drive_credentials_path.exists():
            raise FileNotFoundError(
                f"Google OAuth client file not found: {cfg.drive_credentials_path}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(cfg.drive_credentials_path), cfg.drive_scopes
        )
        creds = flow.run_local_server(port=port, open_browser=open_browser)
    else:
        raise RuntimeError("Google Drive is not authenticated. Run `xbrain drive login`.")
    cfg.drive_token_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.drive_token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _merge_reports(a: SyncReport, b: SyncReport) -> SyncReport:
    return SyncReport(
        downloaded=a.downloaded + b.downloaded,
        uploaded=a.uploaded + b.uploaded,
        folders_created=a.folders_created + b.folders_created,
        trashed=a.trashed + b.trashed,
    )


def _ignored_remote_path(relative_path: tuple[str, ...]) -> bool:
    return relative_path in _IGNORED_REMOTE_PATHS


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()
