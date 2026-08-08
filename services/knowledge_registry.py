import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


REGISTRY_FORMAT_VERSION = "knowledge-registry-v1"


class SourceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    REMOVED = "REMOVED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SourceSnapshot:
    logical_source_id: str
    source_document_id: str
    source_type: str
    source_file: str
    title: str
    content_hash: str
    version_id: str
    page_count: int
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    file_size: int | None = None
    extractable_page_count: int | None = None
    status: SourceStatus = SourceStatus.ACTIVE
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KnowledgeUpdatePlan:
    added_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    modified_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    removed_sources: tuple[dict, ...] = field(default_factory=tuple)
    unchanged_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    restored_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    invalid_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    renamed_sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)
    duplicate_aliases: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def counts(self):
        return {
            "added": len(self.added_sources),
            "modified": len(self.modified_sources),
            "removed": len(self.removed_sources),
            "unchanged": len(self.unchanged_sources),
            "restored": len(self.restored_sources),
            "invalid": len(self.invalid_sources),
            "renamed": len(self.renamed_sources),
            "duplicate_aliases": len(self.duplicate_aliases),
        }


class KnowledgeChangeDetector:
    """Bandingkan registry history dengan snapshot source aktif secara deterministik."""

    def detect(self, previous_records, current_snapshots):
        previous = [dict(item) for item in previous_records]
        active = {
            item["logical_source_id"]: item
            for item in previous
            if item.get("status") == SourceStatus.ACTIVE.value
        }
        historical_by_hash = {}
        for item in previous:
            historical_by_hash.setdefault(
                (item.get("source_type"), item.get("content_hash")), []
            ).append(item)

        added, modified, unchanged, restored, invalid, renamed = [], [], [], [], [], []
        duplicate_aliases = []
        matched_logical = set()
        unique_current = []
        current_by_hash = {}
        for snapshot in sorted(current_snapshots, key=lambda item: (item.source_type, item.source_file.casefold(), item.source_file)):
            if snapshot.status == SourceStatus.INVALID:
                invalid.append(snapshot)
                matched_logical.add(snapshot.logical_source_id)
                continue
            key = (snapshot.source_type, snapshot.content_hash)
            if key in current_by_hash:
                canonical = current_by_hash[key]
                duplicate_aliases.append((snapshot.source_file, canonical.source_file))
                continue
            current_by_hash[key] = snapshot
            unique_current.append(snapshot)

        for snapshot in unique_current:
            prior = active.get(snapshot.logical_source_id)
            if prior:
                matched_logical.add(snapshot.logical_source_id)
                if prior.get("content_hash") == snapshot.content_hash:
                    unchanged.append(snapshot)
                else:
                    modified.append(snapshot)
                continue

            history = historical_by_hash.get((snapshot.source_type, snapshot.content_hash), [])
            reusable = next(
                (item for item in history if item.get("status") in {
                    SourceStatus.ACTIVE.value,
                    SourceStatus.REMOVED.value,
                    SourceStatus.SUPERSEDED.value,
                }),
                None,
            )
            if reusable:
                rebound = replace(
                    snapshot,
                    logical_source_id=reusable["logical_source_id"],
                    aliases=tuple(sorted(set(snapshot.aliases) | {reusable.get("source_file", "")} - {""})),
                )
                matched_logical.add(rebound.logical_source_id)
                if reusable.get("status") in {
                    SourceStatus.REMOVED.value, SourceStatus.SUPERSEDED.value
                }:
                    restored.append(rebound)
                else:
                    unchanged.append(rebound)
                    renamed.append(rebound)
            else:
                matched_logical.add(snapshot.logical_source_id)
                added.append(snapshot)

        removed = tuple(
            item for logical_id, item in sorted(active.items())
            if logical_id not in matched_logical
        )
        return KnowledgeUpdatePlan(
            tuple(added), tuple(modified), removed, tuple(unchanged), tuple(restored),
            tuple(invalid), tuple(renamed), tuple(duplicate_aliases),
        )


class KnowledgeRegistry:
    """Registry metadata source/version; file sumber tetap source of truth."""

    def __init__(self, path):
        self.path = Path(path)
        self.corrupt = False

    def load(self):
        self.corrupt = False
        if not self.path.exists():
            return {"registry_format_version": REGISTRY_FORMAT_VERSION, "sources": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("registry_format_version") != REGISTRY_FORMAT_VERSION or not isinstance(payload.get("sources"), list):
                raise ValueError("format registry tidak dikenal")
            return payload
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            self.corrupt = True
            return {"registry_format_version": REGISTRY_FORMAT_VERSION, "sources": []}

    def plan(self, snapshots):
        return KnowledgeChangeDetector().detect(self.load().get("sources", []), snapshots)

    def commit(self, snapshots, plan=None):
        payload = self.load()
        records = [dict(item) for item in payload.get("sources", [])]
        plan = plan or KnowledgeChangeDetector().detect(records, snapshots)
        now = datetime.now(timezone.utc).isoformat()

        superseded_ids = {item.logical_source_id for item in plan.modified_sources}
        removed_ids = {item["logical_source_id"] for item in plan.removed_sources}
        for record in records:
            if record.get("status") == SourceStatus.ACTIVE.value:
                if record.get("logical_source_id") in superseded_ids:
                    record["status"] = SourceStatus.SUPERSEDED.value
                    record["updated_at"] = now
                elif record.get("logical_source_id") in removed_ids:
                    record["status"] = SourceStatus.REMOVED.value
                    record["updated_at"] = now

        active_snapshots = (
            plan.added_sources + plan.modified_sources + plan.unchanged_sources
            + plan.restored_sources + plan.invalid_sources
        )
        current_by_version = {item.version_id: item for item in active_snapshots}
        for version_id, snapshot in current_by_version.items():
            existing = next((item for item in records if item.get("version_id") == version_id), None)
            serialized = self._serialize_snapshot(snapshot, now, existing)
            if existing is None:
                records.append(serialized)
            else:
                existing.update(serialized)

        payload = {
            "registry_format_version": REGISTRY_FORMAT_VERSION,
            "updated_at": now,
            "sources": sorted(records, key=lambda item: (item["logical_source_id"], item["ingested_at"], item["version_id"])),
        }
        self._atomic_write(payload)
        return payload

    def status(self):
        payload = self.load()
        counts = {status.value: 0 for status in SourceStatus}
        for item in payload.get("sources", []):
            counts[item.get("status", SourceStatus.INVALID.value)] = counts.get(item.get("status"), 0) + 1
        return {
            "path": str(self.path),
            "format": payload["registry_format_version"],
            "corrupt": self.corrupt,
            "source_versions": len(payload.get("sources", [])),
            "status_counts": counts,
            "sources": [
                {
                    "logical_source_id": item.get("logical_source_id"),
                    "active_version": item.get("version_id") if item.get("status") == SourceStatus.ACTIVE.value else None,
                    "hash_prefix": str(item.get("content_hash", ""))[:12],
                    "source_file": item.get("source_file"),
                    "status": item.get("status"),
                }
                for item in payload.get("sources", [])
            ],
        }

    @classmethod
    def build_snapshots(cls, parent_documents, chunks, invalid_entries=()):
        parents_by_source = {}
        for document in parent_documents:
            metadata = document.metadata
            source_file = str(metadata.get("source_file") or metadata.get("source") or "unknown")
            source_type = "pdf" if metadata.get("type") == "pdf" else "csv"
            key = (source_type, source_file, metadata.get("source_document_id") if source_type == "pdf" else source_file)
            parents_by_source.setdefault(key, []).append(document)
        chunks_by_parent = {}
        for chunk in chunks:
            chunks_by_parent.setdefault(chunk.parent_document_id, []).append(chunk.id)

        snapshots = []
        for (source_type, source_file, source_identity), parents in sorted(parents_by_source.items()):
            parent_ids = tuple(sorted(item.id for item in parents))
            chunk_ids = tuple(sorted(chunk_id for parent_id in parent_ids for chunk_id in chunks_by_parent.get(parent_id, [])))
            if source_type == "pdf":
                content_hash = str(parents[0].metadata["content_hash"])
                source_document_id = str(parents[0].metadata["source_document_id"])
                title = str(parents[0].metadata.get("document_title") or source_file)
                page_count = max(int(item.metadata.get("page_end", 0)) for item in parents)
                logical_id = f"pdf-{cls._slug(Path(source_file).stem)}"
                file_size = parents[0].metadata.get("file_size")
            else:
                content_hash = cls._source_content_hash(parents)
                logical_id = f"csv-{cls._slug(Path(source_file).stem)}"
                source_document_id = f"{logical_id}-{content_hash[:12]}"
                title = Path(source_file).stem
                page_count = 0
                file_size = None
            snapshots.append(SourceSnapshot(
                logical_source_id=logical_id,
                source_document_id=source_document_id,
                source_type=source_type,
                source_file=source_file,
                title=title,
                content_hash=content_hash,
                version_id=f"sha256-{content_hash}",
                page_count=page_count,
                parent_ids=parent_ids,
                chunk_ids=chunk_ids,
                file_size=file_size,
                extractable_page_count=len(parents) if source_type == "pdf" else None,
            ))
        snapshots.extend(invalid_entries)
        return tuple(snapshots)

    @staticmethod
    def _source_content_hash(documents):
        digest = hashlib.sha256()
        for document in sorted(documents, key=lambda item: item.id):
            digest.update(document.id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.content.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _serialize_snapshot(snapshot, now, existing):
        payload = asdict(snapshot)
        payload["status"] = snapshot.status.value
        payload["parent_ids"] = list(snapshot.parent_ids)
        payload["chunk_ids"] = list(snapshot.chunk_ids)
        payload["aliases"] = list(snapshot.aliases)
        payload["ingested_at"] = existing.get("ingested_at") if existing else now
        payload["updated_at"] = now
        return payload

    def _atomic_write(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            json.loads(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _slug(value):
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-") or "source"
