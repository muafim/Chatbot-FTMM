import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from domain.models import Document
from services.pdf_document_loader import PDFDocumentLoader, PDFLoadError


logger = logging.getLogger(__name__)
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class PDFIngestionReport:
    discovered_files: int = 0
    loaded_files: int = 0
    duplicate_files: int = 0
    failed_files: int = 0
    extracted_pages: int = 0
    parent_documents: int = 0
    extraction_cache_hits: int = 0
    elapsed_seconds: float = 0.0


class PDFKnowledgeRepository:
    """Direktori PDF lokal dengan manifest ekstraksi yang dapat dibangun ulang."""

    def __init__(self, directory, manifest_path, enabled=True, max_file_size_mb=50):
        self.directory = Path(directory)
        self.manifest_path = Path(manifest_path)
        self.enabled = enabled
        self.loader = PDFDocumentLoader(max_file_size_mb=max_file_size_mb)
        self.errors = []
        self.warnings = []
        self.report = PDFIngestionReport()
        self.last_manifest_files = []

    def load_documents(self, dry_run=False):
        started_at = time.perf_counter()
        self.errors = []
        self.warnings = []
        if not self.enabled:
            self.report = PDFIngestionReport(elapsed_seconds=time.perf_counter() - started_at)
            return []
        if not self.directory.exists():
            self.warnings.append(f"Direktori PDF belum tersedia: {self.directory}.")
            self.report = PDFIngestionReport(elapsed_seconds=time.perf_counter() - started_at)
            return []
        if not self.directory.is_dir():
            self.errors.append(f"Path sumber PDF bukan direktori: {self.directory}.")
            self.report = PDFIngestionReport(failed_files=1, elapsed_seconds=time.perf_counter() - started_at)
            return []

        paths = self.loader.discover(self.directory)
        previous = self._read_manifest()
        previous_by_hash = {
            item.get("content_hash"): item
            for item in previous.get("files", [])
            if item.get("status") == "loaded" and item.get("content_hash")
        }
        documents = []
        manifest_files = []
        seen_hashes = {}
        loaded = duplicates = failed = pages = cache_hits = 0

        for path in paths:
            content_hash = None
            try:
                file_size = path.stat().st_size
                if file_size == 0:
                    raise PDFLoadError("empty_file", f"File PDF kosong: {path.name}.")
                if file_size > self.loader.max_file_size_bytes:
                    raise PDFLoadError(
                        "too_large", f"File PDF melewati batas ukuran: {path.name}."
                    )
                content_hash = self.loader.hash_file(path)
                if content_hash in seen_hashes:
                    duplicates += 1
                    manifest_files.append({
                        "source_file": path.name,
                        "content_hash": content_hash,
                        "file_size": file_size,
                        "status": "duplicate",
                        "duplicate_of": seen_hashes[content_hash],
                    })
                    self.warnings.append(
                        f"PDF duplikat dilewati: {path.name} (sama dengan {seen_hashes[content_hash]})."
                    )
                    continue

                cached = previous_by_hash.get(content_hash)
                if cached and cached.get("source_file") == path.name and cached.get("documents"):
                    try:
                        if cached.get("documents_digest") != self._documents_digest(cached["documents"]):
                            raise ValueError("digest dokumen manifest tidak cocok")
                        restored = [self._deserialize_document(item) for item in cached["documents"]]
                        if any(
                            item.metadata.get("content_hash") != content_hash
                            or item.metadata.get("source_file") != path.name
                            for item in restored
                        ):
                            raise ValueError("metadata dokumen manifest tidak cocok")
                    except (KeyError, TypeError, ValueError):
                        logger.warning(
                            "Entry manifest PDF invalid untuk %s; mengekstrak ulang.", path.name
                        )
                    else:
                        documents.extend(restored)
                        manifest_files.append(cached)
                        seen_hashes[content_hash] = path.name
                        loaded += 1
                        cache_hits += 1
                        pages += len(restored)
                        continue

                result = self.loader.load(path, allowed_root=self.directory)
                if result.content_hash != content_hash or self.loader.hash_file(path) != content_hash:
                    raise PDFLoadError(
                        "changed_during_ingestion",
                        f"PDF berubah saat ingestion dan ditunda: {path.name}.",
                    )
                extracted = result.to_documents()
                documents.extend(extracted)
                seen_hashes[content_hash] = path.name
                loaded += 1
                pages += len(extracted)
                self.warnings.extend(result.warnings)
                serialized_documents = [self._serialize_document(item) for item in extracted]
                manifest_files.append({
                    "source_file": result.source_file,
                    "filename": result.source_file,
                    "source_document_id": result.source_document_id,
                    "document_title": result.document_title,
                    "content_hash": result.content_hash,
                    "sha256": result.content_hash,
                    "file_size": result.file_size,
                    "pages": result.page_count,
                    "page_count": result.page_count,
                    "extracted_pages": len(result.pages),
                    "extractable_pages": len(result.pages),
                    "parent_count": len(extracted),
                    "chunk_count": None,
                    "status": "loaded",
                    "documents_digest": self._documents_digest(serialized_documents),
                    "documents": serialized_documents,
                })
            except (OSError, PDFLoadError) as exc:
                failed += 1
                status = getattr(exc, "status", "read_error")
                message = str(exc)
                self.errors.append(message)
                manifest_files.append({
                    "source_file": path.name,
                    "content_hash": content_hash,
                    "status": status,
                    "error": message,
                })

        if not dry_run:
            self._write_manifest({
                "manifest_version": MANIFEST_VERSION,
                "source_directory": str(self.directory.resolve()),
                "files": manifest_files,
            })
        self.last_manifest_files = list(manifest_files)
        self.report = PDFIngestionReport(
            discovered_files=len(paths),
            loaded_files=loaded,
            duplicate_files=duplicates,
            failed_files=failed,
            extracted_pages=pages,
            parent_documents=len(documents),
            extraction_cache_hits=cache_hits,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return documents

    def _read_manifest(self):
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload.get("manifest_version") != MANIFEST_VERSION:
                return {}
            return payload
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning("Manifest PDF corrupt; membangun ulang dari direktori sumber.")
            return {}

    def record_chunk_counts(self, chunks):
        """Lengkapi tracking manifest setelah chunker existing selesai bekerja."""
        if not self.enabled or not self.manifest_path.exists():
            return
        counts = {}
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {})
            if metadata.get("type") != "pdf":
                continue
            source_id = metadata.get("source_document_id")
            if source_id:
                counts[source_id] = counts.get(source_id, 0) + 1
        payload = self._read_manifest()
        if not payload:
            return
        changed = False
        for item in payload.get("files", []):
            if item.get("status") != "loaded":
                continue
            value = counts.get(item.get("source_document_id"), 0)
            if item.get("chunk_count") != value:
                item["chunk_count"] = value
                changed = True
        if changed:
            self._write_manifest(payload)

    def invalid_source_snapshots(self):
        from services.knowledge_registry import SourceSnapshot, SourceStatus

        snapshots = []
        files = self.last_manifest_files or self._read_manifest().get("files", [])
        for item in files:
            if item.get("status") in {"loaded", "duplicate"}:
                continue
            source_file = str(item.get("source_file") or "invalid.pdf")
            digest = item.get("content_hash") or hashlib.sha256(
                f"{source_file}:{item.get('status')}".encode("utf-8")
            ).hexdigest()
            logical_id = f"pdf-{self.loader._slug(Path(source_file).stem)}"
            snapshots.append(SourceSnapshot(
                logical_source_id=logical_id,
                source_document_id=f"{logical_id}-{digest[:12]}",
                source_type="pdf",
                source_file=source_file,
                title=Path(source_file).stem,
                content_hash=digest,
                version_id=f"sha256-{digest}",
                page_count=0,
                status=SourceStatus.INVALID,
            ))
        return tuple(snapshots)

    def _write_manifest(self, payload):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.manifest_path.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(temporary_path, self.manifest_path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _serialize_document(document):
        return {"id": document.id, "content": document.content, "metadata": document.metadata}

    @staticmethod
    def _deserialize_document(payload):
        return Document(
            id=str(payload["id"]),
            content=str(payload["content"]),
            metadata=dict(payload["metadata"]),
        )

    @staticmethod
    def _documents_digest(documents):
        encoded = json.dumps(
            documents, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
