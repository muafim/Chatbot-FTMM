import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from domain.models import Document


class PDFLoadError(RuntimeError):
    """Error PDF terkontrol yang aman ditampilkan tanpa traceback internal."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    character_count: int


@dataclass(frozen=True)
class PDFLoadResult:
    source_path: Path
    source_document_id: str
    source_file: str
    document_title: str
    content_hash: str
    file_size: int
    page_count: int
    pages: tuple[ExtractedPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_documents(self):
        documents = []
        for page in self.pages:
            parent_id = f"{self.source_document_id}::page-{page.page_number:04d}"
            documents.append(
                Document(
                    id=parent_id,
                    content=(
                        "Jenis: Dokumen PDF FTMM\n"
                        f"Judul Dokumen: {self.document_title}\n"
                        f"Halaman: {page.page_number}\n"
                        f"Isi: {page.text}"
                    ),
                    metadata={
                        "type": "pdf",
                        "source_type": "pdf",
                        "source": self.source_file,
                        "source_file": self.source_file,
                        "name": self.document_title,
                        "title": self.document_title,
                        "document_title": self.document_title,
                        "source_document_id": self.source_document_id,
                        "page_start": page.page_number,
                        "page_end": page.page_number,
                        "content_hash": self.content_hash,
                        "file_size": self.file_size,
                    },
                )
            )
        return documents


class PDFDocumentLoader:
    """Ekstraksi PDF teks lokal per halaman, tanpa OCR dan tanpa network call."""

    def __init__(self, max_file_size_mb=50):
        if max_file_size_mb <= 0:
            raise ValueError("PDF_MAX_FILE_SIZE_MB harus lebih besar dari nol.")
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def load(self, path, allowed_root=None):
        source_path = Path(path)
        if not source_path.exists() or not source_path.is_file():
            raise PDFLoadError("missing", f"File PDF tidak ditemukan: {source_path.name}.")
        if source_path.suffix.casefold() != ".pdf":
            raise PDFLoadError("invalid_extension", "Hanya file berekstensi .pdf yang didukung.")

        resolved_path = source_path.resolve()
        if allowed_root is not None:
            resolved_root = Path(allowed_root).resolve()
            if resolved_path.parent != resolved_root:
                raise PDFLoadError(
                    "outside_source_directory",
                    "File PDF harus berada langsung di direktori sumber yang dikonfigurasi.",
                )

        file_size = resolved_path.stat().st_size
        if file_size == 0:
            raise PDFLoadError("empty_file", f"File PDF kosong: {source_path.name}.")
        if file_size > self.max_file_size_bytes:
            raise PDFLoadError(
                "too_large",
                f"File PDF melewati batas ukuran: {source_path.name}.",
            )

        content_hash = self.hash_file(resolved_path)
        try:
            reader = PdfReader(str(resolved_path), strict=False)
            if reader.is_encrypted:
                raise PDFLoadError(
                    "encrypted", f"PDF terenkripsi tidak didukung: {source_path.name}."
                )
            page_count = len(reader.pages)
            title = self._document_title(reader, source_path)
            pages = []
            empty_pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = self.normalize_text(page.extract_text() or "")
                except Exception as exc:
                    raise PDFLoadError(
                        "page_extraction_failed",
                        f"Ekstraksi halaman {page_number} gagal pada {source_path.name} "
                        f"({type(exc).__name__}).",
                    ) from exc
                if not text:
                    empty_pages.append(page_number)
                    continue
                pages.append(ExtractedPage(page_number, text, len(text)))
        except PDFLoadError:
            raise
        except (OSError, PdfReadError, ValueError, TypeError) as exc:
            raise PDFLoadError(
                "corrupt", f"PDF tidak dapat dibaca: {source_path.name} ({type(exc).__name__})."
            ) from exc

        if page_count == 0 or not pages:
            raise PDFLoadError(
                "no_extractable_text",
                f"PDF tidak memiliki teks yang dapat diekstrak: {source_path.name}. OCR belum didukung.",
            )
        warnings = ()
        if empty_pages:
            warnings = (
                f"Halaman tanpa teks dilewati pada {source_path.name}: "
                + ", ".join(str(value) for value in empty_pages)
                + ".",
            )
        source_id = f"pdf-{self._slug(source_path.stem)}-{content_hash[:12]}"
        return PDFLoadResult(
            source_path=resolved_path,
            source_document_id=source_id,
            source_file=source_path.name,
            document_title=title,
            content_hash=content_hash,
            file_size=file_size,
            page_count=page_count,
            pages=tuple(pages),
            warnings=warnings,
        )

    @staticmethod
    def discover(directory):
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            return []
        return sorted(
            (
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )

    @staticmethod
    def hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def normalize_text(value):
        text = str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
        normalized = []
        previous_blank = False
        for line in lines:
            blank = not line
            if blank and previous_blank:
                continue
            normalized.append(line)
            previous_blank = blank
        return "\n".join(normalized).strip()

    @staticmethod
    def _document_title(reader, path):
        raw_title = getattr(reader.metadata, "title", None) if reader.metadata else None
        title = PDFDocumentLoader.normalize_text(raw_title or "")
        if title:
            return title
        return re.sub(r"[_-]+", " ", Path(path).stem).strip() or Path(path).stem

    @staticmethod
    def _slug(value):
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-") or "document"
