import hashlib
import logging
import re
from statistics import mean, median

from domain.models import DocumentChunk


logger = logging.getLogger(__name__)

FIELD_HEADINGS = {
    "course": {
        "Jenis", "Nama Mata Kuliah", "Kode", "Program Studi", "Semester", "SKS",
        "Prasyarat", "Capaian Pembelajaran", "Deskripsi",
    },
    "lecturer": {
        "Jenis", "Nama", "Program Studi", "Research Interest", "Pendidikan",
        "Deskripsi", "Email", "Portofolio",
    },
    "academic": {"Jenis", "Judul", "Link"},
    "pdf": {"Jenis", "Judul Dokumen", "Halaman", "Isi"},
    "ftmm": {
        "Jenis", "Nama Fakultas", "Visi", "Misi", "Lokasi", "Lokasi kampus",
        "Kontak Fakultas", "Program Studi", "Akreditasi", "Tentang Fakultas",
        "Sejarah Fakultas", "Hasil Karya FTMM", "Prestasi Mahasiswa FTMM",
        "Media Sosial", "Warna Bendera", "Filosofi Bendera", "Filosofi Logo",
        "Maskot Cirion",
    },
}


class DocumentChunker:
    """Structure-aware chunker dengan token-length fallback untuk long prose."""

    def __init__(
        self,
        enabled=True,
        max_tokens=200,
        overlap_tokens=24,
        chunking_version="stage4-lecturer-intent-v1",
        tokenizer_name="BAAI/bge-m3",
        tokenizer_path=None,
        tokenizer=None,
    ):
        if max_tokens <= 0:
            raise ValueError("Chunk max tokens harus lebih besar dari nol.")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("Chunk overlap harus lebih kecil dari max tokens.")
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunking_version = chunking_version
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self._tokenizer = tokenizer
        self.duplicate_chunks_removed = 0

    @property
    def cache_configuration(self):
        return {
            "chunking_enabled": self.enabled,
            "chunk_max_tokens": self.max_tokens,
            "chunk_overlap_tokens": self.overlap_tokens,
            "chunking_version": self.chunking_version,
        }

    def load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        source = self.tokenizer_path or self.tokenizer_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source, local_files_only=True
            )
        except Exception:
            logger.info("Tokenizer chunking belum ada di cache; mencoba mengunduhnya.")
            self._tokenizer = AutoTokenizer.from_pretrained(source)
        return self._tokenizer

    def count_tokens(self, text):
        return len(
            self.load_tokenizer().encode(
                str(text), add_special_tokens=False, truncation=False
            )
        )

    def chunk_documents(self, documents):
        chunks = []
        seen_content = set()
        self.duplicate_chunks_removed = 0
        for document in documents:
            for chunk in self.chunk_document(document):
                content_hash = hashlib.sha256(chunk.content.encode("utf-8")).digest()
                if content_hash in seen_content:
                    self.duplicate_chunks_removed += 1
                    continue
                seen_content.add(content_hash)
                chunks.append(chunk)
        return chunks

    def chunk_document(self, document):
        document_type = document.metadata.get("type")
        if document_type == "ftmm":
            contents = self._chunk_ftmm(document)
        elif document_type == "course":
            contents = self._chunk_course(document)
        elif document_type == "lecturer":
            contents = self._chunk_lecturer(document)
        elif document_type == "academic":
            contents = self._chunk_academic(document)
        elif document_type == "pdf":
            contents = self._chunk_pdf(document)
        else:
            contents = [
                (content, "continuation")
                for content in self._split_with_header("", document.content)
            ]
        chunks = []
        for index, item in enumerate(contents):
            content, section = item[:2]
            chunk_role = item[2] if len(item) > 2 else None
            id_suffix = item[3] if len(item) > 3 else None
            if content.strip():
                chunks.append(
                    self._make_chunk(
                        document,
                        index,
                        content,
                        section,
                        chunk_role=chunk_role,
                        id_suffix=id_suffix,
                    )
                )
        return chunks

    def _chunk_ftmm(self, document):
        fields = self._parse_fields(document.content, FIELD_HEADINGS["ftmm"])
        faculty_name = fields.get(
            "Nama Fakultas",
            "Fakultas Teknologi Maju dan Multidisiplin (FTMM), Universitas Airlangga",
        )
        identity = (
            "Jenis: Informasi Fakultas FTMM\n"
            f"Nama Fakultas: {faculty_name}"
        )
        contents = []
        for heading, value in fields.items():
            if heading in {"Jenis", "Nama Fakultas"} or not value.strip():
                continue
            header = f"{identity}\nTopik: {heading} FTMM"
            body = f"{heading}: {value}"
            contents.extend(
                (content, heading)
                for content in self._split_with_header(header, body)
            )
        return contents or [(document.content, "Informasi FTMM")]

    def _chunk_course(self, document):
        if self.count_tokens(document.content) <= self.max_tokens:
            return [(document.content, "overview")]
        fields = self._parse_fields(document.content, FIELD_HEADINGS["course"])
        identity_keys = (
            "Jenis", "Nama Mata Kuliah", "Kode", "Program Studi", "Semester", "SKS", "Prasyarat"
        )
        header = self._render_fields(fields, identity_keys)
        chunks = [(header, "overview")]
        for heading in ("Capaian Pembelajaran", "Deskripsi"):
            if fields.get(heading, "").strip():
                chunks.extend(
                    (content, heading)
                    for content in self._split_with_header(
                        header, f"{heading}: {fields[heading]}"
                    )
                )
        return chunks

    def _chunk_lecturer(self, document):
        fields = self._parse_fields(document.content, FIELD_HEADINGS["lecturer"])
        identity_header = self._render_fields(
            fields, ("Jenis", "Nama", "Program Studi")
        )
        profile_body = self._render_fields(fields, ("Pendidikan", "Email"))
        profile_content = f"{identity_header}\nTopik: Profil Dosen\n{profile_body}".strip()
        chunks = [(profile_content, "Profil", "profile", "profile")]

        research_interest, _research_source = self._research_interest(fields)
        if research_interest:
            research_content = (
                f"{identity_header}\n"
                "Topik: Bidang Penelitian / Research Interest\n"
                f"Research Interest: {research_interest}"
            )
            chunks.append(
                (
                    research_content,
                    "Research Interest",
                    "research_interest",
                    "research-interest",
                )
            )

        for heading, chunk_role, id_prefix in (
            ("Deskripsi", "description", "description"),
            ("Portofolio", "portfolio", "portfolio"),
        ):
            if self._has_information(fields.get(heading, "")):
                detail_chunks = self._split_with_header(
                    f"{identity_header}\nTopik: {heading}",
                    f"{heading}: {fields[heading]}",
                )
                chunks.extend(
                    (
                        content,
                        heading,
                        chunk_role,
                        f"{id_prefix}-{part_index:04d}",
                    )
                    for part_index, content in enumerate(detail_chunks)
                )
        return chunks

    @staticmethod
    def _has_information(value):
        return str(value).strip().casefold() not in {"", "tidak ada", "none", "-"}

    @staticmethod
    def _research_interest(fields):
        value = fields.get("Research Interest", "").strip()
        if value and value.casefold() not in {"tidak ada", "none", "-"}:
            return value, "Research Interest"

        description = fields.get("Deskripsi", "")
        pattern = re.compile(
            r"(?:her|his|current)?\s*research interests?\s*"
            r"(?:are|is|include|includes|:)\s*(.+?)(?=\n\s*\n|$)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(description)
        if match:
            extracted = re.sub(r"\s+", " ", match.group(1)).strip()
            return extracted, "Description"
        return None, None

    def _chunk_academic(self, document):
        if self.count_tokens(document.content) <= self.max_tokens:
            return [(document.content, "overview")]
        fields = self._parse_fields(document.content, FIELD_HEADINGS["academic"])
        header = self._render_fields(fields, ("Jenis", "Judul"))
        body = "\n".join(
            f"{key}: {value}" for key, value in fields.items() if key not in {"Jenis", "Judul"}
        )
        return [
            (content, "procedure") for content in self._split_with_header(header, body)
        ]

    def _chunk_pdf(self, document):
        fields = self._parse_fields(document.content, FIELD_HEADINGS["pdf"])
        title = fields.get("Judul Dokumen") or document.metadata.get("document_title")
        page = fields.get("Halaman") or document.metadata.get("page_start")
        header = (
            "Jenis: Dokumen PDF FTMM\n"
            f"Judul Dokumen: {title}\n"
            f"Halaman: {page}"
        )
        body = fields.get("Isi", document.content)
        section = f"Halaman {page}"
        return [
            (content, section)
            for content in self._split_with_header(header, f"Isi: {body}")
        ]

    def _split_with_header(self, header, body):
        full_content = f"{header}\n{body}".strip()
        if self.count_tokens(full_content) <= self.max_tokens:
            return [full_content]

        tokenizer = self.load_tokenizer()
        header_tokens = tokenizer.encode(
            header, add_special_tokens=False, truncation=False
        )
        body_tokens = tokenizer.encode(
            body, add_special_tokens=False, truncation=False
        )
        available = self.max_tokens - len(header_tokens)
        if available <= 0:
            raise ValueError("Semantic chunk header melebihi CHUNK_MAX_TOKENS.")
        step = max(1, available - min(self.overlap_tokens, available - 1))
        chunks = []
        for start in range(0, len(body_tokens), step):
            token_slice = body_tokens[start : start + available]
            if not token_slice:
                break
            decoded = tokenizer.decode(token_slice, skip_special_tokens=True).strip()
            chunk = f"{header}\n{decoded}".strip()
            if chunk:
                chunks.append(chunk)
            if start + available >= len(body_tokens):
                break
        return chunks

    def _make_chunk(
        self, document, index, content, section, chunk_role=None, id_suffix=None
    ):
        metadata = dict(document.metadata)
        document_type = metadata.get("type")
        metadata.update(
            {
                "parent_document_id": document.id,
                "chunk_index": index,
                "token_count": self.count_tokens(content),
                "section": section,
            }
        )
        if document_type == "course":
            metadata.update(
                course_code=metadata.get("code"),
                course_name=metadata.get("name"),
                sks=metadata.get("credits"),
            )
        elif document_type == "lecturer":
            metadata["lecturer_name"] = metadata.get("name")
            metadata["chunk_role"] = chunk_role or "profile"
            if section == "Research Interest":
                metadata["research_interest_source"] = (
                    "Research Interest"
                    if str(metadata.get("research_interest", "")).strip().casefold()
                    not in {"", "tidak ada", "none", "-"}
                    else "Description"
                )
        elif document_type == "academic":
            metadata["title"] = metadata.get("name")
        elif document_type == "staff":
            metadata.update(
                position=metadata.get("role"),
                unit="Fakultas Teknologi Maju dan Multidisiplin (FTMM), Universitas Airlangga",
            )
        return DocumentChunk(
            id=(
                f"{document.id}::{id_suffix}"
                if id_suffix else f"{document.id}::chunk-{index:04d}"
            ),
            parent_document_id=document.id,
            content=content,
            chunk_index=index,
            metadata=metadata,
        )

    @staticmethod
    def _parse_fields(content, allowed_headings=None):
        fields = {}
        current = None
        for line in str(content).splitlines():
            match = re.match(r"^([^:\n]{1,80}):\s*(.*)$", line)
            if match and (
                allowed_headings is None or match.group(1).strip() in allowed_headings
            ):
                current = match.group(1).strip()
                fields[current] = match.group(2).strip()
            elif current and line.strip():
                fields[current] = f"{fields[current]}\n{line.strip()}".strip()
        return fields

    @staticmethod
    def _render_fields(fields, keys):
        return "\n".join(
            f"{key}: {fields[key]}" for key in keys if fields.get(key, "").strip()
        )

    def statistics(self, chunks):
        token_counts = sorted(chunk.metadata["token_count"] for chunk in chunks)
        if not token_counts:
            return {}
        p95_index = min(len(token_counts) - 1, int(0.95 * (len(token_counts) - 1)))
        return {
            "min": token_counts[0],
            "median": median(token_counts),
            "mean": mean(token_counts),
            "p95": token_counts[p95_index],
            "max": token_counts[-1],
            "over_max": sum(value > self.max_tokens for value in token_counts),
            "empty": sum(not chunk.content.strip() for chunk in chunks),
            "duplicates_removed": self.duplicate_chunks_removed,
        }
