import logging
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from domain.models import Document


logger = logging.getLogger(__name__)


DATASET_DEFINITIONS = {
    "course": {
        "filename": "Salinan data_matkul.csv",
        "columns": [
            "Nama Mata Kuliah",
            "Kode Mata Kuliah",
            "Beban Studi",
            "Prodi",
            "Semester",
            "Prasyarat",
            "Capaian Pembelajaran Mata Kuliah",
            "Deskripsi Mata Kuliah/Silabus",
        ],
    },
    "lecturer": {
        "filename": "Salinan data_dosen.csv",
        "columns": [
            "Nama",
            "NIK/NIP",
            "Pendidikan",
            "Research Interest",
            "Prodi",
            "Email",
            "Description",
            "Portofolio",
        ],
    },
    "academic": {
        "filename": "Salinan data_akademik.csv",
        "columns": ["Akademik", "Link"],
    },
    "ftmm": {
        "filename": "Salinan data_ftmm.csv",
        "columns": [
            "Tentang Fakultas",
            "Akreditasi",
            "Program Studi",
            "Sejarah Fakultas",
            "Hasil Karya FTMM",
            "Prestasi Mahasiswa FTMM",
            "Visi",
            "Misi",
            "Lokasi",
            "Kontak Fakultas",
            "Media Sosial",
            "Warna Bendera",
            "Filosofi Bendera",
            "Filosofi Logo",
            "Lokasi kampus",
            "Maskot Cirion",
        ],
    },
    "staff": {
        "filename": "Salinan data_pejabat_staf.csv",
        "columns": ["Jabatan", "Nama"],
    },
}


def prepare_document_text(document_type, record):
    """Representasi semantik dari field sumber; tidak menambah fakta baru."""
    if document_type == "course":
        labels = [
            ("Jenis", "Mata Kuliah FTMM"),
            ("Nama Mata Kuliah", record["Nama Mata Kuliah"]),
            ("Kode", record["Kode Mata Kuliah"]),
            ("Program Studi", record["Prodi"]),
            ("Semester", record["Semester"]),
            ("SKS", record["Beban Studi"]),
            ("Prasyarat", record["Prasyarat"]),
            ("Capaian Pembelajaran", record["Capaian Pembelajaran Mata Kuliah"]),
            ("Deskripsi", record["Deskripsi Mata Kuliah/Silabus"]),
        ]
    elif document_type == "lecturer":
        labels = [
            ("Jenis", "Dosen FTMM"),
            ("Nama", record["Nama"]),
            ("Program Studi", record["Prodi"]),
            ("Research Interest", record["Research Interest"]),
            ("Pendidikan", record["Pendidikan"]),
            ("Deskripsi", record["Description"]),
            ("Email", record["Email"]),
            ("Portofolio", record["Portofolio"]),
        ]
    elif document_type == "academic":
        labels = [
            ("Jenis", "Informasi Akademik FTMM"),
            ("Judul", record["Akademik"]),
            ("Link", record["Link"]),
        ]
    elif document_type == "staff":
        role = str(record["Jabatan"]).strip()
        labels = [
            ("Jenis", "Pejabat/Staf FTMM"),
            ("Jabatan", f"{role} FTMM"),
            ("Nama", record["Nama"]),
            (
                "Unit",
                "Fakultas Teknologi Maju dan Multidisiplin (FTMM), "
                "Universitas Airlangga",
            ),
        ]
    else:
        priority_columns = [
            "Visi",
            "Misi",
            "Lokasi",
            "Lokasi kampus",
            "Kontak Fakultas",
            "Program Studi",
            "Akreditasi",
            "Tentang Fakultas",
            "Sejarah Fakultas",
            "Hasil Karya FTMM",
            "Prestasi Mahasiswa FTMM",
            "Media Sosial",
            "Warna Bendera",
            "Filosofi Bendera",
            "Filosofi Logo",
            "Maskot Cirion",
        ]
        labels = [
            ("Jenis", "Informasi Fakultas FTMM"),
            (
                "Nama Fakultas",
                "Fakultas Teknologi Maju dan Multidisiplin (FTMM), "
                "Universitas Airlangga",
            ),
        ]
        labels.extend((column, record[column]) for column in priority_columns)
    return "\n".join(f"{label}: {value}" for label, value in labels)


def _slug(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-") or "unknown"


class KnowledgeRepository:
    """Muat CSV dan normalisasikan seluruh record menjadi Document."""

    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)
        self.errors = []
        self.load_seconds = 0.0
        self._documents = None

    def get_documents(self):
        if self._documents is None:
            self._documents = self._load_documents()
        return list(self._documents)

    def _load_documents(self):
        started_at = time.perf_counter()
        documents = []
        used_ids = set()
        self.errors = []

        for document_type, definition in DATASET_DEFINITIONS.items():
            path = self.data_directory / definition["filename"]
            dataframe = self._read_dataset(document_type, path, definition["columns"])
            if dataframe is None:
                continue

            for row_index, row in dataframe.iterrows():
                record = row.to_dict()
                document_id = self._build_document_id(
                    document_type, record, int(row_index), used_ids
                )
                documents.append(
                    Document(
                        id=document_id,
                        content=prepare_document_text(document_type, record),
                        metadata=self._build_metadata(
                            document_type, record, path.name, int(row_index)
                        ),
                    )
                )

            logger.info(
                "Memuat %s document bertipe %s dari %s.",
                len(dataframe),
                document_type,
                path.name,
            )

        self.load_seconds = time.perf_counter() - started_at
        return documents

    def _read_dataset(self, document_type, path, required_columns):
        if not path.exists():
            message = f"Dataset {document_type} tidak ditemukan: {path.name}."
            logger.error(message)
            self.errors.append(message)
            return None

        try:
            dataframe = pd.read_csv(path, encoding="utf-8")
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            message = f"Dataset {document_type} gagal dibaca ({type(exc).__name__})."
            logger.error(message)
            self.errors.append(message)
            return None

        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]
        if missing_columns:
            message = (
                f"Dataset {document_type} tidak digunakan karena kolom tidak tersedia: "
                f"{', '.join(missing_columns)}."
            )
            logger.error(message)
            self.errors.append(message)
            return None

        if dataframe.empty:
            message = f"Dataset {document_type} kosong."
            logger.error(message)
            self.errors.append(message)
            return None

        invalid_rows = dataframe[required_columns].isnull().any(axis=1)
        if invalid_rows.any():
            message = (
                f"Dataset {document_type} memiliki {int(invalid_rows.sum())} row kosong; "
                "row tersebut dilewati."
            )
            logger.warning(message)
            self.errors.append(message)
            dataframe = dataframe.loc[~invalid_rows]

        return dataframe

    def _build_document_id(self, document_type, record, row_index, used_ids):
        if document_type == "course":
            course_code = str(record["Kode Mata Kuliah"]).strip()
            identity = (
                record["Nama Mata Kuliah"]
                if course_code.lower() == "tidak ada"
                else course_code
            )
        elif document_type == "lecturer":
            identifier = str(record["NIK/NIP"]).strip()
            identity = record["Nama"] if identifier.lower() == "tidak ada" else identifier
        elif document_type == "academic":
            identity = f"{row_index + 1}-{record['Akademik']}"
        elif document_type == "ftmm":
            identity = f"overview-{row_index + 1}"
        else:
            identity = f"{row_index + 1}-{record['Jabatan']}-{record['Nama']}"

        base_id = f"{document_type}-{_slug(identity)}"
        document_id = base_id
        if document_id in used_ids:
            document_id = f"{base_id}-{row_index + 1}"
        used_ids.add(document_id)
        return document_id

    def _build_metadata(self, document_type, record, source, row_index):
        metadata = {
            "type": document_type,
            "source": source,
            "source_row": row_index + 2,
        }

        if document_type == "course":
            metadata.update(
                {
                    "name": record["Nama Mata Kuliah"],
                    "code": record["Kode Mata Kuliah"],
                    "program": record["Prodi"],
                    "semester": record["Semester"],
                    "credits": record["Beban Studi"],
                    "prerequisite": record["Prasyarat"],
                }
            )
        elif document_type == "lecturer":
            metadata.update(
                {
                    "name": record["Nama"],
                    "identifier": record["NIK/NIP"],
                    "program": record["Prodi"],
                    "education": record["Pendidikan"],
                    "research_interest": record["Research Interest"],
                    "email": record["Email"],
                }
            )
        elif document_type == "academic":
            metadata.update({"name": record["Akademik"], "link": record["Link"]})
        elif document_type == "ftmm":
            metadata.update({"name": "FTMM"})
        elif document_type == "staff":
            metadata.update({"name": record["Nama"], "role": record["Jabatan"]})

        return metadata
