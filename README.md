# Chatbot FTMM

Baseline stabil Chatbot FTMM untuk menjawab pertanyaan mengenai fakultas, mata kuliah, dosen, informasi akademik, serta pejabat dan staf FTMM Universitas Airlangga.

## Requirements

- Python 3.10 atau lebih baru
- Akses internet pada pemuatan pertama model `BAAI/bge-m3`, kecuali model sudah tersedia di cache Hugging Face
- OpenAI API key untuk menghasilkan jawaban akhir. Halaman dan retrieval lokal tetap dapat diuji tanpa API key.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

Salin `.env.example` menjadi `.env`, lalu isi nilai rahasia hanya di `.env`.

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
FLASK_SECRET_KEY=
FLASK_DEBUG=false
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
EMBEDDING_BACKEND=bge-m3
EMBEDDING_ALLOW_FALLBACK=false
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_DIRECTORY=cache/embeddings
RETRIEVAL_TOP_K=10
RETRIEVAL_MIN_SCORE=0.5
RAG_DEBUG=false
BGE_M3_MODEL=BAAI/bge-m3
BGE_M3_LOCAL_PATH=
BGE_M3_DEVICE=cpu
BGE_M3_PRECISION=bf16
BGE_M3_BATCH_SIZE=1
BGE_M3_MAX_LENGTH=256
```

`OPENAI_API_KEY` tidak wajib untuk membuka aplikasi. `FLASK_SECRET_KEY` sebaiknya berupa string acak yang panjang. Jika tidak diisi, aplikasi menggunakan kunci sementara dan session tidak akan bertahan setelah proses dimulai ulang.

## Running the Application

```powershell
python app.py
```

Buka `http://127.0.0.1:5000`. Debug nonaktif secara default; aktifkan hanya untuk pengembangan dengan `FLASK_DEBUG=true`.

## Project Structure

```text
Chatbot-FTMM/
|-- app.py
|-- config.py                 # Konfigurasi embedding dan retrieval
|-- domain/models.py          # Document, RetrievedDocument, ChatResult
|-- repositories/
|   `-- knowledge_repository.py
|-- indexes/
|   `-- in_memory_vector_index.py
|-- retrievers/
|   |-- base.py
|   `-- dense_retriever.py
|-- services/
|   |-- embedding_service.py
|   |-- bge_m3_embedding_service.py
|   |-- embedding_factory.py
|   |-- embedding_cache.py
|   |-- context_builder.py
|   |-- llm_service.py
|   `-- chat_service.py
|-- scripts/
|   |-- test_retrieval.py
|   |-- test_bge_m3.py
|   |-- evaluate_retrieval.py
|   `-- compare_retrieval_results.py
|-- tests/
|-- cache/embeddings/       # Generated locally; ignored by Git
|-- code/                    # Versi lama, dipertahankan untuk riwayat
|-- data/                    # Lima dataset CSV sumber
|-- static/css/style.css
|-- templates/
|   |-- index.html
|   `-- evaluation.html
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Retrieval Architecture

```text
User Query
  -> Flask Route
  -> Chat Service
  -> Dense Retriever
  -> BGE-M3 Embedding Service (default)
  -> BGE-M3 In-Memory Vector Index
  -> RetrievedDocument[]
  -> Context Builder
  -> LLM Service
  -> ChatResult
```

`KnowledgeRepository` membaca CSV satu kali dan mengubah setiap record menjadi `Document` dengan ID stabil, content lengkap, serta metadata konsisten. Retriever hanya menerima `Document`; retriever dan route Flask tidak membaca CSV atau mengetahui struktur pandas.

Pada akses retrieval pertama, `DenseRetriever` meminta `EmbeddingService` membuat document embeddings, lalu menyimpannya ke `InMemoryVectorIndex`. Query berikutnya hanya meng-encode query dan mencari pada index yang sudah tersedia. Document tidak di-encode ulang pada setiap request.

`ContextBuilder` mengubah hasil retrieval menjadi blok internal `[SOURCE n]`. Content yang digunakan embedding juga diberikan kepada LLM sehingga deskripsi mata kuliah, prasyarat, research interest, pendidikan, link akademik, dan informasi FTMM tidak hilang.

Konfigurasi `RAG_DEBUG=true` menulis query, document ID, type, dan score ke log. Mode ini tidak menampilkan retrieval detail kepada pengguna dan tidak mencatat API key atau Flask secret.

## Default Embedding Backend

Backend default adalah dense `BAAI/bge-m3` melalui FlagEmbedding. Konfigurasi development yang telah diuji adalah CPU, BF16, batch size 1, dan max length 256. Dense vector memiliki 1024 dimensi berdasarkan output aktual model.

Backend lama `sentence-transformers/all-MiniLM-L6-v2` tetap tersedia untuk rollback dan debugging. Factory hanya membuat backend yang dipilih; kedua model tidak dimuat bersamaan pada aplikasi normal.

Retrieval memilih maksimal 10 dokumen menggunakan cosine similarity dengan threshold `0.5`. Jika tidak ada skor yang melewati threshold, kandidat dengan skor tertinggi tetap digunakan. Index menyimpan identitas model dan dimensionality aktual serta menolak query vector dari model/dimensi berbeda.

Arsitektur ini masih retrieval berbasis embedding/cosine similarity. Ini belum merupakan arsitektur RAG final dan belum memakai vector database, ingestion pipeline, chunking baru, hybrid search, reranker, ataupun citation system.

Session percakapan disimpan sebagai file lokal melalui Flask-Session. Folder session dan file rahasia `.env` diabaikan oleh Git.

## Testing Retrieval

Unit test tidak memerlukan OpenAI API atau download model:

```powershell
python -m unittest discover -s tests -v
```

Smoke test menggunakan model embedding baseline tetapi tidak memanggil LLM:

```powershell
python -m scripts.test_retrieval
```

## Current Limitations

- Dataset evaluasi `data/data_validasi4.csv` tidak tersedia, sehingga route evaluasi menampilkan pesan bahwa evaluasi dinonaktifkan sementara.
- Model embedding perlu diunduh pada pemuatan pertama dan dapat gagal tanpa koneksi atau cache lokal.
- BGE-M3 pada CPU memakai RAM dan waktu loading lebih besar dibanding backend baseline.
- Evaluasi development masih menemukan query `Dosen yang fokus machine learning` yang tidak membawa lecturer dengan label ground truth eksplisit ke Top-5; perbaikan lanjutan tidak dilakukan dengan keyword hack.
- Jawaban akhir memerlukan OpenAI API key dan koneksi jaringan.
- Dataset memuat beberapa teks yang sudah mengalami mojibake (contoh karakter `â€™`) dan anomali konten; isinya belum dibersihkan pada Tahap 0.
- Nama dataset sumber masih menggunakan awalan `Salinan` agar file asli tidak dihapus atau diubah pada tahap audit.
- Versi lama di folder `code/` memiliki referensi file yang hilang dan bukan entry point aktif.
- Output model dirender sebagai teks yang di-escape oleh Jinja, sehingga Markdown belum dirender sebagai HTML.
- Retrieval masih dense-only; belum menggunakan vector database, hybrid retrieval, BM25, reranker, atau citation final.
- Dataset FTMM yang panjang masih direpresentasikan sebagai satu Document dan belum di-chunk.
- In-memory index tetap dibangun setiap proses dimulai, tetapi document vector dimuat dari persistent cache ketika valid.

## Switching Embedding Backend

Pilih BGE-M3 production/default:

```dotenv
EMBEDDING_BACKEND=bge-m3
```

Rollback ke MiniLM:

```dotenv
EMBEDDING_BACKEND=baseline
```

Nilai selain `baseline` dan `bge-m3` ditolak. `EMBEDDING_ALLOW_FALLBACK=false` adalah default: kegagalan konfigurasi BGE ditampilkan secara jelas dan tidak disembunyikan oleh fallback implicit.

## Embedding Cache

Document embeddings disimpan sebagai compressed NumPy `.npz`, dengan metadata JSON terpisah di `cache/embeddings/`. Cache tidak tersedia melalui Flask static route dan diabaikan Git.

Validasi cache mencakup model, dimensi aktual, jumlah dokumen, konfigurasi embedding, serta SHA-256 fingerprint deterministik dari ID, content, dan metadata semantik stabil. Perubahan corpus/model/configuration, dimension mismatch, atau cache corrupt menyebabkan safe rebuild dengan warning log. Query embedding tidak disimpan secara persisten.

Cache dapat dinonaktifkan untuk diagnosis:

```dotenv
EMBEDDING_CACHE_ENABLED=false
```

## Development Configuration

Konfigurasi development yang berhasil pada mesin audit:

```dotenv
BGE_M3_MODEL=BAAI/bge-m3
BGE_M3_DEVICE=cpu
BGE_M3_PRECISION=bf16
BGE_M3_BATCH_SIZE=1
BGE_M3_MAX_LENGTH=256
```

Hasil aktual:

- dense query vector: 1024 dimensi;
- dense document vector: 1024 dimensi;
- sparse dan ColBERT output dinonaktifkan;
- vector dinormalisasi oleh FlagEmbedding (`normalize_embeddings=True`);
- model hanya dimuat satu kali per service instance;
- full corpus berisi 391 dokumen dengan matrix `(391, 1024)`;
- cache hit membangun ulang in-memory index tanpa document re-encoding.

Pada evaluation set development berisi 20 query faktual, hasil aktual adalah Recall@1 `0.90`, Recall@3 `0.95`, Recall@5 `0.95`, dan MRR `0.925`. Dataset ini hanya regression fixture untuk development, bukan evaluasi penelitian final.

CPU FP32 tidak stabil pada mesin RAM 7,7 GiB karena tekanan memory tinggi. CPU BF16 berhasil pada CPU AVX512 dengan PyTorch BF16 support. Jangan menyalin konfigurasi BF16 ke mesin lain tanpa memastikan dukungan CPU/GPU.

## Running BGE-M3 Tests

Smoke test tidak membutuhkan Flask atau OpenAI. Pada PowerShell:

```powershell
$env:BGE_M3_DEVICE="cpu"
$env:BGE_M3_PRECISION="bf16"
$env:BGE_M3_BATCH_SIZE="1"
$env:BGE_M3_MAX_LENGTH="256"
python -m scripts.test_bge_m3
```

Pemuatan pertama mengunduh artefak PyTorch minimum sekitar 2,16 GiB. Format ONNX tidak dibutuhkan. Untuk mode offline, `BGE_M3_LOCAL_PATH` dapat menunjuk ke snapshot PyTorch yang sudah lengkap.

Integration test model penuh dijalankan manual agar normal test suite dan CI tidak mengunduh model besar:

```powershell
$env:RUN_BGE_M3_INTEGRATION="true"
python -m unittest tests.test_bge_m3_adapter.BGEM3AdapterIntegrationTests -v
```

Full-corpus evaluation dan comparison:

```powershell
python -m scripts.evaluate_retrieval --backend baseline --output cache/evaluations/baseline.json
python -m scripts.evaluate_retrieval --backend bge-m3 --output cache/evaluations/bge.json
python -m scripts.compare_retrieval_results cache/evaluations/baseline.json cache/evaluations/bge.json
```

Full-corpus BGE integration test tetap opt-in:

```powershell
$env:RUN_BGE_M3_FULL_CORPUS="true"
python -m unittest tests.test_bge_m3_adapter.BGEM3FullCorpusIntegrationTests -v
```
