import os
import logging
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from stop_words import get_stop_words
import pandas as pd
import joblib
import openai
import re
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
Session(app)

# OpenAI Client Initialization
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY tidak ditemukan dalam file .env")

# File Paths
MODEL_PATH = 'multilingual_model.pkl'
EMBEDDINGS_PATH = 'embeddings.pkl'
DOCUMENTS_PATH = 'documents.pkl'
DATA_MATKUL_LIST_PATH = 'data_matkul_list.pkl'
DOSEN_LIST_PATH = 'dosen_list.pkl'
DATA_AKADEMIK_LIST_PATH = 'data_akademik_list.pkl'
FTMM_LIST_PATH = 'data_ftmm_list.pkl'
PEJABAT_STAF_LIST_PATH = 'pejabat_staf_list.pkl'
DATA_VALIDASI_PATH = 'data/data_validasi4.csv'  # Path untuk data_validasi.csv

# Functions
def create_konten_teks_matkul(row):
    return (
        f"Nama Mata Kuliah: {row['Nama Mata Kuliah']}\n"
        f"Kode Mata Kuliah: {row['Kode Mata Kuliah']}\n"
        f"Beban Studi: {row['Beban Studi']}\n"
        f"Prodi: {row['Prodi']}\n"
        f"Semester: {row['Semester']}\n"
        f"Prasyarat: {row['Prasyarat']}\n"
        f"Capaian Pembelajaran: {row['Capaian Pembelajaran Mata Kuliah']}\n"
        f"Deskripsi: {row['Deskripsi Mata Kuliah/Silabus']}"
    )

def create_konten_teks_dosen(row):
    return (
        f"Nama: {row['Nama']}\n"
        f"NIK/NIP: {row['NIK/NIP']}\n"
        f"Pendidikan: {row['Pendidikan']}\n"
        f"Research Interest: {row['Research Interest']}\n"
        f"Prodi: {row['Prodi']}\n"
        f"Email: {row['Email']}\n"
        f"Description: {row['Description']}\n"
        f"Portofolio: {row['Portofolio']}"
    )

def create_konten_teks_akademik(row):
    return (
        f"Akademik: {row['Akademik']}\n"
        f"Link: {row['Link']}"
    )

def create_konten_teks_ftmm(row):
    konten = "\n".join([f"{col}: {row[col]}" for col in row.index])
    return konten

def create_konten_teks_pejabat_staf(row):
    return (
        f"Jabatan: {row['Jabatan']}\n"
        f"Nama: {row['Nama']}"
    )

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(EMBEDDINGS_PATH) and os.path.exists(DOCUMENTS_PATH):
        model = joblib.load(MODEL_PATH)
        embeddings = joblib.load(EMBEDDINGS_PATH)
        documents = joblib.load(DOCUMENTS_PATH)
        data_matkul_list = joblib.load(DATA_MATKUL_LIST_PATH)
        dosen_list = joblib.load(DOSEN_LIST_PATH)
        data_akademik_list = joblib.load(DATA_AKADEMIK_LIST_PATH)
        data_ftmm_list = joblib.load(FTMM_LIST_PATH)
        pejabat_staf_list = joblib.load(PEJABAT_STAF_LIST_PATH)
        logger.info("Model dan dokumen berhasil dimuat dari file.")
    else:
        # Load Mata Kuliah
        data_matkul_path = 'data/data_matkul.csv'
        if not os.path.exists(data_matkul_path):
            raise FileNotFoundError(f"File {data_matkul_path} tidak ditemukan.")
        data_matkul_df = pd.read_csv(data_matkul_path, encoding='utf-8')
        required_columns_matkul = [
            'Nama Mata Kuliah', 'Kode Mata Kuliah', 'Beban Studi', 'Prodi', 'Semester',
            'Prasyarat', 'Capaian Pembelajaran Mata Kuliah',
            'Deskripsi Mata Kuliah/Silabus'
        ]
        for col in required_columns_matkul:
            if col not in data_matkul_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV Mata Kuliah.")
            if data_matkul_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV Mata Kuliah.")
        data_matkul_df['Konten Teks'] = data_matkul_df.apply(create_konten_teks_matkul, axis=1)
        data_matkul_list = data_matkul_df.to_dict(orient='records')
        
        logger.info(f"Loaded {len(data_matkul_list)} Mata Kuliah.")

        # Load Dosen
        data_dosen_path = 'data/data_dosen.csv'
        if not os.path.exists(data_dosen_path):
            raise FileNotFoundError(f"File {data_dosen_path} tidak ditemukan.")
        data_dosen_df = pd.read_csv(data_dosen_path, encoding='utf-8')
        required_columns_dosen = [
            'Nama', 'NIK/NIP', 'Pendidikan', 'Research Interest',
            'Prodi', 'Email', 'Description', 'Portofolio'
        ]
        for col in required_columns_dosen:
            if col not in data_dosen_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV Dosen.")
            if data_dosen_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV Dosen.")
        data_dosen_df['Konten Teks'] = data_dosen_df.apply(create_konten_teks_dosen, axis=1)
        dosen_list = data_dosen_df.to_dict(orient='records')

        logger.info(f"Loaded {len(dosen_list)} Dosen.")

        # Load Akademik
        data_akademik_path = 'data/Data_akademik.csv'
        if not os.path.exists(data_akademik_path):
            raise FileNotFoundError(f"File {data_akademik_path} tidak ditemukan.")
        data_akademik_df = pd.read_csv(data_akademik_path, encoding='utf-8')
        required_columns_akademik = ['Akademik', 'Link']
        for col in required_columns_akademik:
            if col not in data_akademik_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV Akademik.")
            if data_akademik_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV Akademik.")
        data_akademik_df['Konten Teks'] = data_akademik_df.apply(create_konten_teks_akademik, axis=1)
        data_akademik_list = data_akademik_df.to_dict(orient='records')

        logger.info(f"Loaded {len(data_akademik_list)} Data Akademik.")

        # Load FTMM
        data_ftmm_path = 'data/Data_ftmm.csv'
        if not os.path.exists(data_ftmm_path):
            raise FileNotFoundError(f"File {data_ftmm_path} tidak ditemukan.")
        data_ftmm_df = pd.read_csv(data_ftmm_path, encoding='utf-8')
        required_columns_ftmm = [
            'Tentang Fakultas', 'Akreditasi', 'Program Studi', 'Sejarah Fakultas',
            'Hasil Karya FTMM', 'Prestasi Mahasiswa FTMM', 'Visi', 'Misi',
            'Lokasi', 'Kontak Fakultas', 'Media Sosial', 'Warna Bendera',
            'Filosofi Bendera', 'Filosofi Logo', 'Lokasi kampus', 'Maskot Cirion'
        ]
        for col in required_columns_ftmm:
            if col not in data_ftmm_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV FTMM.")
            if data_ftmm_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV FTMM.")
        data_ftmm_df['Konten Teks'] = data_ftmm_df.apply(create_konten_teks_ftmm, axis=1)
        data_ftmm_list = data_ftmm_df.to_dict(orient='records')

        logger.info(f"Loaded {len(data_ftmm_list)} Data FTMM.")

        # Load Pejabat dan Staf
        data_pejabat_staf_path = 'data/Data_pejabat_staf.csv'
        if not os.path.exists(data_pejabat_staf_path):
            raise FileNotFoundError(f"File {data_pejabat_staf_path} tidak ditemukan.")
        data_pejabat_staf_df = pd.read_csv(data_pejabat_staf_path, encoding='utf-8')
        required_columns_pejabat_staf = ['Jabatan', 'Nama']
        for col in required_columns_pejabat_staf:
            if col not in data_pejabat_staf_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV Pejabat dan Staf.")
            if data_pejabat_staf_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV Pejabat dan Staf.")
        data_pejabat_staf_df['Konten Teks'] = data_pejabat_staf_df.apply(create_konten_teks_pejabat_staf, axis=1)
        pejabat_staf_list = data_pejabat_staf_df.to_dict(orient='records')

        logger.info(f"Loaded {len(pejabat_staf_list)} Pejabat dan Staf.")

        # Combine Documents
        documents = (
            [{'Type': 'Mata Kuliah', 'Konten Teks': mk['Konten Teks']} for mk in data_matkul_list] +
            [{'Type': 'Dosen', 'Konten Teks': ds['Konten Teks']} for ds in dosen_list] +
            [{'Type': 'Akademik', 'Konten Teks': ak['Konten Teks']} for ak in data_akademik_list] +
            [{'Type': 'FTMM', 'Konten Teks': ft['Konten Teks']} for ft in data_ftmm_list] +
            [{'Type': 'Pejabat_Staf', 'Konten Teks': ps['Konten Teks']} for ps in pejabat_staf_list]
        )

        logger.info(f"Total Documents Combined: {len(documents)}")

        # Initialize Multilingual Model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = model.encode([doc['Konten Teks'] for doc in documents], convert_to_tensor=True)

        logger.info("Multilingual Model Embeddings Created.")

        # Save Model and Embeddings
        joblib.dump(model, MODEL_PATH)
        joblib.dump(embeddings, EMBEDDINGS_PATH)
        joblib.dump(documents, DOCUMENTS_PATH)
        joblib.dump(data_matkul_list, DATA_MATKUL_LIST_PATH)
        joblib.dump(dosen_list, DOSEN_LIST_PATH)
        joblib.dump(data_akademik_list, DATA_AKADEMIK_LIST_PATH)
        joblib.dump(data_ftmm_list, FTMM_LIST_PATH)
        joblib.dump(pejabat_staf_list, PEJABAT_STAF_LIST_PATH)
        logger.info("Model dan embeddings baru berhasil dibuat dan disimpan.")

    return model, embeddings, documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list

def find_most_relevant(query, model, embeddings, similarity_threshold=0.5, top_n=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    relevant_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    relevant_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)[:top_n]
    relevant_scores = [similarities[i] for i in relevant_indices]
    return relevant_indices, relevant_scores

def generate_response(query, relevant_indices, chat_history, documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list):
    info_relevant = ""
    mata_kuliah_count = 0
    dosen_count = 0
    akademik_count = 0
    ftmm_count = 0
    pejabat_staf_count = 0

    for idx in relevant_indices:
        doc = documents[idx]
        if doc['Type'] == 'Mata Kuliah':
            mk = data_matkul_list[idx]
            info_relevant += (
                f"- {mk['Nama Mata Kuliah']} (Kode: {mk['Kode Mata Kuliah']}), "
                f"Beban Studi: {mk['Beban Studi']} SKS, Semester: {mk['Semester']}\n"
            )
            mata_kuliah_count += 1
        elif doc['Type'] == 'Dosen':
            dosen_idx = idx - len(data_matkul_list)
            if 0 <= dosen_idx < len(dosen_list):
                dosen = dosen_list[dosen_idx]
                info_relevant += (
                    f"- {dosen['Nama']} (NIK/NIP: {dosen['NIK/NIP']}), "
                    f"Prodi: {dosen['Prodi']}, Email: {dosen['Email']}\n"
                )
                dosen_count += 1
            else:
                logger.warning(f"Indeks dosen {dosen_idx} di luar rentang list dosen.")
        elif doc['Type'] == 'Akademik':
            akademik_idx = idx - len(data_matkul_list) - len(dosen_list)
            if 0 <= akademik_idx < len(data_akademik_list):
                akademik = data_akademik_list[akademik_idx]
                info_relevant += (
                    f"- {akademik['Akademik']}: [Link]({akademik['Link']})\n"
                )
                akademik_count += 1
            else:
                logger.warning(f"Indeks akademik {akademik_idx} di luar rentang list akademik.")
        elif doc['Type'] == 'FTMM':
            ftmm_idx = idx - len(data_matkul_list) - len(dosen_list) - len(data_akademik_list)
            if 0 <= ftmm_idx < len(data_ftmm_list):
                ftmm = data_ftmm_list[ftmm_idx]
                info_relevant += (
                    f"- {ftmm['Tentang Fakultas']}: {ftmm['Sejarah Fakultas']}\n"
                    f"- Akreditasi: {ftmm['Akreditasi']}\n"
                    f"- Program Studi: {ftmm['Program Studi']}\n"
                    f"- Visi: {ftmm['Visi']}\n"
                    f"- Misi: {ftmm['Misi']}\n"
                    f"- Lokasi: {ftmm['Lokasi kampus']}\n"
                    f"- Kontak: {ftmm['Kontak Fakultas']}\n"
                    f"- Media Sosial: {ftmm['Media Sosial']}\n"
                    f"- Warna Bendera: {ftmm['Warna Bendera']}\n"
                    f"- Filosofi Bendera: {ftmm['Filosofi Bendera']}\n"
                    f"- Filosofi Logo: {ftmm['Filosofi Logo']}\n"
                    f"- Maskot Cirion: {ftmm['Maskot Cirion']}\n"
                )
                ftmm_count += 1
            else:
                logger.warning(f"Indeks FTMM {ftmm_idx} di luar rentang list FTMM.")
        elif doc['Type'] == 'Pejabat_Staf':
            pejabat_staf_idx = idx - len(data_matkul_list) - len(dosen_list) - len(data_akademik_list) - len(data_ftmm_list)
            if 0 <= pejabat_staf_idx < len(pejabat_staf_list):
                pejabat_staf = pejabat_staf_list[pejabat_staf_idx]
                info_relevant += (
                    f"- {pejabat_staf['Jabatan']}: {pejabat_staf['Nama']}\n"
                )
                pejabat_staf_count += 1
            else:
                logger.warning(f"Indeks pejabat/staf {pejabat_staf_idx} di luar rentang list pejabat/staf.")

    logger.info(f"Total Mata Kuliah Relevan: {mata_kuliah_count}")
    logger.info(f"Total Dosen Relevan: {dosen_count}")
    logger.info(f"Total Akademik Relevan: {akademik_count}")
    logger.info(f"Total FTMM Relevan: {ftmm_count}")
    logger.info(f"Total Pejabat dan Staf Relevan: {pejabat_staf_count}")
    logger.debug(f"Informasi Relevan yang dikirim ke LLM:\n{info_relevant}")

    # Accumulate chat history into the prompt
    prompt_history = ""
    for exchange in chat_history:
        prompt_history += f"Pengguna: {exchange['user']}\n"
        prompt_history += f"Bot: {exchange['bot']}\n"

    # Menambahkan pertanyaan saat ini dan konten relevan
    prompt = (
        f"{prompt_history}"
        f"Pengguna: {query}\n"
        f"Informasi Relevan:\n{info_relevant}\n"
        f"Bot:"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah asisten AI yang membantu menjawab pertanyaan tentang FTMM Universitas Airlangga. "
                "Baca dengan seksama deskripsi mata kuliah, informasi dosen, akademik, dan data pejabat serta staf yang diberikan, serta gunakan informasi tersebut untuk menjawab pertanyaan pengguna secara detail dan natural. "
                "Jika pengguna bertanya tentang rekomendasi, berikan saran yang sesuai dengan minat mereka berdasarkan deskripsi mata kuliah, dosen, atau data lainnya. "
                "Jawablah dalam Bahasa Indonesia dan Inggris dengan gaya bahasa yang ramah dan profesional. "
                "Jangan berikan jawaban dalam bentuk kode mata kuliah, NIK/NIP, atau jabatan, langsung berikan saja nama mata kuliah, dosen, atau pejabat."
                "Berikan rekomendasi dan jawaban yang tidak terlalu singkat kepada user."
                "Perhatikan dan baca secara menyeluruh informasi yang disediakan sebelum menjawab user."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        # Menggunakan OpenAI API untuk menghasilkan jawaban
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )

        # Mengambil respons dari asisten
        answer = response.choices[0].message.content.strip()

        logger.info(f"Generated answer: {answer}")

        return answer

    except Exception as e:
        # Menangani error, seperti kesalahan API
        logger.error(f"Error generating response: {e}")
        return f"Terjadi kesalahan dalam memproses permintaan: {str(e)}"

def evaluate_model(model, embeddings, documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list):
    # Load data_validasi.csv
    if not os.path.exists(DATA_VALIDASI_PATH):
        logger.error(f"File {DATA_VALIDASI_PATH} tidak ditemukan.")
        return {"error": f"File {DATA_VALIDASI_PATH} tidak ditemukan."}

    data_validasi_df = pd.read_csv(DATA_VALIDASI_PATH, encoding='utf-8')

    required_columns_validasi = ['query', 'relevant_documents', 'reference_answer']
    for col in required_columns_validasi:
        if col not in data_validasi_df.columns:
            logger.error(f"Kolom '{col}' tidak ditemukan dalam CSV Validasi.")
            return {"error": f"Kolom '{col}' tidak ditemukan dalam CSV Validasi."}

    # Initialize metrics
    precision_list = []
    recall_list = []
    bleu_list = []
    rouge_list = []

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for index, row in data_validasi_df.iterrows():
        query = row['query']
        relevant_docs = set([doc.strip() for doc in row['relevant_documents'].split(',')])
        reference_answer = row['reference_answer']

        # Cari dokumen yang paling relevan
        retrieved_indices, _ = find_most_relevant(query, model, embeddings)
        retrieved_docs = set()
        for idx in retrieved_indices:
            doc = documents[idx]
            # Ekstrak identifier unik berdasarkan tipe dokumen
            if doc['Type'] == 'Mata Kuliah':
                match = re.search(r'Kode Mata Kuliah:\s*(\w+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1)
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'Dosen':
                match = re.search(r'NIK/NIP:\s*(\w+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1)
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'Akademik':
                match = re.search(r'Akademik:\s*(.+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1).strip()
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'FTMM':
                match = re.search(r'Tentang Fakultas:\s*(.+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1).strip()
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'Pejabat_Staf':
                match = re.search(r'Jabatan:\s*(.+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1).strip()
                    retrieved_docs.add(doc_id)

        # Precision and Recall
        true_positives = len(retrieved_docs & relevant_docs)
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        precision_list.append(precision)
        recall_list.append(recall)

        # Generate response
        answer = generate_response(query, retrieved_indices, [], documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list)

        # BLEU Score
        reference_tokens = nltk.word_tokenize(reference_answer.lower())
        candidate_tokens = nltk.word_tokenize(answer.lower())
        bleu = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5))
        bleu_list.append(bleu)

        # ROUGE Score
        scores = scorer.score(reference_answer, answer)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        rouge_avg = (rouge1 + rouge2 + rougeL) / 3
        rouge_list.append(rouge_avg)

        logger.info(f"Evaluated Query {index + 1}/{len(data_validasi_df)}")

    # Hitung rata-rata metrik
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_bleu = sum(bleu_list) / len(bleu_list) if bleu_list else 0
    avg_rouge = sum(rouge_list) / len(rouge_list) if rouge_list else 0

    metrics = {
        "average_precision": round(avg_precision, 4),
        "average_recall": round(avg_recall, 4),
        "average_bleu": round(avg_bleu, 4),
        "average_rouge": round(avg_rouge, 4)
    }

    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

# Load Data
model, embeddings, documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list = load_model()

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.now().year
    logger.info(f"Handling {request.method} request.")

    if request.method == 'GET':
        if 'initialized' not in session:
            session.pop('chat_history', None)
            session['chat_history'] = []
            session['initialized'] = True
            logger.info("Chat history reset on first GET request.")

    if request.method == 'POST':
        if 'chat_history' not in session:
            session['chat_history'] = []
            logger.info("Chat history initialized on POST.")
        query = request.form['query']
        logger.info(f"Received query: {query}")
        if not query.strip():
            logger.warning("Received empty query.")
            return render_template('index7.html', chat_history=session.get('chat_history', []), current_year=current_year)
        try:
            # Cari dokumen yang paling relevan
            top_indices, top_scores = find_most_relevant(query, model, embeddings)
            # Generate respons menggunakan OpenAI LLM dengan history
            answer = generate_response(query, top_indices, session.get('chat_history', []), documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list)
            logger.info(f"Generated answer: {answer}")
            # Update chat history
            session['chat_history'].append({'user': query, 'bot': answer})
            session.modified = True  # Menandakan bahwa session telah diubah
        except Exception as e:
            # Tangani kesalahan dan tambahkan ke chat history
            logger.error(f"Error: {e}")
            session['chat_history'].append({'user': query, 'bot': f"Terjadi kesalahan: {str(e)}"})
            session.modified = True
        # Redirect setelah POST untuk menghindari form resubmission
        return redirect(url_for('index'))

    return render_template('index7.html', chat_history=session.get('chat_history', []), current_year=current_year)

@app.route('/reset', methods=['GET'])
def reset():
    session.pop('chat_history', None)
    session.pop('initialized', None)
    logger.info("Chat history dan initialized flag di-reset.")
    return redirect(url_for('index'))

@app.route('/evaluate', methods=['GET'])
def evaluate():
    current_year = datetime.now().year
    logger.info("Memulai evaluasi model dengan data_validasi.csv.")
    metrics = evaluate_model(model, embeddings, documents, data_matkul_list, dosen_list, data_akademik_list, data_ftmm_list, pejabat_staf_list)

    if "error" in metrics:
        return render_template('evaluation.html', error=metrics["error"], current_year=current_year)

    return render_template('evaluation.html', metrics=metrics, current_year=current_year)

if __name__ == '__main__':
    # Pastikan direktori untuk menyimpan session ada
    os.makedirs('./flask_session/', exist_ok=True)
    app.run(debug=True)
