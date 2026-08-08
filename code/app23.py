import os
import logging
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from stop_words import get_stop_words
import pandas as pd
import joblib
import openai
import re
import nltk

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
Session(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY tidak ditemukan dalam file .env")

TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
TFIDF_MATRIX_PATH = 'tfidf_matrix.pkl'
DOCUMENTS_PATH = 'documents.pkl'
DATA_MATKUL_LIST_PATH = 'data_matkul_list.pkl'
DOSEN_LIST_PATH = 'dosen_list.pkl'
DATA_FTMM_LIST_PATH = 'data_ftmm_list.pkl'
DATA_AKADEMIK_LIST_PATH = 'data_akademik_list.pkl'
DATA_VALIDASI_PATH = 'data/data_validasi4.csv'

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
        f"NIP: {row['NIP']}\n"
        f"Pendidikan: {row['Pendidikan']}\n"
        f"Research Interest: {row['Research Interest']}\n"
        f"Prodi: {row['Prodi']}\n"
        f"Email: {row['Email']}\n"
        f"Description: {row['Description']}\n"
        f"Portofolio: {row['Portofolio']}"
    )

def create_konten_teks_ftmm(row):
    return (
        f"Tentang Fakultas: {row['Tentang Fakultas']}\n"
        f"Akreditasi: {row['Akreditasi']}\n"
        f"Program Studi: {row['Program Studi']}\n"
        f"Sejarah Fakultas: {row['Sejarah Fakultas']}\n"
        f"Hasil Karya FTMM: {row['Hasil Karya FTMM']}\n"
        f"Prestasi Mahasiswa FTMM: {row['Prestasi Mahasiswa FTMM']}\n"
        f"Visi: {row['Visi']}\n"
        f"Misi: {row['Misi']}\n"
        f"Lokasi: {row['Lokasi']}\n"
        f"Kontak Fakultas: {row['Kontak Fakultas']}\n"
        f"Media Sosial: {row['Media Sosial']}\n"
        f"Warna Bendera: {row['Warna Bendera']}\n"
        f"Filosofi Bendera: {row['Filosofi Bendera']}\n"
        f"Filosofi Logo: {row['Filosofi Logo']}\n"
        f"Lokasi Kampus: {row['Lokasi Kampus']}\n"
        f"Maskot Cirion: {row['Maskot Cirion']}"
    )

def create_konten_teks_akademik(row):
    return (
        f"Akademik: {row['Akademik']}\n"
        f"Link: {row['Link']}\n"
        f"Deskripsi: {row['Deskripsi']}\n"
        f"Informasi Tambahan: {row['Informasi Tambahan']}"
    )

def load_tfidf():
    if all(os.path.exists(path) for path in [
        TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, DOCUMENTS_PATH,
        DATA_MATKUL_LIST_PATH, DOSEN_LIST_PATH, DATA_FTMM_LIST_PATH, DATA_AKADEMIK_LIST_PATH
    ]):
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
        documents = joblib.load(DOCUMENTS_PATH)
        data_matkul_list = joblib.load(DATA_MATKUL_LIST_PATH)
        dosen_list = joblib.load(DOSEN_LIST_PATH)
        data_ftmm_list = joblib.load(DATA_FTMM_LIST_PATH)
        data_akademik_list = joblib.load(DATA_AKADEMIK_LIST_PATH)
        logger.info("TF-IDF dan dokumen berhasil dimuat dari file.")
    else:
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

        data_dosen_path = 'data/data_dosen.csv'
        if not os.path.exists(data_dosen_path):
            raise FileNotFoundError(f"File {data_dosen_path} tidak ditemukan.")
        data_dosen_df = pd.read_csv(data_dosen_path, encoding='utf-8')
        required_columns_dosen = [
            'Nama', 'NIP', 'Pendidikan', 'Research Interest',
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

        data_ftmm_path = 'data/data_ftmm.csv'
        if not os.path.exists(data_ftmm_path):
            raise FileNotFoundError(f"File {data_ftmm_path} tidak ditemukan.")
        data_ftmm_df = pd.read_csv(data_ftmm_path, encoding='utf-8')
        required_columns_ftmm = [
            'Tentang Fakultas', 'Akreditasi', 'Program Studi', 'Sejarah Fakultas',
            'Hasil Karya FTMM', 'Prestasi Mahasiswa FTMM', 'Visi', 'Misi',
            'Lokasi', 'Kontak Fakultas', 'Media Sosial', 'Warna Bendera',
            'Filosofi Bendera', 'Filosofi Logo', 'Lokasi Kampus', 'Maskot Cirion'
        ]
        for col in required_columns_ftmm:
            if col not in data_ftmm_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV FTMM.")
            if data_ftmm_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV FTMM.")
        data_ftmm_df['Konten Teks'] = data_ftmm_df.apply(create_konten_teks_ftmm, axis=1)
        data_ftmm_list = data_ftmm_df.to_dict(orient='records')
        
        logger.info(f"Loaded {len(data_ftmm_list)} FTMM.")

        data_akademik_path = 'data/data_akademik.csv'
        if not os.path.exists(data_akademik_path):
            raise FileNotFoundError(f"File {data_akademik_path} tidak ditemukan.")
        data_akademik_df = pd.read_csv(data_akademik_path, encoding='utf-8')
        required_columns_akademik = [
            'Akademik', 'Link', 'Deskripsi', 'Informasi Tambahan'
        ]
        for col in required_columns_akademik:
            if col not in data_akademik_df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam CSV Akademik.")
            if data_akademik_df[col].isnull().any():
                raise ValueError(f"Ada nilai kosong di kolom '{col}' dalam CSV Akademik.")
        data_akademik_df['Konten Teks'] = data_akademik_df.apply(create_konten_teks_akademik, axis=1)
        data_akademik_list = data_akademik_df.to_dict(orient='records')
        
        logger.info(f"Loaded {len(data_akademik_list)} Akademik.")

        documents = (
            [{'Type': 'Mata Kuliah', 'Konten Teks': mk['Konten Teks']} for mk in data_matkul_list] +
            [{'Type': 'Dosen', 'Konten Teks': ds['Konten Teks']} for ds in dosen_list] +
            [{'Type': 'FTMM', 'Konten Teks': ft['Konten Teks']} for ft in data_ftmm_list] +
            [{'Type': 'Akademik', 'Konten Teks': ak['Konten Teks']} for ak in data_akademik_list]
        )
        
        logger.info(f"Total Documents Combined: {len(documents)}")

        stop_words = get_stop_words('id')
        vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([doc['Konten Teks'] for doc in documents])
        
        logger.info("TF-IDF Matrix Created.")

        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
        joblib.dump(documents, DOCUMENTS_PATH)
        joblib.dump(data_matkul_list, DATA_MATKUL_LIST_PATH)
        joblib.dump(dosen_list, DOSEN_LIST_PATH)
        joblib.dump(data_ftmm_list, DATA_FTMM_LIST_PATH)
        joblib.dump(data_akademik_list, DATA_AKADEMIK_LIST_PATH)
        logger.info("TF-IDF dan dokumen baru berhasil dibuat dan disimpan.")

    return vectorizer, tfidf_matrix, documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list

def find_most_relevant(query, vectorizer, tfidf_matrix, similarity_threshold=0.001):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    relevant_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    relevant_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    return relevant_indices, similarities[relevant_indices]

def generate_response(query, relevant_indices, chat_history, documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list):
    info_relevant = ""
    mata_kuliah_count = 0
    dosen_count = 0
    ftmm_count = 0
    akademik_count = 0

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
                    f"- {dosen['Nama']} (NIP: {dosen['NIP']}), "
                    f"Pendidikan: {dosen['Pendidikan']}, Research Interest: {dosen['Research Interest']}, "
                    f"Prodi: {dosen['Prodi']}, Email: {dosen['Email']}\n"
                )
                dosen_count += 1
            else:
                logger.warning(f"Indeks dosen {dosen_idx} di luar rentang list dosen.")
        elif doc['Type'] == 'FTMM':
            ftmm_idx = idx - len(data_matkul_list) - len(dosen_list)
            if 0 <= ftmm_idx < len(data_ftmm_list):
                ftmm = data_ftmm_list[ftmm_idx]
                info_relevant += (
                    f"- Tentang Fakultas: {ftmm['Tentang Fakultas']}\n"
                    f"- Akreditasi: {ftmm['Akreditasi']}\n"
                    f"- Program Studi: {ftmm['Program Studi']}\n"
                    f"- Sejarah Fakultas: {ftmm['Sejarah Fakultas']}\n"
                    f"- Hasil Karya FTMM: {ftmm['Hasil Karya FTMM']}\n"
                    f"- Prestasi Mahasiswa FTMM: {ftmm['Prestasi Mahasiswa FTMM']}\n"
                    f"- Visi: {ftmm['Visi']}\n"
                    f"- Misi: {ftmm['Misi']}\n"
                    f"- Lokasi: {ftmm['Lokasi']}\n"
                    f"- Kontak Fakultas: {ftmm['Kontak Fakultas']}\n"
                    f"- Media Sosial: {ftmm['Media Sosial']}\n"
                    f"- Warna Bendera: {ftmm['Warna Bendera']}\n"
                    f"- Filosofi Bendera: {ftmm['Filosofi Bendera']}\n"
                    f"- Filosofi Logo: {ftmm['Filosofi Logo']}\n"
                    f"- Lokasi Kampus: {ftmm['Lokasi Kampus']}\n"
                    f"- Maskot Cirion: {ftmm['Maskot Cirion']}\n"
                )
                ftmm_count += 1
            else:
                logger.warning(f"Indeks FTMM {ftmm_idx} di luar rentang list FTMM.")
        elif doc['Type'] == 'Akademik':
            akademik_idx = idx - len(data_matkul_list) - len(dosen_list) - len(data_ftmm_list)
            if 0 <= akademik_idx < len(data_akademik_list):
                akademik = data_akademik_list[akademik_idx]
                info_relevant += (
                    f"- Akademik: {akademik['Akademik']}\n"
                    f"- Link: {akademik['Link']}\n"
                    f"- Deskripsi: {akademik['Deskripsi']}\n"
                    f"- Informasi Tambahan: {akademik['Informasi Tambahan']}\n"
                )
                akademik_count += 1
            else:
                logger.warning(f"Indeks Akademik {akademik_idx} di luar rentang list Akademik.")

    logger.info(f"Total Mata Kuliah Relevan: {mata_kuliah_count}")
    logger.info(f"Total Dosen Relevan: {dosen_count}")
    logger.info(f"Total FTMM Relevan: {ftmm_count}")
    logger.info(f"Total Akademik Relevan: {akademik_count}")
    logger.debug(f"Informasi Relevan yang dikirim ke LLM:\n{info_relevant}")

    prompt_history = ""
    for exchange in chat_history:
        prompt_history += f"Pengguna: {exchange['user']}\n"
        prompt_history += f"Bot: {exchange['bot']}\n"

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
                "Anda adalah asisten AI yang membantu menjawab pertanyaan tentang dosen, mata kuliah, FTMM, dan akademik di FTMM Universitas Airlangga. "
                "Gunakan informasi dosen, mata kuliah, FTMM, dan akademik yang diberikan untuk menjawab pertanyaan pengguna secara detail dan natural. "
                "Jika pengguna bertanya tentang rekomendasi dosen atau mata kuliah, berikan saran yang sesuai dengan minat mereka berdasarkan research interest atau bidang keahlian dosen serta relevansi mata kuliah. "
                "Jika pertanyaan terkait FTMM, berikan informasi yang relevan tentang fakultas, akreditasi, program studi, sejarah, prestasi, visi, misi, dan lainnya sesuai permintaan. "
                "Jika pertanyaan terkait akademik, berikan informasi yang relevan tentang program akademik, link terkait, deskripsi, dan informasi tambahan sesuai permintaan. "
                "Jawablah dalam Bahasa Indonesia dengan gaya bahasa yang ramah dan profesional. "
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()

        logger.info(f"Generated answer: {answer}")

        return answer

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Terjadi kesalahan dalam memproses permintaan: {str(e)}"

def evaluate_model(vectorizer, tfidf_matrix, documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list):
    if not os.path.exists(DATA_VALIDASI_PATH):
        logger.error(f"File {DATA_VALIDASI_PATH} tidak ditemukan.")
        return {"error": f"File {DATA_VALIDASI_PATH} tidak ditemukan."}
    
    data_validasi_df = pd.read_csv(DATA_VALIDASI_PATH, encoding='utf-8')
    
    required_columns_validasi = ['query', 'relevant_documents', 'reference_answer']
    for col in required_columns_validasi:
        if col not in data_validasi_df.columns:
            logger.error(f"Kolom '{col}' tidak ditemukan dalam CSV Validasi.")
            return {"error": f"Kolom '{col}' tidak ditemukan dalam CSV Validasi."}
    
    precision_list = []
    recall_list = []
    bleu_list = []
    rouge_list = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for index, row in data_validasi_df.iterrows():
        query = row['query']
        relevant_docs = set([doc.strip() for doc in row['relevant_documents'].split(',')])
        reference_answer = row['reference_answer']
        
        retrieved_indices, _ = find_most_relevant(query, vectorizer, tfidf_matrix)
        retrieved_docs = set()
        for idx in retrieved_indices:
            doc = documents[idx]
            if doc['Type'] == 'Mata Kuliah':
                match = re.search(r'Kode Mata Kuliah:\s*(\w+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1)
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'Dosen':
                match = re.search(r'NIP:\s*(\w+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1)
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'FTMM':
                match = re.search(r'Tentang Fakultas:\s*(.+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1).strip()
                    retrieved_docs.add(doc_id)
            elif doc['Type'] == 'Akademik':
                match = re.search(r'Akademik:\s*(.+)', doc['Konten Teks'])
                if match:
                    doc_id = match.group(1).strip()
                    retrieved_docs.add(doc_id)
        
        true_positives = len(retrieved_docs & relevant_docs)
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        precision_list.append(precision)
        recall_list.append(recall)
        
        answer = generate_response(query, retrieved_indices, [], documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list)
        
        reference_tokens = nltk.word_tokenize(reference_answer.lower())
        candidate_tokens = nltk.word_tokenize(answer.lower())
        bleu = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5))
        bleu_list.append(bleu)
        
        scores = scorer.score(reference_answer, answer)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        rouge_avg = (rouge1 + rouge2 + rougeL) / 3
        rouge_list.append(rouge_avg)
        
        logger.info(f"Evaluated Query {index + 1}/{len(data_validasi_df)}")

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

vectorizer, tfidf_matrix, documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list = load_tfidf()

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
            top_indices, top_similarities = find_most_relevant(query, vectorizer, tfidf_matrix)
            answer = generate_response(query, top_indices, session.get('chat_history', []), documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list)
            logger.info(f"Generated answer: {answer}")
            session['chat_history'].append({'user': query, 'bot': answer})
            session.modified = True
        except Exception as e:
            logger.error(f"Error: {e}")
            session['chat_history'].append({'user': query, 'bot': f"Terjadi kesalahan: {str(e)}"})
            session.modified = True
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
    metrics = evaluate_model(vectorizer, tfidf_matrix, documents, data_matkul_list, dosen_list, data_ftmm_list, data_akademik_list)
    
    if "error" in metrics:
        return render_template('evaluation.html', error=metrics["error"], current_year=current_year)
    
    return render_template('evaluation.html', metrics=metrics, current_year=current_year)

if __name__ == '__main__':
    os.makedirs('./flask_session/', exist_ok=True)
    app.run(debug=True)
