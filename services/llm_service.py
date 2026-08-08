import logging


logger = logging.getLogger(__name__)


class LLMService:
    """Adapter OpenAI untuk menghasilkan jawaban dari context hasil retrieval."""

    def __init__(self, api_key=None, model_name="gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_answer(self, question, context, conversation_history=None):
        if not self.api_key:
            return (
                "Retrieval dataset lokal berhasil, tetapi jawaban belum dapat dibuat "
                "karena OPENAI_API_KEY belum dikonfigurasi."
            )

        conversation_history = conversation_history or []
        history_text = "\n".join(
            f"Pengguna: {exchange['user']}\nBot: {exchange['bot']}"
            for exchange in conversation_history
        )
        prompt = (
            f"{history_text}\nPengguna: {question}\n"
            f"Informasi Relevan:\n{context}\nBot:"
        )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten yang menjawab pertanyaan tentang FTMM "
                            "Universitas Airlangga. Gunakan hanya informasi relevan yang "
                            "diberikan. Jawab dalam Bahasa Indonesia dengan ramah dan "
                            "profesional."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            return answer.strip() if answer else "OpenAI tidak mengembalikan jawaban."
        except Exception as exc:
            logger.error("Permintaan OpenAI gagal. Tipe error: %s.", type(exc).__name__)
            return (
                "Permintaan ke OpenAI gagal. Periksa koneksi, API key, dan konfigurasi "
                "OPENAI_MODEL."
            )
