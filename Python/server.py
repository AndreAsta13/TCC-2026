from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import librosa
import ffmpeg
import os, tempfile

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LIMITE_SEGUNDOS = 3600

@app.route('/transcrever', methods=['POST'])
def transcrever():
    arquivo = request.files['file']
    extensao = os.path.splitext(arquivo.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp:
        arquivo.save(tmp.name)
        tmp_path = tmp.name

    audio_path = tmp_path  # por padrão usa o arquivo original

    try:
        # Se for MP4, extrai apenas o áudio em MP3
        if extensao == '.mp4':
            audio_path = tmp_path.replace('.mp4', '.mp3')
            ffmpeg.input(tmp_path).output(audio_path, vn=None, acodec='libmp3lame').run(quiet=True)

        # Verifica duração
        duracao = librosa.get_duration(path=audio_path)
        if duracao > LIMITE_SEGUNDOS:
            return jsonify({
                "erro": f"Áudio muito longo: {int(duracao)}s. Limite: {LIMITE_SEGUNDOS}s (1 hora)."
            }), 400

        # Transcreve
        with open(audio_path, 'rb') as f:
            transcricao = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        return jsonify({"texto": transcricao})

    finally:
        os.remove(tmp_path)
        if audio_path != tmp_path and os.path.exists(audio_path):
            os.remove(audio_path)  # limpa o MP3 extraído também

if __name__ == '__main__':
    app.run(debug=True)