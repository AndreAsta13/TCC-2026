"""
ground_truth_server.py — Transcreva.IA
Gerencia o dataset de Ground Truth com suporte a AWS S3 e Google Cloud Storage.
Instale dependências:
  pip install flask flask-cors groq boto3 google-cloud-storage jiwer python-dotenv yt-dlp
"""

import os
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq


from jiwer import wer as calc_wer, cer as calc_cer


try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from yt_dlp import YoutubeDL
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False


load_dotenv(dotenv_path=Path(__file__).parent / ".env")
app = Flask(__name__)
CORS(app)

GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Configuração de armazenamento ─────────────────────────────────────────────
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")  # "local" | "s3" | "gcs"

# AWS S3
AWS_BUCKET      = os.environ.get("AWS_BUCKET_NAME", "transcreva-ground-truth")
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY  = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Google Cloud Storage
GCS_BUCKET      = os.environ.get("GCS_BUCKET_NAME", "transcreva-ground-truth")
GCS_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # caminho do .json

# Armazenamento local (fallback)
LOCAL_GT_DIR    = Path("ground_truth_dataset")
LOCAL_GT_DIR.mkdir(exist_ok=True)
GT_INDEX_FILE   = LOCAL_GT_DIR / "index.json"


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA DE ARMAZENAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def get_s3_client():
    if not AWS_AVAILABLE:
        raise RuntimeError("boto3 não instalado. Execute: pip install boto3")
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def get_gcs_client():
    if not GCS_AVAILABLE:
        raise RuntimeError("google-cloud-storage não instalado.")
    return gcs_storage.Client()


def upload_audio(local_path: str, filename: str) -> str:
    """Faz upload do áudio para o backend configurado (STORAGE_BACKEND) e retorna a URL/chave."""
    if STORAGE_BACKEND == "s3":
        s3 = get_s3_client()
        key = f"audio/{filename}"
        s3.upload_file(local_path, AWS_BUCKET, key)
        url = f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        return url

    elif STORAGE_BACKEND == "gcs":
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(f"audio/{filename}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url

    else:  # local
        dest = LOCAL_GT_DIR / "audio" / filename
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(local_path, dest)
        return str(dest)


def upload_transcript(text: str, filename: str) -> str:
    """Salva a transcrição ground truth e retorna a URL/caminho."""
    txt_filename = filename.rsplit(".", 1)[0] + ".txt"

    if STORAGE_BACKEND == "s3":
        s3  = get_s3_client()
        key = f"transcripts/{txt_filename}"
        s3.put_object(Body=text.encode("utf-8"), Bucket=AWS_BUCKET, Key=key)
        return f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    elif STORAGE_BACKEND == "gcs":
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(f"transcripts/{txt_filename}")
        blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        blob.make_public()
        return blob.public_url

    else:  # local
        dest = LOCAL_GT_DIR / "transcripts" / txt_filename
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return str(dest)


def upload_to_gcs_forced(local_path: str, filename: str, subfolder: str = "audio") -> str:
    """
    Sobe um arquivo para o Google Cloud Storage independentemente do STORAGE_BACKEND
    configurado no .env. Usado pelo fluxo de importação de vídeos do YouTube, que
    sempre deve ir parar no GCS.
    """
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(f"{subfolder}/{filename}")
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def load_index() -> list:
    if GT_INDEX_FILE.exists():
        return json.loads(GT_INDEX_FILE.read_text(encoding="utf-8"))
    return []


def save_index(data: list):
    GT_INDEX_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE WER
# ══════════════════════════════════════════════════════════════════════════════

def normalizar(texto: str) -> str:
    """Normalização simples: minúsculas + remove pontuação."""
    import re
    texto = texto.lower().strip()
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto


def calcular_wer(ground_truth: str, hipotese: str) -> dict:
    gt_norm  = normalizar(ground_truth)
    hip_norm = normalizar(hipotese)
    wer_val  = round(calc_wer(gt_norm, hip_norm) * 100, 2)
    cer_val  = round(calc_cer(gt_norm, hip_norm) * 100, 2)
    return {
        "wer_percent": wer_val,
        "cer_percent": cer_val,
        "ground_truth_palavras": len(gt_norm.split()),
        "hipotese_palavras":     len(hip_norm.split()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROTAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/transcrever", methods=["POST"])
def transcrever():
    """Transcrição original do Transcreva.IA via Groq."""
    arquivo = request.files.get("file")
    if not arquivo:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    with tempfile.NamedTemporaryFile(suffix=Path(arquivo.filename).suffix, delete=False) as tmp:
        arquivo.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resultado = GROQ_CLIENT.audio.transcriptions.create(
                file=(arquivo.filename, f),
                model="whisper-large-v3",
                language="pt",
                response_format="text",
            )
        return jsonify({"texto": resultado})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        os.unlink(tmp_path)


# ── Ground Truth ──────────────────────────────────────────────────────────────

@app.route("/gt/adicionar", methods=["POST"])
def gt_adicionar():
    """
    Adiciona um arquivo ao dataset de Ground Truth.
    Form-data:
      - file        : arquivo de áudio
      - transcricao : texto exato (ground truth manual)
      - fonte        : origem do áudio (ex: "Common Voice")
    """
    arquivo     = request.files.get("file")
    transcricao = request.form.get("transcricao", "").strip()
    fonte       = request.form.get("fonte", "desconhecida")

    if not arquivo or not transcricao:
        return jsonify({"erro": "Envie o arquivo e a transcrição manual."}), 400

    ext      = Path(arquivo.filename).suffix.lower()
    novo_id  = str(uuid.uuid4())[:8]
    filename = f"gt_{novo_id}{ext}"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        arquivo.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio_url      = upload_audio(tmp_path, filename)
        transcript_url = upload_transcript(transcricao, filename)
    finally:
        os.unlink(tmp_path)

    # Transcreve com Groq para já calcular o WER inicial
    try:
        with open(audio_url if STORAGE_BACKEND == "local" else tmp_path, "rb") as f:
            resultado_groq = GROQ_CLIENT.audio.transcriptions.create(
                file=(filename, f),
                model="whisper-large-v3",
                language="pt",
                response_format="text",
            )
        metricas = calcular_wer(transcricao, resultado_groq)
    except Exception:
        resultado_groq = None
        metricas = {}

    entrada = {
        "id":             novo_id,
        "filename":       filename,
        "fonte":          fonte,
        "audio_url":      audio_url,
        "transcript_url": transcript_url,
        "ground_truth":   transcricao,
        "palavras_gt":    len(transcricao.split()),
        "wer":            metricas.get("wer_percent"),
        "cer":            metricas.get("cer_percent"),
        "status":         "validado" if metricas else "pendente",
        "storage":        STORAGE_BACKEND,
    }

    index = load_index()
    index.append(entrada)
    save_index(index)

    return jsonify({"sucesso": True, "entrada": entrada})


@app.route("/gt/youtube", methods=["POST"])
def gt_youtube():
    """
    Baixa o áudio de um vídeo do YouTube, sobe SEMPRE para o Google Cloud Storage
    (independente do STORAGE_BACKEND configurado) e cadastra a entrada no dataset
    de Ground Truth. Por enquanto não pede a transcrição manual — a entrada entra
    como "pendente_transcricao_manual" e pode ser completada depois.

    JSON body:
      - url   : link do vídeo do YouTube (obrigatório)
      - fonte : origem, opcional (default: "YouTube")
    """
    if not YTDLP_AVAILABLE:
        return jsonify({"erro": "yt-dlp não instalado. Execute: pip install yt-dlp"}), 500
    if not GCS_AVAILABLE:
        return jsonify({"erro": "google-cloud-storage não instalado."}), 500

    body  = request.get_json(force=True, silent=True) or {}
    url   = body.get("url", "").strip()
    fonte = body.get("fonte", "YouTube")

    if not url:
        return jsonify({"erro": "Envie o campo 'url' com o link do YouTube."}), 400

    novo_id  = str(uuid.uuid4())[:8]
    tmp_dir  = tempfile.mkdtemp(prefix=f"gt_yt_{novo_id}_")
    filename = f"gt_{novo_id}.mp3"

    try:
        # ── Download do áudio via yt-dlp ────────────────────────────────────
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp_dir, f"gt_{novo_id}.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                titulo = info.get("title", "Vídeo do YouTube")
                duracao = info.get("duration")
        except Exception as e:
            return jsonify({"erro": f"Falha ao baixar o vídeo: {e}"}), 502

        tmp_path = os.path.join(tmp_dir, filename)
        if not os.path.exists(tmp_path):
            return jsonify({"erro": "Não foi possível localizar o áudio extraído após o download."}), 500

        # ── Upload forçado para o GCS ───────────────────────────────────────
        try:
            audio_url = upload_to_gcs_forced(tmp_path, filename, subfolder="audio")
        except Exception as e:
            return jsonify({"erro": f"Falha ao subir para o Google Cloud Storage: {e}"}), 502

        # ── Transcrição automática via Groq (referência inicial, não é o ground truth) ──
        try:
            with open(tmp_path, "rb") as f:
                resultado_groq = GROQ_CLIENT.audio.transcriptions.create(
                    file=(filename, f),
                    model="whisper-large-v3",
                    language="pt",
                    response_format="text",
                )
        except Exception:
            resultado_groq = None

        entrada = {
            "id":                       novo_id,
            "filename":                 filename,
            "fonte":                    fonte,
            "origem_youtube":           url,
            "titulo_youtube":           titulo,
            "duracao_segundos":         duracao,
            "audio_url":                audio_url,
            "transcript_url":           None,
            "ground_truth":             None,
            "transcricao_automatica":   resultado_groq,
            "palavras_gt":              0,
            "wer":                      None,
            "cer":                      None,
            "status":                   "pendente",
            "precisa_transcricao_manual": True,
            "storage":                  "gcs",
        }

        index = load_index()
        index.append(entrada)
        save_index(index)

        return jsonify({"sucesso": True, "entrada": entrada})

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/gt/ground_truth/<arquivo_id>", methods=["POST"])
def gt_definir_ground_truth(arquivo_id):
    """
    Cadastra/atualiza a transcrição manual (ground truth) de uma entrada já
    existente no dataset — usado principalmente para completar as entradas
    importadas do YouTube, que entram sem ground truth.
    JSON body: { "transcricao": "texto exato do áudio" }
    """
    body        = request.get_json(force=True, silent=True) or {}
    transcricao = body.get("transcricao", "").strip()
    if not transcricao:
        return jsonify({"erro": "Envie o campo 'transcricao'."}), 400

    index   = load_index()
    entrada = next((e for e in index if e["id"] == arquivo_id), None)
    if not entrada:
        return jsonify({"erro": "Arquivo não encontrado no dataset."}), 404

    transcript_url = upload_transcript(transcricao, entrada["filename"])

    entrada["ground_truth"]              = transcricao
    entrada["transcript_url"]            = transcript_url
    entrada["palavras_gt"]               = len(transcricao.split())
    entrada["precisa_transcricao_manual"] = False

    # Se já existe uma transcrição automática (ex: da importação do YouTube),
    # aproveita pra calcular o WER inicial na hora.
    if entrada.get("transcricao_automatica"):
        metricas = calcular_wer(transcricao, entrada["transcricao_automatica"])
        entrada["wer"] = metricas["wer_percent"]
        entrada["cer"] = metricas["cer_percent"]
        entrada["status"] = "validado"
    else:
        entrada["status"] = "pendente"

    save_index(index)
    return jsonify({"sucesso": True, "entrada": entrada})


@app.route("/gt/listar", methods=["GET"])
def gt_listar():
    """Retorna todos os arquivos do dataset."""
    return jsonify(load_index())


@app.route("/gt/wer/<arquivo_id>", methods=["POST"])
def gt_wer(arquivo_id):
    """
    Recalcula o WER de um arquivo do dataset com uma nova hipótese.
    JSON body: { "hipotese": "texto gerado pela IA" }
    """
    body     = request.get_json(force=True)
    hipotese = body.get("hipotese", "").strip()
    if not hipotese:
        return jsonify({"erro": "Envie o campo 'hipotese'."}), 400

    index  = load_index()
    entrada = next((e for e in index if e["id"] == arquivo_id), None)
    if not entrada:
        return jsonify({"erro": "Arquivo não encontrado no dataset."}), 404
    if not entrada.get("ground_truth"):
        return jsonify({"erro": "Este arquivo ainda não tem transcrição manual (ground truth). Use /gt/ground_truth/<id> para cadastrá-la primeiro."}), 400

    metricas = calcular_wer(entrada["ground_truth"], hipotese)
    entrada.update({"wer": metricas["wer_percent"], "cer": metricas["cer_percent"]})
    save_index(index)

    return jsonify({"id": arquivo_id, **metricas})


@app.route("/gt/resumo", methods=["GET"])
def gt_resumo():
    """Retorna estatísticas agregadas do dataset."""
    index = load_index()
    if not index:
        return jsonify({"total": 0, "wer_medio": None})

    wers = [e["wer"] for e in index if e.get("wer") is not None]
    return jsonify({
        "total":           len(index),
        "validados":       sum(1 for e in index if e.get("status") == "validado"),
        "pendentes":       sum(1 for e in index if e.get("status") == "pendente"),
        "wer_medio":       round(sum(wers) / len(wers), 2) if wers else None,
        "wer_melhor":      round(min(wers), 2) if wers else None,
        "wer_pior":        round(max(wers), 2) if wers else None,
        "palavras_total":  sum(e.get("palavras_gt", 0) for e in index),
        "storage_backend": STORAGE_BACKEND,
    })


@app.route("/storage/status", methods=["GET"])
def storage_status():
    """Verifica qual backend de armazenamento está ativo e acessível."""
    status = {"backend": STORAGE_BACKEND, "ok": False, "detalhes": ""}

    if STORAGE_BACKEND == "s3":
        try:
            s3 = get_s3_client()
            s3.head_bucket(Bucket=AWS_BUCKET)
            status.update({"ok": True, "detalhes": f"Bucket S3 '{AWS_BUCKET}' acessível."})
        except Exception as e:
            status["detalhes"] = str(e)

    elif STORAGE_BACKEND == "gcs":
        try:
            client = get_gcs_client()
            bucket = client.bucket(GCS_BUCKET)
            bucket.reload()
            status.update({"ok": True, "detalhes": f"Bucket GCS '{GCS_BUCKET}' acessível."})
        except Exception as e:
            status["detalhes"] = str(e)

    else:
        status.update({"ok": True, "detalhes": f"Armazenamento local em '{LOCAL_GT_DIR}'."})

    return jsonify(status)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Transcreva.IA] Storage backend: {STORAGE_BACKEND.upper()}")
    app.run(host="0.0.0.0", port=5000, debug=True)