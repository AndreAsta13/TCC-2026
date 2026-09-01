from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import librosa
import ffmpeg
import os, tempfile, shutil, json
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline as PyannotePipeline
import difflib
import requests
CACHE_NOMES_PATH = "cache_nomes_ibge.json"

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LIMITE_SEGUNDOS = 3600
HF_TOKEN = os.getenv("HF_TOKEN")

_diarization_pipeline = None


# ---------------------------------------------------------------------------
# ENQUADRAMENTO + REMOÇÃO DE SILÊNCIO
# ---------------------------------------------------------------------------
def _enquadrar_e_remover_silencio(caminho_entrada, caminho_saida, top_db=40):
    y, sr = librosa.load(caminho_entrada, sr=None, mono=False)
    y_mono_deteccao = y if y.ndim == 1 else librosa.to_mono(y)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    print(f"    Enquadrando: frame={frame_length} amostras (25ms), hop={hop_length} amostras (10ms), sr={sr}Hz")

    intervalos = librosa.effects.split(
        y_mono_deteccao, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length
    )

    if len(intervalos) == 0:
        print("    Aviso: nenhum trecho de voz detectado, mantendo áudio original.")
        y_final = y
    else:
        if y.ndim == 1:
            y_final = np.concatenate([y[ini:fim] for ini, fim in intervalos])
        else:
            y_final = np.concatenate([y[:, ini:fim] for ini, fim in intervalos], axis=1)

    duracao_antes = y_mono_deteccao.shape[-1] / sr
    duracao_depois = y_final.shape[-1] / sr
    print(f"    Duração: {duracao_antes:.2f}s -> {duracao_depois:.2f}s após remoção de silêncio")

    dados_para_salvar = y_final.T if y_final.ndim > 1 else y_final
    sf.write(caminho_saida, dados_para_salvar, sr)


# ---------------------------------------------------------------------------
# NORMALIZAÇÃO EBU R128 — MEDIÇÃO
# ---------------------------------------------------------------------------
def _medir_loudness(caminho_entrada):
    out, err = (
        ffmpeg
        .input(caminho_entrada)
        .output('-', af='loudnorm=I=-23:LRA=7:TP=-2:print_format=json', format='null')
        .run(capture_stdout=True, capture_stderr=True)
    )
    saida = err.decode('utf-8')
    inicio = saida.rfind('{')
    fim = saida.rfind('}') + 1
    return json.loads(saida[inicio:fim])


# ---------------------------------------------------------------------------
# DIARIZAÇÃO (pyannote.audio, local)
# ---------------------------------------------------------------------------
def _carregar_pipeline_diarizacao():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN não encontrado no .env. Necessário para usar o pyannote.audio.")
        print("    Carregando modelo de diarização (pode demorar na primeira vez)...")
        _diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
        )
    return _diarization_pipeline


def _diarizar_audio(caminho_wav):
    pipeline = _carregar_pipeline_diarizacao()
    resultado = pipeline(caminho_wav)
    segmentos = []
    for turno, _, falante in resultado.itertracks(yield_label=True):
        segmentos.append({"inicio": round(turno.start, 2), "fim": round(turno.end, 2), "falante": falante})
    return segmentos


# ---------------------------------------------------------------------------
# TRANSCRIÇÃO COM TIMESTAMPS POR PALAVRA (Groq)
# ---------------------------------------------------------------------------
def _transcrever_com_timestamps(audio_path):
    with open(audio_path, 'rb') as f:
        resposta = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    return resposta.words


# ---------------------------------------------------------------------------
# COMBINAÇÃO: diarização + transcrição, por palavra
# ---------------------------------------------------------------------------
def _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao):
    palavras_com_falante = []
    for palavra in palavras_transcricao:
        inicio_p, fim_p = palavra['start'], palavra['end']
        melhor_falante, maior_sobreposicao = "DESCONHECIDO", 0.0
        for seg_falante in segmentos_falantes:
            sobreposicao = min(fim_p, seg_falante['fim']) - max(inicio_p, seg_falante['inicio'])
            if sobreposicao > maior_sobreposicao:
                maior_sobreposicao = sobreposicao
                melhor_falante = seg_falante['falante']
        palavras_com_falante.append({
            "inicio": round(inicio_p, 2), "fim": round(fim_p, 2),
            "falante": melhor_falante, "palavra": palavra['word'].strip()
        })

    if not palavras_com_falante:
        return []

    blocos = []
    bloco_atual = {
        "inicio": palavras_com_falante[0]["inicio"], "fim": palavras_com_falante[0]["fim"],
        "falante": palavras_com_falante[0]["falante"], "palavras": [palavras_com_falante[0]["palavra"]]
    }

    for p in palavras_com_falante[1:]:
        if p["falante"] == bloco_atual["falante"]:
            bloco_atual["fim"] = p["fim"]
            bloco_atual["palavras"].append(p["palavra"])
        else:
            blocos.append({
                "inicio": bloco_atual["inicio"], "fim": bloco_atual["fim"],
                "falante": bloco_atual["falante"], "texto": " ".join(bloco_atual["palavras"])
            })
            bloco_atual = {"inicio": p["inicio"], "fim": p["fim"], "falante": p["falante"], "palavras": [p["palavra"]]}

    blocos.append({
        "inicio": bloco_atual["inicio"], "fim": bloco_atual["fim"],
        "falante": bloco_atual["falante"], "texto": " ".join(bloco_atual["palavras"])
    })

    return blocos


# ---------------------------------------------------------------------------
# CORREÇÃO DE TEXTO (LLM via Groq) — pontuação, capitalização, termos técnicos
# ---------------------------------------------------------------------------
def _corrigir_texto(texto):
    """
    Usa um LLM (Groq) para corrigir pontuação, capitalização e ajustar
    termos técnicos que o Whisper pode ter transcrito de forma imprecisa.
    """
    prompt = (
        "Você é um corretor de texto. Corrija SOMENTE pontuação, capitalização e "
        "ajuste termos técnicos que possam estar errados por transcrição automática. "
        "NÃO adicione, remova ou reescreva o conteúdo. NÃO resuma. "
        "Responda apenas com o texto corrigido, sem comentários.\n\n"
        f"Texto original: {texto}"
    )

    resposta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip()


def _corrigir_blocos(blocos):
    """Aplica a correção de texto em cada bloco de fala."""
    for bloco in blocos:
        bloco["texto_corrigido"] = _corrigir_texto(bloco["texto"])
    return blocos


# ---------------------------------------------------------------------------
# CONSULTA (ADICIONAL/OPCIONAL) — correção de nomes próprios
# ---------------------------------------------------------------------------
def _consultar_nomes_proprios(texto, base_nomes=None):
    """
    Passo ADICIONAL opcional: tenta identificar e corrigir substantivos próprios
    (nomes de pessoas) contra uma base de nomes conhecidos.

    IMPORTANTE: isso ainda precisa de uma fonte de dados real. Opções para conectar:
    - Base de nomes do IBGE (https://servicodados.ibge.gov.br/api/docs/nomes) — nomes
      mais comuns no Brasil, gratuita, sem necessidade de chave de API.
    - Uma lista própria carregada de um arquivo CSV/JSON com nomes esperados
      (ex: nomes de participantes de uma entrevista específica do seu TCC).

    Esta função está com uma implementação mínima usando uma lista local passada
    por parâmetro. Troque `base_nomes` por uma consulta real à API do IBGE ou
    outra fonte, se quiser essa correção automática de fato.
    """
    if not base_nomes:
        return texto  # sem base de nomes definida, não faz nada

    palavras = texto.split()
    palavras_corrigidas = []

    for palavra in palavras:
        palavra_limpa = palavra.strip('.,!?;:')
        # Busca aproximada simples (case-insensitive) contra a base de nomes
        correspondencia = next(
            (nome for nome in base_nomes if nome.lower() == palavra_limpa.lower()), None
        )
        if correspondencia:
            palavras_corrigidas.append(palavra.replace(palavra_limpa, correspondencia))
        else:
            palavras_corrigidas.append(palavra)

    return " ".join(palavras_corrigidas)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def transcrever_arquivo(caminho_original, base_nomes=None):
    """Recebe um caminho de arquivo já existente em disco e retorna a transcrição com falantes."""
    extensao = os.path.splitext(caminho_original)[1].lower()
    print(f"[1/9] Copiando arquivo temporário...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp:
        shutil.copyfile(caminho_original, tmp.name)
        tmp_path = tmp.name

    audio_path = tmp_path
    sem_silencio_path = None

    try:
        # --- Verificação de duração logo após a recepção (movida para cá) ---
        print(f"[2/9] Verificando duração do arquivo recebido...")
        duracao_bruta = librosa.get_duration(path=tmp_path)
        print(f"[2/9] Duração: {int(duracao_bruta)}s")

        if duracao_bruta > LIMITE_SEGUNDOS:
            raise ValueError(f"Áudio muito longo: {int(duracao_bruta)}s. Limite: {LIMITE_SEGUNDOS}s (1 hora).")

        if extensao in ('.mp4', '.mp3'):
            print(f"[3/9] Enquadrando e removendo trechos de silêncio (VAD por energia, 25ms/10ms, top_db=40)...")
            sem_silencio_path = tmp_path.rsplit('.', 1)[0] + '_sem_silencio.wav'
            _enquadrar_e_remover_silencio(tmp_path, sem_silencio_path, top_db=40)
            print(f"[3/9] Concluído: {sem_silencio_path}")

            print(f"[4/9] Verificando canais de áudio...")
            probe = ffmpeg.probe(sem_silencio_path)
            audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

            if not audio_streams:
                raise ValueError("Nenhuma trilha de áudio encontrada no arquivo.")

            canais = audio_streams[0].get('channels', 1)
            layout = audio_streams[0].get('channel_layout', 'desconhecido')
            taxa_original = audio_streams[0].get('sample_rate', 'desconhecida')

            audio_path = tmp_path.rsplit('.', 1)[0] + '.wav'

            if canais == 1:
                print(f"[5/9] Áudio detectado: mono (taxa: {taxa_original} Hz)")
                print(f"[5/9] Convertendo para WAV com normalização EBU R128 (single-pass)...")
                (
                    ffmpeg.input(sem_silencio_path)
                    .output(audio_path, vn=None, acodec='pcm_s16le', ac=1,
                            af='loudnorm=I=-23:LRA=7:TP=-2')
                    .run(quiet=True, overwrite_output=True)
                )
            else:
                tipo_audio = "estéreo" if canais == 2 else f"multicanal ({canais} canais)"
                print(f"[5/9] Áudio detectado: {tipo_audio} (layout: {layout}, taxa: {taxa_original} Hz)")

                print(f"[5a/9] Medindo loudness (passada 1/2)...")
                medidas = _medir_loudness(sem_silencio_path)
                print(f"[5a/9] Medido: I={medidas['input_i']} LUFS, TP={medidas['input_tp']} dBTP")

                print(f"[5b/9] Reamostrando 16kHz + normalização precisa (passada 2/2)...")
                filtro_preciso = (
                    f"loudnorm=I=-23:LRA=7:TP=-2:"
                    f"measured_I={medidas['input_i']}:"
                    f"measured_LRA={medidas['input_lra']}:"
                    f"measured_TP={medidas['input_tp']}:"
                    f"measured_thresh={medidas['input_thresh']}:"
                    f"offset={medidas['target_offset']}:linear=true"
                )
                (
                    ffmpeg.input(sem_silencio_path)
                    .output(audio_path, vn=None, acodec='pcm_s16le', ac=canais,
                            ar=16000, af=filtro_preciso)
                    .run(quiet=True, overwrite_output=True)
                )

            print(f"[5/9] WAV gerado: {audio_path}")

        # --- Diarização ---
        print(f"[6/9] Identificando falantes (diarização)...")
        segmentos_falantes = _diarizar_audio(audio_path)
        print(f"[6/9] {len(segmentos_falantes)} segmentos de fala identificados")

        # --- Transcrição com timestamps por palavra ---
        print(f"[7/9] Transcrevendo com timestamps por palavra (Groq)...")
        palavras_transcricao = _transcrever_com_timestamps(audio_path)

        # --- Combinação ---
        resultado = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)

        # --- Correção de texto (LLM) ---
        print(f"[8/9] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
        resultado = _corrigir_blocos(resultado)

        # --- Consulta de nomes próprios (adicional/opcional) ---
        print(f"[9/9] Consultando nomes próprios conhecidos (opcional)...")
        for bloco in resultado:
            bloco["texto_corrigido"] = _consultar_nomes_proprios(bloco["texto_corrigido"], base_nomes)

        print("\n--- RESULTADO FINAL ---")
        for item in resultado:
            print(f"[{item['inicio']}s - {item['fim']}s] {item['falante']}: {item['texto_corrigido']}")

        return resultado

    finally:
        os.remove(tmp_path)
        if sem_silencio_path and os.path.exists(sem_silencio_path):
            os.remove(sem_silencio_path)
        if audio_path != tmp_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.route('/transcrever', methods=['POST'])
def transcrever():
    arquivo = request.files['file']
    extensao = os.path.splitext(arquivo.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp:
        arquivo.save(tmp.name)
        tmp_path = tmp.name

    try:
        resultado = transcrever_arquivo(tmp_path)
        return jsonify({"segmentos": resultado})
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    finally:
        os.remove(tmp_path)


if __name__ == '__main__':
    TESTE_LOCAL = False

    if TESTE_LOCAL:
        caminho = r"../Teste Video/Teste1.mp4"
        resultado = transcrever_arquivo(caminho)
    else:
        app.run(debug=True)