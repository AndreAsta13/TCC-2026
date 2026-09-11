import sys

print("Importando Flask...", flush=True)
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Importando ffmpeg...", flush=True)
import ffmpeg

print("Importando difflib...", flush=True)
from difflib import SequenceMatcher
import re
import unicodedata

print("Importando librosa (pode demorar um pouco)...", flush=True)
import librosa

print("Importando Groq...", flush=True)
from groq import Groq
from dotenv import load_dotenv

print("Importando numpy/soundfile...", flush=True)
import os, tempfile, shutil, json
import numpy as np
import soundfile as sf
import concurrent.futures

# --- Carrega o .env ANTES de qualquer uso de variável de ambiente (ex: FFMPEG_BIN) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
print(f"Carregando .env: {ENV_PATH}", flush=True)
load_dotenv(ENV_PATH, override=True)

FFMPEG_BIN = os.getenv("FFMPEG_BIN")

if FFMPEG_BIN and os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]
    print(f"FFmpeg Shared configurado: {FFMPEG_BIN}", flush=True)
elif not FFMPEG_BIN:
    print("AVISO: variável FFMPEG_BIN não definida no .env. "
          "Adicione uma linha tipo: FFMPEG_BIN=C:\\ffmpeg\\ffmpeg-9.0.1-full_build-shared\\bin", flush=True)
else:
    print(f"AVISO: FFmpeg Shared não encontrado no caminho do .env: {FFMPEG_BIN}", flush=True)

print("Importando pyannote.audio (ESTA É A MAIS LENTA — pode levar de 1 a 3+ minutos)...", flush=True)
print("  -> carregando torch/lightning/torchmetrics por baixo dos panos...", flush=True)
from pyannote.audio import Pipeline as PyannotePipeline
print("pyannote.audio carregado com sucesso!", flush=True)

print("Importando difflib...", flush=True)
import difflib
import unicodedata

print("Importando DadosAbertosBrasil (ranking de nomes IBGE, sem o limite de 20 nomes da API crua)...", flush=True)
from DadosAbertosBrasil import ibge as dab_ibge

print("Todos os imports concluídos. Iniciando servidor...\n", flush=True)


CACHE_NOMES_PATH = "cache_nomes_ibge.json"

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LIMITE_SEGUNDOS = 3600
HF_TOKEN = os.getenv("HF_TOKEN")

_diarization_pipeline = None


# ---------------------------------------------------------------------------
# DETECÇÃO DE CANAIS (mono / estéreo / multicanal)
# ---------------------------------------------------------------------------
def _detectar_canais(caminho_entrada):
    """
    Inspeciona o arquivo via ffprobe (sem decodificar o áudio de verdade)
    e retorna um dicionário com o número de canais, layout, taxa de
    amostragem original e uma categoria (tipo_canal) usada para decidir
    a estratégia de processamento: "mono", "estereo" ou "multicanal".
    """
    probe = ffmpeg.probe(caminho_entrada)
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

    if not audio_streams:
        print("    ERRO: Nenhuma trilha de áudio encontrada no arquivo.")
        raise ValueError("Nenhuma trilha de áudio encontrada no arquivo.")

    canais = audio_streams[0].get('channels', 1)
    layout = audio_streams[0].get('channel_layout', 'desconhecido')
    taxa_original = audio_streams[0].get('sample_rate', 'desconhecida')

    if canais == 1:
        tipo_canal = "mono"
        tipo_audio = "mono"
        print(f"    Canal detectado: MONO (1 canal)")
    elif canais == 2:
        tipo_canal = "estereo"
        tipo_audio = "estéreo"
        print(f"    Canal detectado: ESTÉREO (2 canais)")
    else:
        tipo_canal = "multicanal"
        tipo_audio = f"multicanal ({canais} canais)"
        print(f"    Canal detectado: MULTICANAL ({canais} canais, layout: {layout})")

    return {
        "canais": canais,
        "layout": layout,
        "taxa_original": taxa_original,
        "tipo_audio": tipo_audio,
        "tipo_canal": tipo_canal  # "mono" | "estereo" | "multicanal" — usado nas decisões do pipeline
    }


# ---------------------------------------------------------------------------
# CONVERSÃO LOSSLESS PARA WAV (antes de qualquer normalização/VAD)
# ---------------------------------------------------------------------------
def _converter_para_wav_lossless(caminho_entrada, caminho_saida, canais):
    """
    Converte MP4/MP3 para WAV sem perda de qualidade (PCM 16-bit),
    preservando canais e taxa de amostragem original. Sem normalização
    nem reamostragem ainda — isso acontece na etapa seguinte.
    """
    (
        ffmpeg
        .input(caminho_entrada)
        .output(caminho_saida, vn=None, acodec='pcm_s16le', ac=canais)
        .run(quiet=True, overwrite_output=True)
    )


# ---------------------------------------------------------------------------
# SEPARAÇÃO DE CANAIS — SÓ PARA ESTÉREO
# (assume que cada canal do estéreo carrega uma voz/falante diferente,
#  ex: microfone de entrevista com um falante por canal)
# ---------------------------------------------------------------------------
def _separar_canais_estereo(caminho_entrada, caminho_esquerdo, caminho_direito):
    """
    Separa um WAV estéreo em dois arquivos MONO independentes:
    canal esquerdo (voz 1) e canal direito (voz 2). Cada um vira um
    arquivo mono próprio, que segue o pipeline de forma independente.
    """
    (
        ffmpeg
        .input(caminho_entrada)
        .output(caminho_esquerdo, af='pan=mono|c0=FL', acodec='pcm_s16le')
        .run(quiet=True, overwrite_output=True)
    )
    (
        ffmpeg
        .input(caminho_entrada)
        .output(caminho_direito, af='pan=mono|c0=FR', acodec='pcm_s16le')
        .run(quiet=True, overwrite_output=True)
    )
   # ---------------------------------------------------------------------------
# SEPARAÇÃO DE CANAIS — MULTICANAL (generalização de _separar_canais_estereo
# para N canais, um arquivo mono por canal: c0, c1, ..., cN-1)
# ---------------------------------------------------------------------------
def _separar_canais_multicanal(caminho_entrada, num_canais):
    """
    Separa um WAV multicanal em N arquivos MONO independentes (um por
    canal). Mesma ideia de _separar_canais_estereo, só que para qualquer
    número de canais, não só 2.
    """
    caminhos = []
    base = caminho_entrada.rsplit('.', 1)[0]
    for i in range(num_canais):
        caminho_canal = f"{base}_canal{i}.wav"
        (
            ffmpeg
            .input(caminho_entrada)
            .output(caminho_canal, af=f'pan=mono|c0=c{i}', acodec='pcm_s16le')
            .run(quiet=True, overwrite_output=True)
        )
        caminhos.append(caminho_canal)
    return caminhos

def _run_ffmpeg(stream, contexto=""):
    """
    Roda um stream do ffmpeg-python capturando stdout/stderr de verdade,
    para que erros mostrem a causa real em vez de 'ffmpeg error (see stderr...)'.
    """
    try:
        out, err = stream.run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        return out, err
    except ffmpeg.Error as e:
        stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr capturado)"
        print(f"    [FFMPEG ERROR] {contexto}\n{stderr_txt}", flush=True)
        raise


# ---------------------------------------------------------------------------
# NORMALIZAÇÃO EBU R128 — SINGLE-PASS (reutilizável para qualquer mono:
# tanto o caminho MONO original quanto cada canal separado do ESTÉREO)
# ---------------------------------------------------------------------------
def _normalizar_mono_single_pass(caminho_entrada, caminho_saida):
    (
        ffmpeg
        .input(caminho_entrada)
        .output(caminho_saida, vn=None, acodec='pcm_s16le', ac=1,
                af='loudnorm=I=-23:LRA=7:TP=-2')
        .run(quiet=True, overwrite_output=True)
    )


# ---------------------------------------------------------------------------
# ENQUADRAMENTO
# (roda sobre o áudio já convertido em WAV E normalizado)
# Usado no caminho MONO — aqui o áudio É cortado, gerando um novo arquivo
# sem os trechos de silêncio.
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
# DETECÇÃO DE INTERVALOS DE FALA — SEM CORTAR O ÁUDIO (usado no ESTÉREO/MULTICANAL)
# ---------------------------------------------------------------------------
# Diferente de _enquadrar_e_remover_silencio, esta função NÃO gera um
# novo arquivo e NÃO corta nada — ela só identifica onde está a fala.
# Isso é importante para os caminhos com múltiplos canais: se cortássemos
# cada canal de forma independente (cada um com silêncios diferentes
# removidos), os timestamps deixariam de corresponder ao tempo real do
# áudio original, e a comparação entre canais (que depende de
# sobreposição de tempo) ficaria errada.
def _detectar_intervalos_fala(caminho_entrada, top_db=40):
    y, sr = librosa.load(caminho_entrada, sr=None, mono=True)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    intervalos = librosa.effects.split(
        y, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length
    )

    return intervalos



# ---------------------------------------------------------------------------
# NORMALIZAÇÃO EBU R128 — TWO-PASS PARA UM ÚNICO CANAL MONO
# (usado nos canais já separados do MULTICANAL — cada canal vira um
# arquivo mono independente, mas continua usando two-pass, seguindo a
# regra já combinada: a decisão de single/two-pass é pelo tipo de
# arquivo ORIGINAL — mono usa single-pass, multicanal usa two-pass,
# mesmo depois de cada canal do multicanal virar um arquivo mono próprio)
# ---------------------------------------------------------------------------
def _medir_loudness_mono(caminho_entrada):
    """Medição EBU R128 (passada 1/2) para um único arquivo mono."""
    try:
        out, err = (
            ffmpeg
            .input(caminho_entrada)
            .output('-', af='loudnorm=I=-23:LRA=7:TP=-2:print_format=json', format='null')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr)"
        print(f"    [FFMPEG ERROR] medir loudness de {caminho_entrada}:\n{stderr_txt}", flush=True)
        raise

    saida = err.decode('utf-8')
    inicio = saida.rfind('{')
    fim = saida.rfind('}') + 1
    medidas = json.loads(saida[inicio:fim])
    return medidas


def _normalizar_mono_two_pass(caminho_entrada, caminho_saida):
    """Normalização EBU R128 two-pass (mede, depois aplica com precisão) para um canal mono."""
    medidas = _medir_loudness_mono(caminho_entrada)

    # --- Blindagem contra canal silencioso (LFE, canal vazio em teste curto, etc.) ---
    # Quando o canal é silêncio digital puro, o loudnorm retorna measured_I = "-inf",
    # e o filtro two-pass com linear=true quebra ao receber esse valor.
    # Nesse caso, caímos para single-pass (não-linear), que tolera silêncio.
    try:
        measured_i = float(medidas['input_i'])
        measured_tp = float(medidas['input_tp'])
        canal_silencioso = measured_i == float('-inf') or measured_tp == float('-inf')
    except (ValueError, KeyError):
        canal_silencioso = True

    if canal_silencioso:
        print(f"    [AVISO] Canal praticamente silencioso detectado ({caminho_entrada}); "
              f"usando normalização single-pass (não-linear) em vez de two-pass.", flush=True)
        _normalizar_mono_single_pass(caminho_entrada, caminho_saida)
        return

    filtro_preciso = (
        f"loudnorm=I=-23:LRA=7:TP=-2:"
        f"measured_I={medidas['input_i']}:"
        f"measured_LRA={medidas['input_lra']}:"
        f"measured_TP={medidas['input_tp']}:"
        f"measured_thresh={medidas['input_thresh']}:"
        f"offset={medidas['target_offset']}:linear=true"
    )
    try:
        (
            ffmpeg
            .input(caminho_entrada)
            .output(caminho_saida, vn=None, acodec='pcm_s16le', ac=1, af=filtro_preciso)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr — rode com capture_stderr=True)"
        print(f"    [FFMPEG ERROR] normalizar two-pass {caminho_entrada}:\n{stderr_txt}", flush=True)
        raise



# ---------------------------------------------------------------------------
# Comparação de Falantes (difflib, re, unicodedata) — Estéreo
# ---------------------------------------------------------------------------

def _normalizar_texto_comparacao(texto):
    """
    Normaliza o texto apenas para comparação.
    Não altera o texto original da transcrição.
    """

    if not texto:
        return ""

    texto = texto.lower().strip()

    # Remove acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    # Remove pontuação
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Remove espaços duplicados
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def _similaridade_texto(texto_a, texto_b):
    """
    Retorna uma similaridade entre 0 e 1.
    1.0 = textos praticamente iguais
    0.0 = textos completamente diferentes
    """

    a = _normalizar_texto_comparacao(texto_a)
    b = _normalizar_texto_comparacao(texto_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def _calcular_sobreposicao(inicio_a, fim_a, inicio_b, fim_b):
    """
    Calcula quanto dois segmentos se sobrepõem.

    Retorna um valor entre 0 e 1 baseado no menor segmento.

    Exemplo:

    A = 10s -> 20s
    B = 12s -> 18s

    Sobreposição = 6s
    Menor segmento = 6s
    Resultado = 1.0
    """

    inicio_sobreposicao = max(inicio_a, inicio_b)
    fim_sobreposicao = min(fim_a, fim_b)

    if fim_sobreposicao <= inicio_sobreposicao:
        return 0.0

    duracao_sobreposicao = (
        fim_sobreposicao - inicio_sobreposicao 
    )

    duracao_a = fim_a - inicio_a
    duracao_b = fim_b - inicio_b

    menor_duracao = min(duracao_a, duracao_b)

    if menor_duracao <= 0:
        return 0.0

    return duracao_sobreposicao / menor_duracao


def _comparar_falas_estereo(
    segmentos_esquerda,
    segmentos_direita,
    limite_sobreposicao=0.70,
    limite_similaridade=0.80
):
    """
    Compara as falas dos canais esquerdo e direito.

    A função procura segmentos que:

    1. Acontecem praticamente no mesmo momento;
    2. Possuem texto muito semelhante.

    Quando os dois critérios são atendidos,
    considera-se que provavelmente é a mesma fala
    captada pelos dois canais.

    O canal esquerdo é usado como referência quando
    os dois segmentos são considerados duplicados.
    """

    resultado_esquerda = []
    resultado_direita = []

    # ---------------------------------------------------------
    # COPIA OS SEGMENTOS PARA NÃO ALTERAR OS ORIGINAIS
    # ---------------------------------------------------------

    for segmento in segmentos_esquerda:

        novo_segmento = segmento.copy()

        novo_segmento["canal"] = "L"
        novo_segmento["duplicado"] = False
        novo_segmento["similaridade"] = 0.0
        novo_segmento["sobreposicao"] = 0.0

        resultado_esquerda.append(novo_segmento)

    for segmento in segmentos_direita:

        novo_segmento = segmento.copy()

        novo_segmento["canal"] = "R"
        novo_segmento["duplicado"] = False
        novo_segmento["similaridade"] = 0.0
        novo_segmento["sobreposicao"] = 0.0

        resultado_direita.append(novo_segmento)

    # ---------------------------------------------------------
    # COMPARA L CONTRA R
    # ---------------------------------------------------------

    for segmento_l in resultado_esquerda:

        melhor_correspondencia = None
        melhor_score = 0.0

        for segmento_r in resultado_direita:

            # ---------------------------------------------
            # SOBREPOSIÇÃO TEMPORAL
            # ---------------------------------------------

            sobreposicao = _calcular_sobreposicao(
                segmento_l["inicio"],
                segmento_l["fim"],
                segmento_r["inicio"],
                segmento_r["fim"]
            )

            # Se praticamente não existe sobreposição,
            # não faz sentido comparar como duplicação.
            if sobreposicao < limite_sobreposicao:
                continue

            # ---------------------------------------------
            # SIMILARIDADE DO TEXTO
            # ---------------------------------------------

            similaridade = _similaridade_texto(
                segmento_l.get("texto", ""),
                segmento_r.get("texto", "")
            )

            if similaridade < limite_similaridade:
                continue

            # ---------------------------------------------
            # SCORE FINAL
            # ---------------------------------------------

            score = (
                sobreposicao * 0.5
                +
                similaridade * 0.5
            )

            if score > melhor_score:

                melhor_score = score

                melhor_correspondencia = {
                    "segmento": segmento_r,
                    "sobreposicao": sobreposicao,
                    "similaridade": similaridade,
                    "score": score
                }

        # -----------------------------------------------------
        # ENCONTROU UMA DUPLICAÇÃO
        # -----------------------------------------------------

        if melhor_correspondencia:

            segmento_r = melhor_correspondencia["segmento"]

            sobreposicao = melhor_correspondencia["sobreposicao"]
            similaridade = melhor_correspondencia["similaridade"]

            segmento_l["sobreposicao"] = sobreposicao
            segmento_l["similaridade"] = similaridade

            segmento_r["sobreposicao"] = sobreposicao
            segmento_r["similaridade"] = similaridade

            # -------------------------------------------------
            # DECISÃO
            #
            # Mantemos L
            # Anulamos R
            # -------------------------------------------------

            segmento_r["duplicado"] = True

    # ---------------------------------------------------------
    # JUNTA OS RESULTADOS
    # ---------------------------------------------------------

    resultado = (
        resultado_esquerda
        +
        resultado_direita
    )

    # Ordena cronologicamente
    resultado.sort(
        key=lambda x: x.get("inicio", 0)
    )

    return resultado


def _remover_duplicatas_multicanal(
    resultados_por_canal,
    limite_sobreposicao=0.70,
    limite_similaridade=0.80
):
    """
    Trata cada canal do MULTICANAL como independente — nenhum canal é
    fixado como "primário"/referência. Compara TODOS os pares de canais
    entre si (canal 0 x 1, 0 x 2, 1 x 2, etc.), usando o mesmo critério
    de sobreposição temporal + similaridade de texto já usado no estéreo.

    Quando dois segmentos de canais DIFERENTES são considerados a mesma
    fala (duplicata), a decisão de qual manter NÃO é pelo índice do canal
    (ex: "canal 0 sempre vence") — é pelo "primeiro falante": mantém-se o
    segmento cujo "inicio" é mais cedo no tempo, e marca-se o outro (que
    começou depois) como duplicado, não importa em qual canal ele esteja.

    Recebe `resultados_por_canal`: lista de dicionários no formato
    {"segmentos": [...]}, um por canal, na ordem dos canais (índice 0..N-1).

    Retorna uma lista única com os segmentos de todos os canais, cada um
    marcado com "canal" (ex: "C0", "C1", ...) e "duplicado" (True/False).
    """
    todos_segmentos = []
    for indice_canal, resultado_canal in enumerate(resultados_por_canal):
        for segmento in resultado_canal["segmentos"]:
            novo_segmento = segmento.copy()
            novo_segmento["canal"] = f"C{indice_canal}"
            novo_segmento["_canal_idx"] = indice_canal  # uso interno, removido no final
            novo_segmento["duplicado"] = False
            novo_segmento["similaridade"] = 0.0
            novo_segmento["sobreposicao"] = 0.0
            todos_segmentos.append(novo_segmento)

    total_segmentos = len(todos_segmentos)

    # ---------------------------------------------------------
    # COMPARAÇÃO PAR A PAR ENTRE TODOS OS CANAIS (não só contra um "primário")
    # ---------------------------------------------------------
    for i in range(total_segmentos):
        segmento_a = todos_segmentos[i]

        # Se este segmento já foi marcado como duplicado por uma
        # comparação anterior, ele não pode mais "vencer" outra —
        # já foi descartado, pula.
        if segmento_a["duplicado"]:
            continue

        for j in range(i + 1, total_segmentos):
            segmento_b = todos_segmentos[j]

            # Só compara canais DIFERENTES — o mesmo canal nunca duplica
            # fala consigo mesmo.
            if segmento_a["_canal_idx"] == segmento_b["_canal_idx"]:
                continue

            if segmento_b["duplicado"]:
                continue

            sobreposicao = _calcular_sobreposicao(
                segmento_a["inicio"], segmento_a["fim"],
                segmento_b["inicio"], segmento_b["fim"]
            )
            if sobreposicao < limite_sobreposicao:
                continue

            similaridade = _similaridade_texto(
                segmento_a.get("texto", ""), segmento_b.get("texto", "")
            )
            if similaridade < limite_similaridade:
                continue

            # ---------------------------------------------
            # DUPLICATA CONFIRMADA — decide pelo "primeiro falante"
            # (quem começou a falar primeiro no tempo), não pelo canal.
            # ---------------------------------------------
            segmento_a["sobreposicao"] = sobreposicao
            segmento_a["similaridade"] = similaridade
            segmento_b["sobreposicao"] = sobreposicao
            segmento_b["similaridade"] = similaridade

            if segmento_a["inicio"] <= segmento_b["inicio"]:
                # A começou primeiro (ou empatou) -> mantém A, descarta B
                segmento_b["duplicado"] = True
            else:
                # B começou primeiro -> mantém B, descarta A
                segmento_a["duplicado"] = True
                break  # A virou duplicado; não faz sentido compará-lo com mais ninguém

    for segmento in todos_segmentos:
        segmento.pop("_canal_idx", None)

    return todos_segmentos


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
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
    return _diarization_pipeline


def _diarizar_audio(caminho_wav):
    pipeline = _carregar_pipeline_diarizacao()
    resultado = pipeline(caminho_wav)

    if hasattr(resultado, "speaker_diarization"):
        diarizacao = resultado.speaker_diarization
    else:
        diarizacao = resultado

    segmentos = []
    for turno, _, falante in diarizacao.itertracks(yield_label=True):
        segmentos.append({
            "inicio": round(turno.start, 2),
            "fim": round(turno.end, 2),
            "falante": falante
        })

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
# PROCESSAMENTO DE UM CANAL (usado no ESTÉREO/MULTICANAL — cada canal
# processado de forma independente)
# ---------------------------------------------------------------------------
def _processar_canal_audio(caminho_audio):
    """
    Roda diarização + transcrição + combinação para UM canal de áudio
    já normalizado (sem cortar silêncio — os timestamps continuam
    correspondendo ao áudio original, o que é necessário para a
    comparação entre canais funcionar corretamente).
    Retorna um dicionário com a lista de segmentos (blocos de fala)
    desse canal, no mesmo formato usado pelo caminho mono.
    """
    segmentos_falantes = _diarizar_audio(caminho_audio)
    palavras_transcricao = _transcrever_com_timestamps(caminho_audio)
    segmentos = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)
    return {"segmentos": segmentos}


def _processar_um_canal_multicanal(caminho_canal, top_db):
    """
    Estágio "processamento" de UM canal do MULTICANAL (roda em paralelo
    para cada C0, C1, ..., CN): normaliza (two-pass), detecta intervalos
    de fala (informativo, não corta) e roda diarização + transcrição.
    Retorna os segmentos desse canal, junto com o caminho do arquivo
    normalizado e os intervalos de fala (só para log/diagnóstico).
    """
    caminho_normalizado = caminho_canal.rsplit('.', 1)[0] + '_normalizado.wav'
    _normalizar_mono_two_pass(caminho_canal, caminho_normalizado)
    intervalos_fala = _detectar_intervalos_fala(caminho_normalizado, top_db)
    resultado_canal = _processar_canal_audio(caminho_normalizado)
    return {
        "caminho_normalizado": caminho_normalizado,
        "intervalos_fala": intervalos_fala,
        "segmentos": resultado_canal["segmentos"]
    }


def _pipeline_multicanal(wav_lossless_path, canais, top_db, base_nomes):
    """
    Pipeline completo do MULTICANAL, em estágios explícitos:

        MULTICANAL
            |
      C0  C1  C2 ... CN        <- separar canais
            |
        processamento          <- por canal, em paralelo (normalizar + VAD + diarizar/transcrever)
            |
   comparação entre canais     <- todos os pares, sem canal fixo como referência
            |
    remover apenas duplicatas  <- mantém quem começou a falar primeiro
            |
       resultado final         <- correção de texto + nomes próprios

    Retorna (resultado, canais_multicanal_paths, canais_normalizados_paths) —
    os dois últimos são devolvidos só para o chamador poder limpar os
    arquivos temporários depois.
    """
    # --- ESTÁGIO 1: SEPARAR CANAIS (C0, C1, C2, ..., CN) ---
    rotulos_canais = ", ".join(f"C{i}" for i in range(canais))
    print(f"[5/10] [MULTICANAL] Separando em {canais} canais: {rotulos_canais}")
    canais_multicanal_paths = _separar_canais_multicanal(wav_lossless_path, canais)
    print(f"[5/10] [MULTICANAL] Canais separados: {canais_multicanal_paths}")

    # --- ESTÁGIO 2: PROCESSAMENTO (por canal, em paralelo) ---
    # Cada canal é tratado de forma totalmente independente: normalização
    # two-pass, VAD informativo (não corta) e diarização+transcrição.
    print(f"[6-8/10] [MULTICANAL] Processando os {canais} canais em paralelo "
          f"(normalização two-pass + VAD informativo + diarização/transcrição)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=canais) as executor:
        futuros = [
            executor.submit(_processar_um_canal_multicanal, caminho, top_db)
            for caminho in canais_multicanal_paths
        ]
        processados = [futuro.result() for futuro in futuros]

    canais_normalizados_paths = [p["caminho_normalizado"] for p in processados]
    resultados_por_canal = [{"segmentos": p["segmentos"]} for p in processados]

    for i, p in enumerate(processados):
        print(f"    C{i}: {len(p['intervalos_fala'])} intervalos de fala, "
              f"{len(p['segmentos'])} segmentos de fala transcritos.")

    # --- ESTÁGIO 3: COMPARAÇÃO ENTRE CANAIS ---
    # Nenhum canal é "primário"/referência fixa — compara todos os pares
    # entre si (C0xC1, C0xC2, C1xC2, ...).
    print(f"[9/10] [MULTICANAL] Comparando todos os canais entre si "
          f"(sem canal fixo como referência)...")
    segmentos_comparados = _remover_duplicatas_multicanal(
        resultados_por_canal, limite_sobreposicao=0.70, limite_similaridade=0.80
    )

    # --- ESTÁGIO 4: REMOVER APENAS DUPLICATAS ---
    # Mantém sempre o segmento cujo falante começou a falar primeiro no
    # tempo, independente do índice do canal.
    resultado_combinado = [
        s for s in segmentos_comparados if not s.get("duplicado", False)
    ]
    total_duplicadas = len(segmentos_comparados) - len(resultado_combinado)
    print(f"[9/10] [MULTICANAL] Duplicatas removidas: {total_duplicadas} "
          f"(mantendo sempre quem começou a falar primeiro)")
    resultado_combinado.sort(key=lambda s: (s.get("inicio", 0), s.get("fim", 0)))

    # --- ESTÁGIO 5: RESULTADO FINAL (correção de texto + nomes próprios) ---
    print(f"[10/10] [MULTICANAL] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
    resultado = _corrigir_blocos(resultado_combinado)

    print(f"[10/10] [MULTICANAL] Consultando nomes próprios conhecidos (IBGE)...")
    if base_nomes is None:
        base_nomes = _obter_ranking_nomes_ibge(quantidade=100)
    for bloco in resultado:
        bloco["texto_corrigido"] = _consultar_nomes_proprios(bloco["texto_corrigido"], base_nomes)

    print("\n--- RESULTADO FINAL (MULTICANAL) ---")
    for item in resultado:
        print(f"[{item['inicio']}s - {item['fim']}s] canal={item.get('canal')} "
              f"{item['falante']}: {item['texto_corrigido']}")

    return resultado, canais_multicanal_paths, canais_normalizados_paths


# ---------------------------------------------------------------------------
# CORREÇÃO DE TEXTO (LLM via Groq) — pontuação, capitalização, termos técnicos
# ---------------------------------------------------------------------------
def _corrigir_texto(texto):
    prompt = (
        "Você é um corretor de texto. Corrija SOMENTE pontuação, capitalização e "
        "ajuste termos técnicos que possam estar errados por transcrição automática. "
        "NÃO adicione, remova ou reescreva o conteúdo. NÃO resuma. "
        "Responda apenas com o texto corrigido, sem comentários.\n\n"
        f"Texto original: {texto}"
    )

    resposta = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip()


def _corrigir_blocos(blocos):
    for bloco in blocos:
        bloco["texto_corrigido"] = _corrigir_texto(bloco["texto"])
    return blocos


# ---------------------------------------------------------------------------
# RANKING DE NOMES MAIS COMUNS NO BRASIL (via DadosAbertosBrasil / IBGE)
# ---------------------------------------------------------------------------
DECADA_MAIS_RECENTE = 2010
DECADA_MAIS_ANTIGA = 1930


def _obter_ranking_nomes_ibge(quantidade=100, forcar_atualizacao=False):
    if not forcar_atualizacao and os.path.exists(CACHE_NOMES_PATH):
        with open(CACHE_NOMES_PATH, 'r', encoding='utf-8') as f:
            nomes_em_cache = json.load(f)
        if len(nomes_em_cache) >= quantidade:
            return nomes_em_cache[:quantidade]

    print(f"    Buscando ranking de nomes via DadosAbertosBrasil (IBGE), combinando décadas e sexos...")

    nomes = []
    nomes_vistos = set()
    decada = DECADA_MAIS_RECENTE

    while len(nomes) < quantidade and decada >= DECADA_MAIS_ANTIGA:
        for sexo in ('f', 'm'):
            try:
                df = dab_ibge.nomes_ranking(decada=decada, sexo=sexo, formato='pandas')
                for nome_bruto in df['nome']:
                    nome_formatado = str(nome_bruto).title()
                    if nome_formatado not in nomes_vistos:
                        nomes_vistos.add(nome_formatado)
                        nomes.append(nome_formatado)
            except Exception as e:
                print(f"    Aviso: falha ao buscar ranking da década {decada} (sexo={sexo}): {e}")
        print(f"    Década {decada}: {len(nomes)} nomes únicos acumulados")
        decada -= 10

    nomes = nomes[:quantidade]

    with open(CACHE_NOMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(nomes, f, ensure_ascii=False, indent=2)

    print(f"    {len(nomes)} nomes obtidos e salvos em cache: {CACHE_NOMES_PATH}")
    return nomes


def _consultar_nomes_proprios(texto, base_nomes, limiar_similaridade=0.75):
    if not base_nomes:
        return texto

    palavras = texto.split()
    palavras_corrigidas = []

    for palavra in palavras:
        pontuacao_final = ""
        palavra_limpa = palavra
        while palavra_limpa and palavra_limpa[-1] in '.,!?;:':
            pontuacao_final = palavra_limpa[-1] + pontuacao_final
            palavra_limpa = palavra_limpa[:-1]

        if palavra_limpa and palavra_limpa[0].isupper() and len(palavra_limpa) > 2:
            candidatos = difflib.get_close_matches(
                palavra_limpa, base_nomes, n=1, cutoff=limiar_similaridade
            )
            if candidatos:
                candidato = candidatos[0]

                def remover_acentos(s):
                    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

                if candidato != palavra_limpa and remover_acentos(candidato).lower() != remover_acentos(palavra_limpa).lower():
                    print(f"    Nome corrigido: '{palavra_limpa}' -> '{candidato}'")
                    palavra_limpa = candidato

        palavras_corrigidas.append(palavra_limpa + pontuacao_final)

    return " ".join(palavras_corrigidas)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def transcrever_arquivo(caminho_original, base_nomes=None, top_db=40):
    """Recebe um caminho de arquivo já existente em disco e retorna a transcrição com falantes."""
    extensao = os.path.splitext(caminho_original)[1].lower()
    print(f"[1/10] Copiando arquivo temporário...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp:
        shutil.copyfile(caminho_original, tmp.name)
        tmp_path = tmp.name

    audio_path = tmp_path
    wav_lossless_path = None
    normalizado_path = None
    sem_silencio_path = None

    # --- Temporários específicos do processamento estéreo ---
    canal_esq_path = None
    canal_dir_path = None
    normalizado_esq_path = None
    normalizado_dir_path = None

    # --- Temporários específicos do processamento multicanal (N canais) ---
    canais_multicanal_paths = None
    canais_normalizados_paths = None

    try:
        print(f"[2/10] Verificando duração do arquivo recebido...")
        duracao_bruta = librosa.get_duration(path=tmp_path)
        print(f"[2/10] Duração: {int(duracao_bruta)}s")

        if duracao_bruta > LIMITE_SEGUNDOS:
            raise ValueError(f"Áudio muito longo: {int(duracao_bruta)}s. Limite: {LIMITE_SEGUNDOS}s (1 hora).")

        if extensao in ('.mp4', '.mp3'):
            # --- Detecção de canais (função própria) ---
            print(f"[3/10] Verificando canais de áudio...")
            info_canais = _detectar_canais(tmp_path)
            canais = info_canais["canais"]
            tipo_canal = info_canais["tipo_canal"]  # "mono" | "estereo" | "multicanal"
            print(f"[3/10] Áudio detectado: {info_canais['tipo_audio']} "
                  f"(layout: {info_canais['layout']}, taxa: {info_canais['taxa_original']} Hz)")

            # --- Conversão lossless para WAV (igual para os 3 tipos) ---
            print(f"[4/10] Convertendo {extensao} para WAV sem perda de dados (PCM 16-bit)...")
            wav_lossless_path = tmp_path.rsplit('.', 1)[0] + '_lossless.wav'
            _converter_para_wav_lossless(tmp_path, wav_lossless_path, canais)
            print(f"[4/10] WAV lossless gerado: {wav_lossless_path}")

            # =========================================================
            # ESTRATÉGIA POR TIPO DE CANAL (a partir daqui os caminhos divergem)
            # =========================================================
            if tipo_canal == "estereo":
                # --- ESTÉREO: separa as duas vozes (canal esquerdo/direito) ---
                print(f"[5/10] Separando canais estéreo em duas vozes (esquerda/direita)...")
                canal_esq_path = tmp_path.rsplit('.', 1)[0] + '_esq.wav'
                canal_dir_path = tmp_path.rsplit('.', 1)[0] + '_dir.wav'
                _separar_canais_estereo(wav_lossless_path, canal_esq_path, canal_dir_path)
                print(f"[5/10] Canais separados: {canal_esq_path} | {canal_dir_path}")

                # --- Normalização EBU R128 dos dois canais EM PARALELO (multitarefa) ---
                print(f"[6/10] Normalizando os dois canais em paralelo (EBU R128 single-pass, multitarefa)...")
                normalizado_esq_path = tmp_path.rsplit('.', 1)[0] + '_esq_normalizado.wav'
                normalizado_dir_path = tmp_path.rsplit('.', 1)[0] + '_dir_normalizado.wav'

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futuro_esq = executor.submit(
                        _normalizar_mono_single_pass, canal_esq_path, normalizado_esq_path
                    )
                    futuro_dir = executor.submit(
                        _normalizar_mono_single_pass, canal_dir_path, normalizado_dir_path
                    )
                    futuro_esq.result()
                    futuro_dir.result()

                print(f"[6/10] Normalização concluída em paralelo para os dois canais.")
                print(f"    Canal esquerdo normalizado: {normalizado_esq_path}")
                print(f"    Canal direito normalizado: {normalizado_dir_path}")

                # --- VAD informativo (NÃO corta o áudio) dos dois canais, em paralelo ---
                # Só serve para diagnóstico (quantos trechos de fala existem);
                # a diarização/transcrição roda sobre o áudio normalizado
                # completo (sem cortes), preservando os timestamps originais.
                print(f"[7/10] Detectando intervalos de fala nos dois canais em paralelo "
                      f"(VAD informativo, NÃO corta o áudio, top_db={top_db})...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futuro_vad_esq = executor.submit(_detectar_intervalos_fala, normalizado_esq_path, top_db)
                    futuro_vad_dir = executor.submit(_detectar_intervalos_fala, normalizado_dir_path, top_db)
                    intervalos_esq = futuro_vad_esq.result()
                    intervalos_dir = futuro_vad_dir.result()

                print(f"[7/10] VAD concluído: canal esquerdo com {len(intervalos_esq)} intervalos de fala, "
                      f"canal direito com {len(intervalos_dir)} intervalos de fala.")

                # --- Diarização + transcrição de cada canal, EM PARALELO ---
                print(f"[8/10] Diarizando e transcrevendo os dois canais em paralelo...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futuro_esq = executor.submit(_processar_canal_audio, normalizado_esq_path)
                    futuro_dir = executor.submit(_processar_canal_audio, normalizado_dir_path)
                    resultado_esq = futuro_esq.result()
                    resultado_dir = futuro_dir.result()

                segmentos_esq = resultado_esq["segmentos"]
                segmentos_dir = resultado_dir["segmentos"]
                print(f"[8/10] Segmentos encontrados — esquerdo: {len(segmentos_esq)}, direito: {len(segmentos_dir)}")

                # --- Compara L x R para eliminar duplicação (mesma fala captada nos dois canais) ---
                print(f"[9/10] Comparando canais para remover falas duplicadas...")
                segmentos_comparados = _comparar_falas_estereo(
                    segmentos_esq, segmentos_dir,
                    limite_sobreposicao=0.70, limite_similaridade=0.80
                )

                segmentos_sem_duplicacao = [
                    s for s in segmentos_comparados if not s.get("duplicado", False)
                ]
                quantidade_duplicadas = len(segmentos_comparados) - len(segmentos_sem_duplicacao)
                print(f"[9/10] Duplicações removidas: {quantidade_duplicadas}")

                segmentos_sem_duplicacao.sort(key=lambda s: (s.get("inicio", 0), s.get("fim", 0)))

                # --- Correção de texto (mesma etapa do caminho mono/multicanal) ---
                print(f"[10/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
                resultado = _corrigir_blocos(segmentos_sem_duplicacao)

                print(f"[10/10] Consultando nomes próprios conhecidos (IBGE)...")
                if base_nomes is None:
                    base_nomes = _obter_ranking_nomes_ibge(quantidade=100)
                for bloco in resultado:
                    bloco["texto_corrigido"] = _consultar_nomes_proprios(bloco["texto_corrigido"], base_nomes)

                print("\n--- RESULTADO FINAL (ESTÉREO) ---")
                for item in resultado:
                    print(f"[{item['inicio']}s - {item['fim']}s] canal={item.get('canal')} "
                          f"{item['falante']}: {item['texto_corrigido']}")

                return resultado

            elif tipo_canal == "multicanal":
                # --- MULTICANAL: pipeline dedicado, em estágios explícitos
                # (separar canais -> processamento -> comparação entre canais
                # -> remover apenas duplicatas -> resultado final). Ver
                # _pipeline_multicanal para o detalhe de cada estágio. ---
                resultado, canais_multicanal_paths, canais_normalizados_paths = _pipeline_multicanal(
                    wav_lossless_path, canais, top_db, base_nomes
                )
                return resultado

            else:
                # --- MONO: fluxo original, sem separação de canais ---
                normalizado_path = tmp_path.rsplit('.', 1)[0] + '_normalizado.wav'

                print(f"[5/10] Normalizando loudness EBU R128 (single-pass)...")
                _normalizar_mono_single_pass(wav_lossless_path, normalizado_path)

                print(f"[5/10] WAV normalizado gerado: {normalizado_path}")

                # --- Enquadramento + remoção de silêncio (sobre o áudio já normalizado) ---
                print(f"[6/10] Enquadrando e removendo trechos de silêncio (VAD por energia, 25ms/10ms, top_db={top_db})...")
                sem_silencio_path = tmp_path.rsplit('.', 1)[0] + '_final.wav'
                _enquadrar_e_remover_silencio(normalizado_path, sem_silencio_path, top_db=top_db)
                print(f"[6/10] Concluído: {sem_silencio_path}")

                audio_path = sem_silencio_path

                print(f"[7/10] Identificando falantes (diarização)...")
                segmentos_falantes = _diarizar_audio(audio_path)
                print(f"[7/10] {len(segmentos_falantes)} segmentos de fala identificados")

                print(f"[8/10] Transcrevendo com timestamps por palavra (Groq)...")
                palavras_transcricao = _transcrever_com_timestamps(audio_path)

                resultado = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)

                print(f"[9/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
                resultado = _corrigir_blocos(resultado)

                print(f"[10/10] Consultando nomes próprios conhecidos (IBGE)...")
                if base_nomes is None:
                    base_nomes = _obter_ranking_nomes_ibge(quantidade=100)
                for bloco in resultado:
                    bloco["texto_corrigido"] = _consultar_nomes_proprios(bloco["texto_corrigido"], base_nomes)

                print("\n--- RESULTADO FINAL ---")
                for item in resultado:
                    print(f"[{item['inicio']}s - {item['fim']}s] {item['falante']}: {item['texto_corrigido']}")

                return resultado

        # Caso o arquivo não seja .mp4/.mp3 (ex: .wav puro) — fluxo mínimo direto
        print(f"[7/10] Identificando falantes (diarização)...")
        segmentos_falantes = _diarizar_audio(audio_path)
        print(f"[7/10] {len(segmentos_falantes)} segmentos de fala identificados")

        print(f"[8/10] Transcrevendo com timestamps por palavra (Groq)...")
        palavras_transcricao = _transcrever_com_timestamps(audio_path)

        resultado = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)

        print(f"[9/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
        resultado = _corrigir_blocos(resultado)

        print(f"[10/10] Consultando nomes próprios conhecidos (IBGE)...")
        if base_nomes is None:
            base_nomes = _obter_ranking_nomes_ibge(quantidade=100)
        for bloco in resultado:
            bloco["texto_corrigido"] = _consultar_nomes_proprios(bloco["texto_corrigido"], base_nomes)

        print("\n--- RESULTADO FINAL ---")
        for item in resultado:
            print(f"[{item['inicio']}s - {item['fim']}s] {item['falante']}: {item['texto_corrigido']}")

        return resultado

    finally:
        os.remove(tmp_path)
        if wav_lossless_path and os.path.exists(wav_lossless_path):
            os.remove(wav_lossless_path)
        if normalizado_path and os.path.exists(normalizado_path):
            os.remove(normalizado_path)
        if sem_silencio_path and os.path.exists(sem_silencio_path):
            os.remove(sem_silencio_path)
        # Intermediários da separação estéreo — sempre removidos
        if canal_esq_path and os.path.exists(canal_esq_path):
            os.remove(canal_esq_path)
        if canal_dir_path and os.path.exists(canal_dir_path):
            os.remove(canal_dir_path)
        if normalizado_esq_path and os.path.exists(normalizado_esq_path):
            os.remove(normalizado_esq_path)
        if normalizado_dir_path and os.path.exists(normalizado_dir_path):
            os.remove(normalizado_dir_path)
        # Intermediários da separação multicanal (N canais) — sempre removidos
        if canais_multicanal_paths:
            for p in canais_multicanal_paths:
                if p and os.path.exists(p):
                    os.remove(p)
        if canais_normalizados_paths:
            for p in canais_normalizados_paths:
                if p and os.path.exists(p):
                    os.remove(p)


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


# ---------------------------------------------------------------------------
# TESTE ISOLADO — só a consulta de nomes (IBGE), sem rodar o pipeline de áudio
# ---------------------------------------------------------------------------
def _testar_consulta_nomes():
    print("=== TESTE: Consulta de nomes IBGE ===\n")

    base_nomes = _obter_ranking_nomes_ibge(quantidade=100, forcar_atualizacao=True)
    print(f"\nExemplos da base carregada: {base_nomes[:10]}\n")
    print(f"Total de nomes na base: {len(base_nomes)}\n")

    textos_teste = [
        "Bom dia, meu nome é Marcus e trabalho com o João.",
        "A Anna disse que vai encontrar o Fabio amanhã.",
        "Oi Ricardo, tudo bem? Aqui é a Juliia falando.",
    ]

    for texto in textos_teste:
        print(f"Original:  {texto}")
        corrigido = _consultar_nomes_proprios(texto, base_nomes)
        print(f"Corrigido: {corrigido}\n")


if __name__ == '__main__':
    TESTE_LOCAL = True
    TESTE_NOMES_IBGE = False

    if TESTE_NOMES_IBGE:
        _testar_consulta_nomes()
    elif TESTE_LOCAL:
        diretorio_script = os.path.dirname(os.path.abspath(__file__))
        pasta_testes = os.path.join(os.path.dirname(diretorio_script), "Teste Video/Teste_canal")
        arquivos_teste = ["teste_multicanal.mp4"]

        print(f"\n[DIAGNÓSTICO] Diretório do script: {diretorio_script}")
        print(f"[DIAGNÓSTICO] Pasta de testes esperada: {pasta_testes}")
        print(f"[DIAGNÓSTICO] Pasta existe? {os.path.isdir(pasta_testes)}")
        if os.path.isdir(pasta_testes):
            print(f"[DIAGNÓSTICO] Conteúdo da pasta: {os.listdir(pasta_testes)}")
        else:
            print(f"[DIAGNÓSTICO] Conteúdo do diretório do script: {os.listdir(diretorio_script)}")

        base_nomes = _obter_ranking_nomes_ibge(quantidade=100)

        resultados = {}
        for nome_arquivo in arquivos_teste:
            caminho = os.path.join(pasta_testes, nome_arquivo)
            print(f"\n{'=' * 70}")
            print(f"  RODANDO PIPELINE: {nome_arquivo}")
            print(f"{'=' * 70}")
            try:
                resultados[nome_arquivo] = transcrever_arquivo(caminho, base_nomes=base_nomes)
            except Exception as e:
                print(f"  ERRO ao processar {nome_arquivo}: {e}")
                resultados[nome_arquivo] = None

        print(f"\n{'=' * 70}")
        print("  RESUMO DOS TESTES")
        print(f"{'=' * 70}")
        for nome_arquivo, resultado in resultados.items():
            status = f"OK ({len(resultado)} blocos)" if resultado is not None else "FALHOU"
            print(f"  {nome_arquivo}: {status}")
    else:
        app.run(debug=True)