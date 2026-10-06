import sys

print("Importando Flask...", flush=True)
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Importando ffmpeg...", flush=True)
import ffmpeg

print("Importando difflib/re/unicodedata...", flush=True)
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
import subprocess
import hashlib
import httpx
import threading
import bisect
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

print("Importando torch (necessário para detectar/usar GPU na diarização)...", flush=True)
import torch

print("Importando pyannote.audio (ESTA É A MAIS LENTA — pode levar de 1 a 3+ minutos)...", flush=True)
print("  -> carregando torch/lightning/torchmetrics por baixo dos panos...", flush=True)
from pyannote.audio import Pipeline as PyannotePipeline
print("pyannote.audio carregado com sucesso!", flush=True)

print("Todos os imports concluídos. Iniciando servidor...\n", flush=True)

app = Flask(__name__)
CORS(app)

# O SDK da Groq usa timeout padrão de só 60s por requisição — curto demais
# mesmo para um único chunk de 10min (upload + processamento pode passar
# disso). Aumentamos para 600s (o SDK ainda faz até 2 retries automáticos
# por cima disso em caso de erro de conexão/timeout).
client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=600.0)

LIMITE_SEGUNDOS = 3600
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES AJUSTÁVEIS VIA .env (sem mexer no código)
# ---------------------------------------------------------------------------
# [NOVO] Modelo de transcrição. Teste "whisper-large-v3" (mais preciso em
# português, porém mais lento/caro) no lugar do turbo.
MODELO_TRANSCRICAO = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
# [NOVO] Idioma fixo evita detecção automática errada e ajuda a reduzir
# alucinações do tipo "thank you" / "legendas pela comunidade".
IDIOMA_TRANSCRICAO = os.getenv("IDIOMA_TRANSCRICAO", "pt")
# [NOVO] Modelo de diarização. Com pyannote 4.x, teste
# "pyannote/speaker-diarization-community-1".
MODELO_DIARIZACAO = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
# [NOVO] Reparo de fronteiras de turno por LLM (1 chamada por troca de falante).
REPARAR_FRONTEIRAS_LLM = os.getenv("REPARAR_FRONTEIRAS_LLM", "true").strip().lower() in ("1", "true", "sim", "yes")
# [NOVO] No fluxo MONO, remove os trechos de silêncio antes de diarizar/transcrever
# (os timestamps finais são SEMPRE devolvidos na linha do tempo do áudio original,
# graças ao mapa de tempo). Coloque "false" para não cortar nada.
REMOVER_SILENCIO_MONO = os.getenv("REMOVER_SILENCIO_MONO", "true").strip().lower() in ("1", "true", "sim", "yes")

# ---------------------------------------------------------------------------
# [NOVO] PERFIL DE ÁUDIO: "conversa" (padrão, entrevistas) ou "musica" (trap/rap)
# ---------------------------------------------------------------------------
PERFIL_AUDIO = os.getenv("PERFIL_AUDIO", "conversa").strip().lower()
EH_MUSICA = PERFIL_AUDIO == "musica"

# Em música estéreo, os dois canais carregam a MESMA voz (mix), então
# por padrão forçamos mono para não duplicar processamento.
FORCAR_MONO = EH_MUSICA or os.getenv("FORCAR_MONO", "false").strip().lower() in ("1", "true", "sim", "yes")

# Opcional (pesado): isola o stem de vocais com Demucs antes do Whisper.
ISOLAR_VOCAIS_DEMUCS = os.getenv("ISOLAR_VOCAIS_DEMUCS", "false").strip().lower() in ("1", "true", "sim", "yes")

if EH_MUSICA:
    # VAD por energia não funciona com beat/instrumental: tudo parece "fala".
    REMOVER_SILENCIO_MONO = False

# ---------------------------------------------------------------------------
# [NOVO] RECONHECIMENTO DE MÚSICA NA NUVEM (AudD) — usado no perfil "musica"
# ---------------------------------------------------------------------------
# AUDD_API_TOKEN: token da API do AudD (https://audd.io). Sem token, a etapa é pulada.
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "").strip()
# Liga/desliga o reconhecimento (padrão: ligado só no perfil música).
RECONHECER_MUSICA = os.getenv("RECONHECER_MUSICA", "true" if EH_MUSICA else "false").strip().lower() in ("1", "true", "sim", "yes")
# Busca a letra via API e usa como REFERÊNCIA para corrigir as palavras do ASR
# (os timestamps continuam sendo os do Whisper).
USAR_LETRA_OFICIAL = os.getenv("USAR_LETRA_OFICIAL", "true").strip().lower() in ("1", "true", "sim", "yes")
# Só aplica a letra se ela for parecida o bastante com o que o ASR ouviu
# (protege contra identificação errada).
LIMITE_SIMILARIDADE_LETRA = float(os.getenv("LIMITE_SIMILARIDADE_LETRA", "0.35"))

_diarization_pipeline = None
# Protege tanto o carregamento (lazy singleton) quanto as chamadas de
# inferência do pipeline de diarização. Sem isso, dois canais rodando em
# paralelo (ThreadPoolExecutor) podem: (a) cair juntos no "if is None" e
# CADA UM carregar sua própria cópia do modelo na GPU — visível nos logs
# como "Carregando modelo..." e "Rodando em: cuda" duplicados, arriscado
# numa GPU com pouca VRAM — e (b) mesmo com uma única cópia, rodar duas
# inferências ao mesmo tempo na MESMA instância do pipeline, o que o
# pyannote não garante ser thread-safe. Serializar apenas esta etapa é
# barato: a diarização é local e rápida perto do tempo gasto em rede
# esperando a Groq, então o resto do pipeline (normalização, VAD,
# transcrição) continua paralelo normalmente.
_diarization_lock = threading.Lock()


# ---------------------------------------------------------------------------
# ACESSO SEGURO A CAMPOS (dict OU objeto do SDK) — usado pelas heurísticas de
# alucinação, que leem campos opcionais (no_speech_prob, avg_logprob, etc.)
# que podem não existir dependendo da versão da resposta do Groq.
# ---------------------------------------------------------------------------
def _campo(obj, chave, padrao=None):
    if obj is None:
        return padrao
    if isinstance(obj, dict):
        return obj.get(chave, padrao)
    return getattr(obj, chave, padrao)


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
# CONVERSÃO PARA WAV PCM 16-BIT
# ---------------------------------------------------------------------------
def _converter_para_wav_lossless(caminho_entrada, caminho_saida, canais):
    """
    Converte o áudio para WAV PCM 16-bit.

    Registra as características do áudio original e as características
    do áudio convertido.
    """

    # ---------------------------------------------------------------
    # 1. Descobrir informações do áudio original
    # ---------------------------------------------------------------
    probe = ffmpeg.probe(caminho_entrada)

    stream_audio = next(
        (
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None
    )

    if stream_audio is None:
        raise ValueError("Nenhuma trilha de áudio encontrada.")

    codec_original = stream_audio.get("codec_name", "desconhecido")
    sample_rate_original = stream_audio.get("sample_rate", "desconhecido")
    canais_original = stream_audio.get("channels", "desconhecido")
    bits_original = stream_audio.get("bits_per_sample", 0)

    # AAC/MP3/etc. normalmente não possuem bits_per_sample PCM.
    if bits_original in (None, 0, "0"):
        bits_original_texto = "não aplicável (áudio comprimido)"
    else:
        bits_original_texto = f"{bits_original}-bit"

    # ---------------------------------------------------------------
    # 2. Conversão
    # ---------------------------------------------------------------
    (
        ffmpeg
        .input(caminho_entrada)
        .output(
            caminho_saida,
            vn=None,
            acodec="pcm_s16le",
            ac=canais
        )
        .run(
            quiet=True,
            overwrite_output=True
        )
    )

    # ---------------------------------------------------------------
    # 3. Informações do WAV convertido
    # ---------------------------------------------------------------
    probe_convertido = ffmpeg.probe(caminho_saida)

    stream_convertido = next(
        (
            stream
            for stream in probe_convertido.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None
    )

    if stream_convertido is None:
        raise ValueError("Não foi possível verificar o WAV convertido.")

    sample_rate_convertido = stream_convertido.get(
        "sample_rate",
        "desconhecido"
    )

    canais_convertido = stream_convertido.get(
        "channels",
        "desconhecido"
    )

    bits_convertido = stream_convertido.get(
        "bits_per_sample",
        16
    )

    print(
        "\n--- CONVERSÃO DE ÁUDIO ---",
        flush=True
    )

    print(
        f"Formato original: {codec_original.upper()}",
        flush=True
    )

    print(
        f"Profundidade original: {bits_original_texto}",
        flush=True
    )

    print(
        f"Taxa de amostragem original: "
        f"{sample_rate_original} Hz",
        flush=True
    )

    print(
        f"Canais originais: {canais_original}",
        flush=True
    )

    print(
        f"Formato convertido: PCM {bits_convertido}-bit",
        flush=True
    )

    print(
        f"Taxa de amostragem convertida: "
        f"{sample_rate_convertido} Hz",
        flush=True
    )

    print(
        f"Canais convertidos: {canais_convertido}",
        flush=True
    )

    print(
        "---------------------------\n",
        flush=True
    )

    return {
        "formato_original": codec_original,
        "profundidade_original": bits_original_texto,
        "taxa_amostragem_original_hz": sample_rate_original,
        "canais_originais": canais_original,
        "formato_convertido": f"PCM {bits_convertido}-bit",
        "taxa_amostragem_convertida_hz": sample_rate_convertido,
        "canais_convertidos": canais_convertido,
    }


# ---------------------------------------------------------------------------
# [NOVO] ISOLAMENTO DE VOCAIS (Demucs) — opcional, pesado. Útil em música.
# Requer: pip install demucs
# ---------------------------------------------------------------------------
def _isolar_vocais_demucs(caminho_wav, pasta_saida):
    """
    Roda o Demucs e devolve o caminho do vocals.wav, ou None se falhar
    (nesse caso o erro REAL do Demucs é impresso e o pipeline segue com o
    áudio original, em vez de abortar tudo).
    """
    resultado = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
         "-o", pasta_saida, caminho_wav],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if resultado.returncode != 0:
        print(f"    [DEMUCS] FALHOU (código {resultado.returncode}). Saída de erro:\n"
              f"{(resultado.stderr or resultado.stdout or '(sem saída)').strip()[-3000:]}",
              flush=True)
        return None

    nome = os.path.splitext(os.path.basename(caminho_wav))[0]
    caminho_vocais = os.path.join(pasta_saida, "htdemucs", nome, "vocals.wav")
    if not os.path.exists(caminho_vocais):
        print(f"    [DEMUCS] Terminou sem erro, mas não achei {caminho_vocais}.", flush=True)
        return None
    return caminho_vocais


# ---------------------------------------------------------------------------
# [NOVO] RECONHECIMENTO DE MÚSICA NA NUVEM (AudD) + LETRA + CACHE LOCAL
# ---------------------------------------------------------------------------
# Fluxo: recorta ~20s do áudio ORIGINAL (a mixagem completa, antes do Demucs,
# porque fingerprint funciona melhor com o instrumental junto) -> AudD
# identifica título/artista -> busca a letra -> tudo é guardado em
# cache_musicas.json (chave = SHA-1 do arquivo), então a mesma faixa não
# volta a consultar a nuvem.
AUDD_URL = "https://api.audd.io/"
AUDD_URL_LETRA = "https://api.audd.io/findLyrics/"
CAMINHO_CACHE_MUSICAS = os.path.join(BASE_DIR, "cache_musicas.json")
_cache_musicas_lock = threading.Lock()


def _hash_arquivo(caminho):
    h = hashlib.sha1()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def _ler_cache_musicas():
    try:
        with open(CAMINHO_CACHE_MUSICAS, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _salvar_no_cache_musicas(chave, info):
    with _cache_musicas_lock:
        cache = _ler_cache_musicas()
        cache[chave] = info
        with open(CAMINHO_CACHE_MUSICAS, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


def _extrair_trecho_para_reconhecimento(wav_path, saida, duracao_total, duracao_trecho=15):
    # Trecho de ~15s a partir dos 15s (onde beat e voz já entraram juntos);
    # em áudios curtos, começa em 1/3 da duração.
    inicio = min(15.0, max(0.0, duracao_total / 3))
    duracao = min(float(duracao_trecho), float(duracao_total))
    (
        ffmpeg
        .input(wav_path, ss=inicio, t=duracao)
        .output(saida, acodec='pcm_s16le', ac=1, ar=44100)
        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    )


def _reconhecer_musica_audd(caminho_trecho):
    with open(caminho_trecho, "rb") as f:
        resp = httpx.post(
            AUDD_URL,
            data={"api_token": AUDD_API_TOKEN, "return": "lyrics,spotify"},
            files={"file": (os.path.basename(caminho_trecho), f.read())},
            timeout=30.0,
        )
    dados = resp.json()
    if not isinstance(dados, dict) or dados.get("status") != "success" or not dados.get("result"):
        return None
    resultado = dados["result"]

    # A letra pode vir como {"lyrics": "..."} ou direto como texto, dependendo da resposta.
    letra_bruta = resultado.get("lyrics")
    letra = letra_bruta.get("lyrics") if isinstance(letra_bruta, dict) else letra_bruta

    return {
        "titulo": resultado.get("title"),
        "artista": resultado.get("artist"),
        "album": resultado.get("album"),
        "lancamento": resultado.get("release_date"),
        "link": resultado.get("song_link"),
        "letra": letra if isinstance(letra, str) and letra.strip() else None,
    }


def _buscar_letra_audd(artista, titulo):
    resp = httpx.get(
        AUDD_URL_LETRA,
        params={"q": f"{artista} {titulo}", "api_token": AUDD_API_TOKEN},
        timeout=30.0,
    )
    dados = resp.json()
    candidatos = (dados.get("result") if isinstance(dados, dict) else None) or []
    alvo_t = _normalizar_texto_comparacao(titulo or "")
    alvo_a = _normalizar_texto_comparacao(artista or "")
    melhor, melhor_score = None, 0.0
    for c in candidatos:
        letra = c.get("lyrics")
        if not letra:
            continue
        s_t = SequenceMatcher(None, alvo_t, _normalizar_texto_comparacao(c.get("title", ""))).ratio()
        s_a = SequenceMatcher(None, alvo_a, _normalizar_texto_comparacao(c.get("artist", ""))).ratio()
        score = (s_t + s_a) / 2
        if score > melhor_score:
            melhor, melhor_score = letra, score
    return melhor if melhor_score >= 0.6 else None


def _identificar_musica(caminho_arquivo, wav_path, duracao_total):
    """
    Devolve um dict {titulo, artista, album, lancamento, link, letra} ou
    None (não identificada / sem token / erro de rede). Nunca derruba o
    pipeline: qualquer falha vira log e o fluxo segue com ASR puro.
    """
    chave = _hash_arquivo(caminho_arquivo)
    with _cache_musicas_lock:
        em_cache = _ler_cache_musicas().get(chave)
    if em_cache:
        print(f"    [MÚSICA] Já identificada antes (cache): "
              f"{em_cache.get('artista')} - {em_cache.get('titulo')}", flush=True)
        return em_cache

    if not AUDD_API_TOKEN:
        print("    [MÚSICA] AUDD_API_TOKEN não definido no .env — pulando reconhecimento.", flush=True)
        return None

    trecho = wav_path.rsplit('.', 1)[0] + '_trecho_id.wav'
    try:
        _extrair_trecho_para_reconhecimento(wav_path, trecho, duracao_total)
        info = _reconhecer_musica_audd(trecho)
        if not info:
            print("    [MÚSICA] Faixa não identificada na nuvem — seguindo com a transcrição (ASR).", flush=True)
            return None
        print(f"    [MÚSICA] Identificada: {info.get('artista')} - {info.get('titulo')}", flush=True)

        if not USAR_LETRA_OFICIAL:
            info["letra"] = None
        elif info.get("letra"):
            print("    [MÚSICA] Letra veio junto com o reconhecimento.", flush=True)
        else:
            # Reconhecimento não trouxe a letra: tenta a busca separada por artista + título
            try:
                info["letra"] = _buscar_letra_audd(info.get("artista"), info.get("titulo"))
                print(f"    [MÚSICA] Letra {'encontrada (busca separada)' if info['letra'] else 'NÃO encontrada'}.", flush=True)
            except Exception as e:
                print(f"    [MÚSICA] Falha ao buscar a letra: {e}", flush=True)

        _salvar_no_cache_musicas(chave, info)
        return info
    except Exception as e:
        print(f"    [MÚSICA] Falha no reconhecimento: {e}", flush=True)
        return None
    finally:
        if os.path.exists(trecho):
            os.remove(trecho)


def _resumo_musica(info):
    """Versão da info da música para devolver na API (sem o texto da letra)."""
    if not info:
        return None
    resumo = {k: info.get(k) for k in ("titulo", "artista", "album", "lancamento", "link")}
    resumo["letra_encontrada"] = bool(info.get("letra"))
    return resumo


def _preparar_tokens_letra(letra):
    """Letra -> lista de (texto_exibicao, token_normalizado). Ignora linhas
    de marcação como [Refrão] e palavras que viram vazio após normalizar."""
    linhas = [l for l in letra.splitlines() if not re.match(r"^\s*\[.*\]\s*$", l)]
    tokens = []
    for palavra in " ".join(linhas).split():
        partes = _normalizar_texto_comparacao(palavra).split()
        if len(partes) == 1:
            tokens.append((palavra, partes[0]))
        else:
            tokens.extend((p, p) for p in partes)
    return tokens


def _alinhar_palavras_com_letra(palavras, letra, limite=LIMITE_SIMILARIDADE_LETRA):
    """
    Usa a letra como REFERÊNCIA de texto, mantendo os timestamps do ASR:
    alinha as palavras do Whisper com as da letra (difflib) e, onde elas
    batem ou há uma troca 1-para-1, usa a grafia da letra. Trechos que não
    casam (ad-libs, trocas de tamanho diferente) ficam como o ASR ouviu.
    Se a letra for parecida demais com pouco do que foi ouvido (similaridade
    < limite), não aplica nada — provável identificação errada.
    """
    tokens_letra = _preparar_tokens_letra(letra or "")
    if not palavras or not tokens_letra:
        return palavras

    asr = [(_normalizar_texto_comparacao(p["word"]).replace(" ", "") or f"\u00a7{i}")
           for i, p in enumerate(palavras)]
    ref = [n for _, n in tokens_letra]

    matcher = SequenceMatcher(None, asr, ref, autojunk=False)
    similaridade = matcher.ratio()
    if similaridade < limite:
        print(f"    [LETRA] Similaridade {similaridade:.2f} < {limite:.2f}: letra ignorada "
              f"(identificação provavelmente errada).", flush=True)
        return palavras

    novas = list(palavras)
    ajustadas = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                novo = tokens_letra[j1 + k][0]
                if novo != palavras[i1 + k]["word"].strip():
                    novas[i1 + k] = {**palavras[i1 + k], "word": novo}
                    ajustadas += 1

    print(f"    [LETRA] Similaridade ASR x letra: {similaridade:.2f} | "
          f"{ajustadas} palavra(s) ajustada(s) pela letra.", flush=True)
    return novas


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
def _normalizar_mono_single_pass(caminho_entrada, caminho_saida, taxa_saida=16000):
    """
    Normalização EBU R128 single-pass.

    IMPORTANTE: o filtro `loudnorm` faz upsampling interno para 192kHz para
    medir true peak (parte do algoritmo EBU R128). Se a taxa de saída não
    for especificada explicitamente, o arquivo gerado FICA em 192kHz — o
    que não só desperdiça espaço como pode estourar a memória em etapas
    seguintes que carregam o áudio inteiro (VAD/librosa) para arquivos
    longos. Por isso sempre fixamos `ar=taxa_saida`. Como diarização
    (pyannote) e transcrição (Groq) operam a 16kHz internamente, usar
    16kHz aqui não perde qualidade nenhuma — só evita o inchaço do loudnorm.
    """
    (
        ffmpeg
        .input(caminho_entrada)
        .output(caminho_saida, vn=None, acodec='pcm_s16le', ac=1, ar=taxa_saida,
                af='loudnorm=I=-23:LRA=7:TP=-2')
        .run(quiet=True, overwrite_output=True)
    )


# ---------------------------------------------------------------------------
# ENQUADRAMENTO
# (roda sobre o áudio já convertido em WAV E normalizado)
# Usado no caminho MONO — aqui o áudio É cortado, gerando um novo arquivo
# sem os trechos de silêncio.
#
# [AJUSTADO] Como o corte desloca todos os tempos, a função agora também
# devolve um MAPA DE TEMPO: lista de (inicio_cortado_s, fim_cortado_s,
# inicio_original_s), um item por trecho mantido. Com ele,
# _criar_conversor_tempo converte qualquer timestamp do áudio cortado de
# volta para o tempo do áudio ORIGINAL.
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

    # [NOVO] Mapa de tempo: (inicio_cortado, fim_cortado, inicio_original), em segundos
    mapa_tempo = []
    if len(intervalos) == 0:
        mapa_tempo.append((0.0, float(duracao_antes), 0.0))  # nada foi cortado: mapa identidade
    else:
        acumulado = 0.0
        for ini, fim in intervalos:
            duracao_trecho = float(fim - ini) / sr
            mapa_tempo.append((acumulado, acumulado + duracao_trecho, float(ini) / sr))
            acumulado += duracao_trecho
    return mapa_tempo


def _criar_conversor_tempo(mapa_tempo):
    """
    Devolve uma função converter(t, eh_fim=False) que leva um tempo do áudio
    CORTADO de volta ao tempo do áudio ORIGINAL. Sem mapa (None/vazio), é a
    função identidade.

    `eh_fim=True` resolve a ambiguidade nas emendas: um tempo exatamente na
    junção de dois trechos pertence ao FIM do trecho anterior (e não ao
    início do próximo), o que é o correto para o "fim" de uma palavra ou de
    um segmento. Tempos ligeiramente fora do trecho (imprecisão do Whisper)
    são limitados às bordas do trecho.
    """
    if not mapa_tempo:
        return lambda t, eh_fim=False: t

    inicios_cortados = [m[0] for m in mapa_tempo]

    def converter(t, eh_fim=False):
        if eh_fim:
            k = bisect.bisect_left(inicios_cortados, t) - 1
        else:
            k = bisect.bisect_right(inicios_cortados, t) - 1
        k = max(0, min(k, len(mapa_tempo) - 1))
        ini_cortado, fim_cortado, ini_original = mapa_tempo[k]
        t_limitado = min(max(t, ini_cortado), fim_cortado)
        return ini_original + (t_limitado - ini_cortado)

    return converter


def _remapear_segmentos_diarizacao(segmentos, converter):
    """Converte inicio/fim dos segmentos da diarização para o tempo original (in-place)."""
    for s in segmentos:
        ini = converter(s["inicio"])
        fim = converter(s["fim"], eh_fim=True)
        s["inicio"] = round(ini, 2)
        s["fim"] = round(max(fim, ini), 2)
    return segmentos


def _remapear_palavras(palavras, converter):
    """Converte start/end das palavras transcritas para o tempo original."""
    resultado = []
    for p in palavras:
        ini = converter(p["start"])
        fim = converter(p["end"], eh_fim=True)
        resultado.append({**p, "start": ini, "end": max(fim, ini)})
    return resultado


# ---------------------------------------------------------------------------
# DETECÇÃO DE INTERVALOS DE FALA — SEM CORTAR O ÁUDIO (usado no ESTÉREO/MULTICANAL
# e também para cruzar com a transcrição na checagem de "fala fantasma")
# ---------------------------------------------------------------------------
# Diferente de _enquadrar_e_remover_silencio, esta função NÃO gera um
# novo arquivo e NÃO corta nada — ela só identifica onde está a fala.
# Isso é importante para os caminhos com múltiplos canais: se cortássemos
# cada canal de forma independente (cada um com silêncios diferentes
# removidos), os timestamps deixariam de corresponder ao tempo real do
# áudio original, e a comparação entre canais (que depende de
# sobreposição de tempo) ficaria errada.
#
# Retorna (intervalos, sr): os intervalos vêm em AMOSTRAS (não segundos),
# por isso devolvemos também o sample rate — quem for cruzar esses
# intervalos com timestamps em segundos (ex: a checagem de alucinação)
# precisa dividir por sr.
def _detectar_intervalos_fala(caminho_entrada, top_db=40):
    y, sr = librosa.load(caminho_entrada, sr=None, mono=True)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    intervalos = librosa.effects.split(
        y, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length
    )

    return intervalos, sr


# ---------------------------------------------------------------------------
# NORMALIZAÇÃO EBU R128 — TWO-PASS PARA UM ÚNICO CANAL MONO
# (usado nos canais já separados do MULTICANAL — cada canal vira um
# arquivo mono independente, mas continua usando two-pass, seguindo a
# regra já combinada: a decisão de single/two-pass é pelo tipo de
# arquivo ORIGINAL — mono usa single-pass, multicanal usa two-pass,
# mesmo depois de cada canal do multicanal virar um arquivo mono próprio)
# ---------------------------------------------------------------------------
def _medir_loudness_mono(caminho_entrada):
    """Medição EBU R128 (passada 1/2, ou diagnóstico avulso) para um único arquivo mono."""
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


def _normalizar_mono_two_pass(caminho_entrada, caminho_saida, taxa_saida=16000):
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
        _normalizar_mono_single_pass(caminho_entrada, caminho_saida, taxa_saida=taxa_saida)
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
            .output(caminho_saida, vn=None, acodec='pcm_s16le', ac=1, ar=taxa_saida, af=filtro_preciso)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr — rode com capture_stderr=True)"
        print(f"    [FFMPEG ERROR] normalizar two-pass {caminho_entrada}:\n{stderr_txt}", flush=True)
        raise


# ---------------------------------------------------------------------------
# ANÁLISE DE QUALIDADE PÓS-NORMALIZAÇÃO
# ---------------------------------------------------------------------------
# Roda logo depois de QUALQUER normalização (mono single-pass, canais do
# estéreo, canais do multicanal two-pass). Não altera o áudio — é só
# diagnóstico, tanto para log quanto para devolver no retorno da API,
# junto da transcrição.
def _analisar_qualidade_audio(caminho_audio, rotulo=""):
    """
    Diagnóstico de qualidade sobre um arquivo JÁ normalizado:
    - Loudness integrada, LRA e true peak (via ffmpeg loudnorm, só medindo)
    - Pico e RMS em dBFS (via numpy, sobre as amostras reais do arquivo)
    - Proporção de amostras "coladas" no teto (indício de clipping)

    Gera alertas quando algo foge do esperado para o alvo de normalização
    usado no pipeline (I=-23 LUFS, TP=-2 dBTP).
    """
    loudness_integrada = true_peak = lra = None
    try:
        medidas = _medir_loudness_mono(caminho_audio)
        loudness_integrada = float(medidas['input_i'])
        true_peak = float(medidas['input_tp'])
        lra = float(medidas['input_lra'])
    except Exception as e:
        print(f"    [QUALIDADE] Aviso: não foi possível medir loudness de {caminho_audio}: {e}", flush=True)

    pico_dbfs = rms_dbfs = None
    proporcao_clipping = 0.0
    try:
        y, sr = sf.read(caminho_audio)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float64)

        pico = float(np.max(np.abs(y))) if y.size else 0.0
        pico_dbfs = 20 * np.log10(pico) if pico > 0 else float('-inf')

        rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
        rms_dbfs = 20 * np.log10(rms) if rms > 0 else float('-inf')

        amostras_no_teto = int(np.sum(np.abs(y) >= 0.999))
        proporcao_clipping = amostras_no_teto / y.size if y.size else 0.0
    except Exception as e:
        print(f"    [QUALIDADE] Aviso: não foi possível analisar amostras de {caminho_audio}: {e}", flush=True)

    alertas = []
    if proporcao_clipping > 0.001:
        alertas.append(f"possível clipping ({proporcao_clipping * 100:.2f}% das amostras no teto)")
    if loudness_integrada is not None and loudness_integrada < -40:
        alertas.append(f"áudio muito baixo mesmo após normalização ({loudness_integrada:.1f} LUFS)")
    if true_peak is not None and true_peak > -1.0:
        alertas.append(f"true peak acima do recomendado ({true_peak:.1f} dBTP, alvo é -2 dBTP)")
    if rms_dbfs is not None and rms_dbfs != float('-inf') and rms_dbfs < -50:
        alertas.append(f"RMS muito baixo ({rms_dbfs:.1f} dBFS) — possível trecho quase silencioso")

    relatorio = {
        "canal": rotulo,
        "loudness_integrada_lufs": loudness_integrada,
        "true_peak_dbtp": true_peak,
        "lra": lra,
        "pico_dbfs": pico_dbfs,
        "rms_dbfs": rms_dbfs,
        "proporcao_clipping": proporcao_clipping,
        "alertas": alertas,
    }

    prefixo = f"[QUALIDADE{' - ' + rotulo if rotulo else ''}]"
    if loudness_integrada is not None:
        print(f"    {prefixo} Loudness: {loudness_integrada:.1f} LUFS | "
              f"TP: {true_peak:.1f} dBTP | LRA: {lra:.1f} | "
              f"Pico: {pico_dbfs:.1f} dBFS | RMS: {rms_dbfs:.1f} dBFS", flush=True)
    if alertas:
        for alerta in alertas:
            print(f"    {prefixo} ALERTA: {alerta}", flush=True)
    else:
        print(f"    {prefixo} Nenhum problema detectado.", flush=True)

    return relatorio


# ---------------------------------------------------------------------------
# Comparação de Falantes (difflib, re, unicodedata) — Estéreo/Multicanal
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


def _tokenizar_comparacao(texto):
    """Normaliza e quebra em palavras — usado pela comparação por token."""
    texto_normalizado = _normalizar_texto_comparacao(texto)
    if not texto_normalizado:
        return []
    return texto_normalizado.split()


def _similaridade_texto(texto_a, texto_b):
    """
    Retorna uma similaridade entre 0 e 1.
    1.0 = textos praticamente iguais
    0.0 = textos completamente diferentes

    Combina duas métricas calculadas por PALAVRA:

    - SequenceMatcher.ratio() sobre a lista de tokens: mede o quanto as
      falas se alinham NA MESMA ORDEM.
    - Jaccard (interseção / união dos conjuntos de palavras): mede o
      quanto o VOCABULÁRIO é parecido, mesmo com ordem diferente.

    A média das duas é mais tolerante a variações do que comparar
    caractere a caractere, sem exigir um texto idêntico.
    """
    tokens_a = _tokenizar_comparacao(texto_a)
    tokens_b = _tokenizar_comparacao(texto_b)

    if not tokens_a or not tokens_b:
        return 0.0

    ratio_sequencial = SequenceMatcher(None, tokens_a, tokens_b).ratio()

    conjunto_a, conjunto_b = set(tokens_a), set(tokens_b)
    uniao = conjunto_a | conjunto_b
    jaccard = len(conjunto_a & conjunto_b) / len(uniao) if uniao else 0.0

    return (ratio_sequencial + jaccard) / 2


def _calcular_sobreposicao(inicio_a, fim_a, inicio_b, fim_b):
    """
    Calcula quanto dois segmentos se sobrepõem.
    Retorna um valor entre 0 e 1 baseado no menor segmento.
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

    Quando os dois critérios são atendidos, considera-se que
    provavelmente é a mesma fala captada pelos dois canais.
    O canal esquerdo é usado como referência quando os dois
    segmentos são considerados duplicados.
    """

    resultado_esquerda = []
    resultado_direita = []

    for segmento in segmentos_esquerda:
        novo_segmento = segmento.copy()
        novo_segmento["canal"] = "C0"
        novo_segmento["duplicado"] = False
        novo_segmento["similaridade"] = 0.0
        novo_segmento["sobreposicao"] = 0.0
        resultado_esquerda.append(novo_segmento)

    for segmento in segmentos_direita:
        novo_segmento = segmento.copy()
        novo_segmento["canal"] = "C1"
        novo_segmento["duplicado"] = False
        novo_segmento["similaridade"] = 0.0
        novo_segmento["sobreposicao"] = 0.0
        resultado_direita.append(novo_segmento)

    for segmento_l in resultado_esquerda:

        melhor_correspondencia = None
        melhor_score = 0.0

        for segmento_r in resultado_direita:

            sobreposicao = _calcular_sobreposicao(
                segmento_l["inicio"],
                segmento_l["fim"],
                segmento_r["inicio"],
                segmento_r["fim"]
            )

            if sobreposicao < limite_sobreposicao:
                continue

            similaridade = _similaridade_texto(
                segmento_l.get("texto", ""),
                segmento_r.get("texto", "")
            )

            if similaridade < limite_similaridade:
                continue

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

        if melhor_correspondencia:
            segmento_r = melhor_correspondencia["segmento"]
            sobreposicao = melhor_correspondencia["sobreposicao"]
            similaridade = melhor_correspondencia["similaridade"]

            segmento_l["sobreposicao"] = sobreposicao
            segmento_l["similaridade"] = similaridade
            segmento_r["sobreposicao"] = sobreposicao
            segmento_r["similaridade"] = similaridade

            segmento_r["duplicado"] = True

    resultado = (
        resultado_esquerda
        +
        resultado_direita
    )

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
    entre si, usando o mesmo critério de sobreposição temporal +
    similaridade de texto já usado no estéreo.

    Quando dois segmentos de canais DIFERENTES são considerados a mesma
    fala (duplicata), mantém-se o segmento cujo "inicio" é mais cedo no
    tempo, e marca-se o outro como duplicado, não importa em qual canal.
    """
    todos_segmentos = []
    for indice_canal, resultado_canal in enumerate(resultados_por_canal):
        for segmento in resultado_canal["segmentos"]:
            novo_segmento = segmento.copy()
            novo_segmento["canal"] = f"C{indice_canal}"
            novo_segmento["_canal_idx"] = indice_canal
            novo_segmento["duplicado"] = False
            novo_segmento["similaridade"] = 0.0
            novo_segmento["sobreposicao"] = 0.0
            todos_segmentos.append(novo_segmento)

    total_segmentos = len(todos_segmentos)

    for i in range(total_segmentos):
        segmento_a = todos_segmentos[i]

        if segmento_a["duplicado"]:
            continue

        for j in range(i + 1, total_segmentos):
            segmento_b = todos_segmentos[j]

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

            segmento_a["sobreposicao"] = sobreposicao
            segmento_a["similaridade"] = similaridade
            segmento_b["sobreposicao"] = sobreposicao
            segmento_b["similaridade"] = similaridade

            if segmento_a["inicio"] <= segmento_b["inicio"]:
                segmento_b["duplicado"] = True
            else:
                segmento_a["duplicado"] = True
                break

    for segmento in todos_segmentos:
        segmento.pop("_canal_idx", None)

    return todos_segmentos


# ---------------------------------------------------------------------------
# PROGRESSO DA DIARIZAÇÃO (hook customizado, seguro para threads paralelas)
# ---------------------------------------------------------------------------
# O ProgressHook padrão do pyannote usa barras tqdm/rich que dependem de
# controlar o cursor do terminal — quando dois canais rodam em paralelo
# (ThreadPoolExecutor) e escrevem no mesmo console ao mesmo tempo, essas
# barras se sobrepõem e viram bagunça visual. Este hook, em vez disso,
# imprime uma LINHA NOVA E COMPLETA a cada marco de 10% de cada etapa
# interna do pyannote (segmentação, embedding, clustering...), prefixada
# com um rótulo (ex: nome do canal) — assim, mesmo intercalado entre
# threads, cada linha continua legível.
class _HookProgressoDiarizacao:
    def __init__(self, rotulo=""):
        self.rotulo = rotulo
        self._ultimo_marco = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if not total or completed is None:
            return
        percentual = int((completed / total) * 100)
        marco = (percentual // 10) * 10
        if self._ultimo_marco.get(step_name, -1) != marco:
            self._ultimo_marco[step_name] = marco
            prefixo = f"[DIARIZAÇÃO{' - ' + self.rotulo if self.rotulo else ''}]"
            print(f"    {prefixo} {step_name}: {marco}% ({completed}/{total})", flush=True)


# ---------------------------------------------------------------------------
# PÓS-PROCESSAMENTO DA DIARIZAÇÃO — reduz dois tipos de erro comuns do
# pyannote: (1) fragmentação espúria (segmentos-ilha muito curtos) e
# (2) vazamento de palavras na fronteira entre dois falantes.
# ---------------------------------------------------------------------------
LIMIAR_DURACAO_ILHA_DIARIZACAO = 0.5  # segundos
PADDING_FRONTEIRA_DIARIZACAO = 0.20   # segundos


def _suavizar_diarizacao(segmentos, duracao_minima=LIMIAR_DURACAO_ILHA_DIARIZACAO):
    """
    Remove fragmentação espúria: quando um segmento muito curto (ex: um
    "hum" ou ruído mal classificado) aparece cercado pelo MESMO falante
    antes e depois, ele provavelmente é erro do diarizador, não uma troca
    real de falante — é absorvido pelo falante vizinho (o segmento
    anterior é estendido até o fim do segmento seguinte, e a "ilha" no
    meio é descartada).

    NÃO resolve o caso oposto (dois falantes DIFERENTES fundidos em um
    único bloco, como mediador+entrevistado grudados) — isso é limite do
    próprio modelo de diarização, não dá pra corrigir de forma confiável
    só com regras de pós-processamento.
    """
    if len(segmentos) < 3:
        return segmentos

    segmentos = sorted(segmentos, key=lambda s: s["inicio"])
    resultado = [segmentos[0]]

    i = 1
    while i < len(segmentos) - 1:
        atual = segmentos[i]
        anterior = resultado[-1]
        proximo = segmentos[i + 1]
        duracao_atual = atual["fim"] - atual["inicio"]

        eh_ilha = (
            duracao_atual < duracao_minima
            and anterior["falante"] == proximo["falante"]
            and atual["falante"] != anterior["falante"]
        )

        if eh_ilha:
            anterior["fim"] = proximo["fim"]
            i += 2
            continue

        resultado.append(atual)
        i += 1

    if i == len(segmentos) - 1:
        resultado.append(segmentos[-1])

    return resultado


def _aplicar_padding_fronteiras(segmentos, padding=PADDING_FRONTEIRA_DIARIZACAO):
    """
    Encolhe o FIM de cada segmento em até `padding` segundos sempre que
    ele for seguido de perto por um segmento de OUTRO falante, cedendo
    essa margem ambígua da fronteira para quem vem a seguir.

    Motivação: o diarizador tende a detectar a troca de falante um pouco
    ATRASADA (a fronteira "fim" do falante anterior chega tarde demais),
    então palavras finais de frase (ex: "Bom,", "Né?") acabam presas no
    bloco de quem NÃO as disse, quando na verdade já pertencem ao
    próximo falante. Encolher a cauda do segmento anterior devolve essa
    margem ambígua para o falante seguinte na hora de casar as palavras
    transcritas com os segmentos (_combinar_diarizacao_transcricao).

    Isso é uma heurística, não uma correção perfeita: só cobre UM sentido
    do erro (palavra de quem entra presa em quem sai). O sentido oposto
    (ex: "Né?" que pertence a quem saiu) e frases cortadas no meio são
    tratados por _reparar_fronteiras_por_regra e _reparar_fronteiras_llm.

    [AJUSTADO] As palavras que caem no "vão" aberto por este encolhimento
    agora são atribuídas ao segmento mais próximo em _falante_da_palavra,
    em vez de virarem "DESCONHECIDO".
    """
    segmentos = sorted(segmentos, key=lambda s: s["inicio"])
    for i in range(len(segmentos) - 1):
        atual, proximo = segmentos[i], segmentos[i + 1]
        if atual["falante"] == proximo["falante"]:
            continue
        atual["fim"] = max(atual["fim"] - padding, atual["inicio"])
    return segmentos


# ---------------------------------------------------------------------------
# DIARIZAÇÃO (pyannote.audio, local — com aceleração por GPU quando disponível
# e progresso por canal via _HookProgressoDiarizacao)
# ---------------------------------------------------------------------------
def _carregar_pipeline_diarizacao():
    global _diarization_pipeline
    with _diarization_lock:
        if _diarization_pipeline is None:
            if not HF_TOKEN:
                raise ValueError("HF_TOKEN não encontrado no .env. Necessário para usar o pyannote.audio.")
            print(f"    Carregando modelo de diarização '{MODELO_DIARIZACAO}' "
                  f"(pode demorar na primeira vez)...")
            _diarization_pipeline = PyannotePipeline.from_pretrained(
                MODELO_DIARIZACAO, token=HF_TOKEN
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _diarization_pipeline.to(device)
            print(f"    [DIARIZAÇÃO] Rodando em: {device} "
                  f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    return _diarization_pipeline


def _diarizar_audio(caminho_wav, rotulo=""):
    pipeline = _carregar_pipeline_diarizacao()
    # Serializa a INFERÊNCIA em si (não só o carregamento): dois canais
    # nunca rodam pyannote ao mesmo tempo na mesma instância de modelo.
    with _diarization_lock:
        with _HookProgressoDiarizacao(rotulo=rotulo) as hook:
            resultado = pipeline(caminho_wav, hook=hook)

    # [AJUSTADO] Se o pipeline expõe a diarização "exclusiva" (sem
    # sobreposição entre falantes, pyannote 4.x / community-1), ela é
    # preferida: foi pensada justamente para casar com timestamps de ASR.
    exclusiva = getattr(resultado, "exclusive_speaker_diarization", None)
    if exclusiva is not None:
        diarizacao = exclusiva
    elif hasattr(resultado, "speaker_diarization"):
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

    # Pós-processamento: reduz fragmentação espúria e vazamento de
    # palavras na fronteira entre falantes (ver funções acima).
    segmentos = _suavizar_diarizacao(segmentos)
    segmentos = _aplicar_padding_fronteiras(segmentos)

    return segmentos


# ---------------------------------------------------------------------------
# ORGANIZAÇÃO DE FALANTES: canal + falante_local + falante_global
# ---------------------------------------------------------------------------
def _atribuir_falantes_globais(segmentos, canal_padrao="C0"):
    """
    Preenche "canal" (usando `canal_padrao` quando o segmento ainda não tem
    um, como no fluxo mono) e substitui o campo "falante" por dois novos
    campos: "falante_local" (rótulo bruto da diarização) e "falante_global"
    (identidade única atribuída por canal+falante_local, em ordem
    cronológica de primeira aparição). Modifica e retorna a mesma lista.
    """
    segmentos_por_tempo = sorted(segmentos, key=lambda s: s.get("inicio", 0))

    mapa_falante_global = {}
    proximo_numero = 1
    for segmento in segmentos_por_tempo:
        canal = segmento.get("canal") or canal_padrao
        falante_local = segmento.get("falante", "DESCONHECIDO")
        chave = (canal, falante_local)
        if chave not in mapa_falante_global:
            mapa_falante_global[chave] = f"Falante {proximo_numero}"
            proximo_numero += 1

    for segmento in segmentos:
        canal = segmento.get("canal") or canal_padrao
        falante_local = segmento.pop("falante", "DESCONHECIDO")
        segmento["canal"] = canal
        segmento["falante_local"] = falante_local
        segmento["falante_global"] = mapa_falante_global[(canal, falante_local)]

    return segmentos


# ---------------------------------------------------------------------------
# UNIFICAÇÃO DE FALANTES POR SEMELHANÇA (mesma pessoa em canais diferentes,
# mesmo quando a remoção de duplicatas não descartou nenhum dos blocos)
# ---------------------------------------------------------------------------
# A remoção de duplicatas (_comparar_falas_estereo / _remover_duplicatas_multicanal)
# compara BLOCO INTEIRO contra BLOCO INTEIRO. Isso falha quando a diarização
# corta a fala do MESMO falante em blocos diferentes em cada canal (ex: um
# canal captou tudo como um bloco só de 2s a 118s, o outro canal cortou em
# 3 blocos por causa de um ruído no meio) — nesse caso os textos têm
# tamanhos bem diferentes, a similaridade SIMÉTRICA cai abaixo do limite, e
# cada lado acaba virando um "Falante" global diferente mesmo sendo a
# mesma pessoa (vazamento de microfone entre canais).
#
# Esta etapa roda DEPOIS de _atribuir_falantes_globais e corrige isso: em
# vez de comparar blocos inteiros, usa CONTENÇÃO assimétrica (quanto do
# texto do bloco MENOR aparece dentro do bloco MAIOR) — resolve o caso
# acima, já que o bloco menor é sempre um trecho quase idêntico ao que
# aparece dentro do bloco maior. Uma exigência de tamanho mínimo evita
# falso-positivo em blocos curtos demais (ex: um "e" ou "ok" avulso, que
# bateria por acaso em qualquer texto longo).
LIMITE_SOBREPOSICAO_MESMO_FALANTE = 0.5
LIMITE_CONTENCAO_MESMO_FALANTE = 0.6
MINIMO_TOKENS_MESMO_FALANTE = 4


def _similaridade_contencao(texto_menor, texto_maior):
    """
    Quanto do texto MENOR está contido no texto MAIOR (0 a 1) —
    assimétrico, ao contrário de _similaridade_texto. Usado quando os
    dois blocos podem ter tamanhos bem diferentes (um é um trecho mais
    curto dentro do outro), caso em que a similaridade simétrica
    subestima o quanto eles são "a mesma fala".
    """
    tokens_menor = _tokenizar_comparacao(texto_menor)
    tokens_maior = _tokenizar_comparacao(texto_maior)
    if not tokens_menor or not tokens_maior:
        return 0.0
    matcher = SequenceMatcher(None, tokens_menor, tokens_maior, autojunk=False)
    total_correspondido = sum(bloco.size for bloco in matcher.get_matching_blocks())
    return total_correspondido / len(tokens_menor)


def _unificar_falantes_por_semelhanca(
    segmentos,
    limite_sobreposicao=LIMITE_SOBREPOSICAO_MESMO_FALANTE,
    limite_contencao=LIMITE_CONTENCAO_MESMO_FALANTE,
    minimo_tokens=MINIMO_TOKENS_MESMO_FALANTE,
):
    """
    Agrupa identidades (canal, falante_local) DIFERENTES que na verdade são
    a mesma pessoa: sobreposição temporal + texto de um contido no outro.
    Nunca une duas identidades do MESMO canal (isso é papel da diarização
    em si, não desta etapa). Une via union-find e renumera falante_global
    por ordem cronológica de primeira aparição do grupo unido. Modifica e
    retorna a mesma lista de segmentos.
    """
    identidades = {}
    for s in segmentos:
        chave = (s["canal"], s["falante_local"])
        identidades.setdefault(chave, []).append(s)

    pais = {chave: chave for chave in identidades}

    def encontrar(chave):
        while pais[chave] != chave:
            pais[chave] = pais[pais[chave]]
            chave = pais[chave]
        return chave

    def unir(a, b):
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            pais[raiz_b] = raiz_a

    chaves = list(identidades.keys())
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            chave_a, chave_b = chaves[i], chaves[j]
            if chave_a[0] == chave_b[0]:
                continue  # mesmo canal — não é um caso de vazamento entre canais
            if encontrar(chave_a) == encontrar(chave_b):
                continue

            encontrou_evidencia = False
            for seg_a in identidades[chave_a]:
                for seg_b in identidades[chave_b]:
                    sobreposicao = _calcular_sobreposicao(
                        seg_a["inicio"], seg_a["fim"], seg_b["inicio"], seg_b["fim"]
                    )
                    if sobreposicao < limite_sobreposicao:
                        continue

                    texto_a = seg_a.get("texto", "")
                    texto_b = seg_b.get("texto", "")
                    tokens_a = _tokenizar_comparacao(texto_a)
                    tokens_b = _tokenizar_comparacao(texto_b)
                    menor, maior = (texto_a, texto_b) if len(tokens_a) <= len(tokens_b) else (texto_b, texto_a)

                    if min(len(tokens_a), len(tokens_b)) < minimo_tokens:
                        continue  # bloco curto demais pra ser evidência confiável

                    if _similaridade_contencao(menor, maior) >= limite_contencao:
                        unir(chave_a, chave_b)
                        encontrou_evidencia = True
                        break
                if encontrou_evidencia:
                    break

    primeira_aparicao = {}
    for chave, segs in identidades.items():
        raiz = encontrar(chave)
        primeira_aparicao[raiz] = min(
            primeira_aparicao.get(raiz, float('inf')),
            min(s["inicio"] for s in segs)
        )

    grupos_para_numero = {}
    proximo_numero = 1
    for raiz in sorted(set(encontrar(c) for c in chaves), key=lambda r: primeira_aparicao[r]):
        grupos_para_numero[raiz] = f"Falante {proximo_numero}"
        proximo_numero += 1

    for s in segmentos:
        chave = (s["canal"], s["falante_local"])
        s["falante_global"] = grupos_para_numero[encontrar(chave)]

    return segmentos


# ---------------------------------------------------------------------------
# FILTRO DE "FALA FANTASMA" (alucinação do Whisper em trechos sem fala real)
# ---------------------------------------------------------------------------
LIMIAR_NO_SPEECH_PROB = 0.6
LIMIAR_AVG_LOGPROB = -1.0
LIMIAR_COMPRESSION_RATIO = 2.4

FRASES_ALUCINACAO_CONHECIDAS = {
    "thank you",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to the channel",
    "obrigado por assistir",
    "se inscreva no canal",
    "inscreva se no canal",
    "curta e compartilhe",
    "ate a proxima",
    "legendas pela comunidade amara org",
    "subtitles by the amara org community",
    "www amara org",
}

# [NOVO] Padrões de "lixo de legenda" que o Whisper herda do dataset de
# treino (ex: "Legenda Adriana Zanotto"). Aplicados sobre o texto JÁ
# normalizado (minúsculo, sem acento/pontuação).
PADROES_LEGENDA_ALUCINADA = [re.compile(p) for p in (
    r"^legendas?( \w+){1,5}$",                                   # "Legenda Adriana Zanotto"
    r"^legendas? (pela|pelos|por|de|feitas? por|realizadas? por)\b.*",
    r"^(transcricao|traducao|revisao|sincronizacao)( e legendas?)? (por|pela|de)\b.*",
    r"^subtitles? (by|from)\b.*",
    r"^captions? by\b.*",
    r"\bamara org\b",
)]


def _eh_legenda_alucinada(texto_norm):
    return any(p.search(texto_norm) for p in PADROES_LEGENDA_ALUCINADA)


def _tempo_sobreposto_com_vad(inicio, fim, intervalos_fala, sr):
    """
    Soma quanto tempo (em segundos) do trecho [inicio, fim] cai dentro de
    algum intervalo de fala detectado pelo VAD. `intervalos_fala` vem de
    _detectar_intervalos_fala, em AMOSTRAS — por isso recebe `sr` para
    converter para segundos antes de comparar com inicio/fim.
    """
    if intervalos_fala is None or sr is None or len(intervalos_fala) == 0:
        return None

    sobreposicao_total = 0.0
    for ini_amostra, fim_amostra in intervalos_fala:
        ini_s, fim_s = ini_amostra / sr, fim_amostra / sr
        inicio_sobre = max(inicio, ini_s)
        fim_sobre = min(fim, fim_s)
        if fim_sobre > inicio_sobre:
            sobreposicao_total += (fim_sobre - inicio_sobre)

    return sobreposicao_total


def _segmento_e_provavel_alucinacao(segmento, intervalos_fala=None, sr=None):
    """
    Decide se um segmento do Whisper é provavelmente uma "fala fantasma".
    Combina métricas do modelo com o cruzamento contra o VAD sempre que
    possível, para não arriscar apagar fala real por engano.
    """
    texto = _campo(segmento, 'text', '') or ''
    texto_norm = _normalizar_texto_comparacao(texto)

    # [NOVO] Lixo de legenda: descarte direto, sem depender do VAD (em música
    # o VAD por energia enxerga o beat como "fala" e não pegaria isso).
    if _eh_legenda_alucinada(texto_norm):
        print(f"    [FALA FANTASMA] Legenda alucinada descartada: '{texto.strip()}'", flush=True)
        return True

    no_speech_prob = _campo(segmento, 'no_speech_prob')
    avg_logprob = _campo(segmento, 'avg_logprob')
    compression_ratio = _campo(segmento, 'compression_ratio')
    inicio = _campo(segmento, 'start')
    fim = _campo(segmento, 'end')

    suspeita_por_metricas = False
    if no_speech_prob is not None and avg_logprob is not None:
        suspeita_por_metricas = (
            no_speech_prob > LIMIAR_NO_SPEECH_PROB and avg_logprob < LIMIAR_AVG_LOGPROB
        )
    if compression_ratio is not None and compression_ratio > LIMIAR_COMPRESSION_RATIO:
        suspeita_por_metricas = True

    suspeita_por_frase_conhecida = texto_norm in FRASES_ALUCINACAO_CONHECIDAS

    if not (suspeita_por_metricas or suspeita_por_frase_conhecida):
        return False

    if inicio is None or fim is None:
        return suspeita_por_metricas and suspeita_por_frase_conhecida

    duracao = fim - inicio
    sobreposicao_vad = _tempo_sobreposto_com_vad(inicio, fim, intervalos_fala, sr)

    if sobreposicao_vad is not None and duracao > 0:
        proporcao_vad = sobreposicao_vad / duracao
        if proporcao_vad < 0.3:
            print(f"    [FALA FANTASMA] Descartando trecho suspeito: "
                  f"'{texto.strip()}' ({inicio:.2f}s-{fim:.2f}s, "
                  f"{proporcao_vad * 100:.0f}% de sobreposição com VAD)", flush=True)
            return True

    if suspeita_por_metricas and suspeita_por_frase_conhecida:
        print(f"    [FALA FANTASMA] Descartando trecho suspeito (métricas + frase conhecida): "
              f"'{texto.strip()}'", flush=True)
        return True

    return False


def _filtrar_falas_fantasma(palavras, segmentos_whisper, intervalos_fala=None, sr=None):
    """
    Remove da lista de palavras transcritas (word-level) aquelas que
    pertencem a um segmento do Whisper identificado como provável
    alucinação. Roda ANTES da combinação com a diarização, para que a
    fala fantasma nunca chegue a virar um bloco de fala "de verdade".
    """
    if not segmentos_whisper:
        return palavras

    segmentos_para_descartar = []
    for segmento in segmentos_whisper:
        inicio = _campo(segmento, 'start')
        fim = _campo(segmento, 'end')
        if inicio is None or fim is None:
            continue
        if _segmento_e_provavel_alucinacao(segmento, intervalos_fala, sr):
            segmentos_para_descartar.append((inicio, fim))

    if not segmentos_para_descartar:
        return palavras

    palavras_filtradas = []
    for palavra in palavras:
        inicio_p = _campo(palavra, 'start')
        fim_p = _campo(palavra, 'end')
        dentro_de_trecho_descartado = (
            inicio_p is not None and fim_p is not None and any(
                inicio_p >= ini_seg - 0.05 and fim_p <= fim_seg + 0.05
                for ini_seg, fim_seg in segmentos_para_descartar
            )
        )
        if not dentro_de_trecho_descartado:
            palavras_filtradas.append(palavra)

    total_removidas = len(palavras) - len(palavras_filtradas)
    if total_removidas:
        print(f"    [FALA FANTASMA] {total_removidas} palavra(s) removida(s) por pertencer(em) "
              f"a trecho(s) identificado(s) como alucinação.", flush=True)

    return palavras_filtradas


# ---------------------------------------------------------------------------
# COMPRESSÃO + CHUNKING PARA UPLOAD NA GROQ (16kHz mono FLAC)
# ---------------------------------------------------------------------------
# A Groq reamostra tudo para 16kHz mono internamente antes de transcrever,
# então enviar em taxa/bit depth maior não melhora a qualidade da
# transcrição — só aumenta o payload e o risco de estourar o limite de
# tamanho (25MB no tier free, 100MB no dev tier). FLAC é lossless, então
# não há perda adicional na compressão em si.
#
# Para áudios longos, mesmo comprimido o arquivo pode passar de 25MB —
# nesse caso dividimos em chunks de duração fixa, transcrevemos cada um
# separadamente e somamos os timestamps de volta à linha do tempo global.
LIMITE_MB_GROQ = 24  # margem de segurança sobre o limite real de 25MB da Groq
DURACAO_CHUNK_SEGUNDOS = 600  # 10 minutos por chunk — boa margem sob 25MB


# ---------------------------------------------------------------------------
# VOCABULÁRIO DE CONTEXTO — reduz alucinação fonética de termos/entidades
# ---------------------------------------------------------------------------
# A API da Groq (assim como a da OpenAI) aceita um "prompt" curto que NÃO é
# transcrito, mas funciona como contexto: influencia como o modelo interpreta
# sons ambíguos, siglas e nomes próprios foneticamente parecidos com outras
# palavras. Ex.: sem contexto, o Whisper pode ouvir "Lula da Silva" como
# "Lula da Silvana", "MEI" como "meio", "pool de imprensa" como "PUM".
#
# [AJUSTADO] Escrito como frase natural (o Whisper responde melhor a
# texto corrido do que a lista solta) e SEM "Jair Bolsonaro", que puxava a
# transcrição de "Eduardo" para "Jair" quando o orador citava os filhos.
# Mantenha CURTO (limite de ~224 tokens). Edite para o vocabulário do
# conteúdo que for transcrever ou sobrescreva via variável de ambiente
# PROMPT_VOCABULARIO_TRANSCRICAO.
#
# [NOVO] Agora há um prompt por perfil de áudio (PERFIL_AUDIO).
PROMPT_CONVERSA = (
    "Entrevista política no Palácio do Planalto, com Lula da Silva, Ronaldo Caiado "
    "e o jornalista Ernesto Paglia, em um pool de imprensa. Assuntos: eleições 2026, "
    "MEI (Microempreendedor Individual), INSS, Daniel Vorcaro, escândalos, desfachatez, "
    "cana, Carluxo, Eduardo e Renan Bolsonaro. Sai governo, entra governo."
)
PROMPT_MUSICA = (
    "Letra de trap brasileiro, com gírias e estrangeirismos: Teto, Doode, Reid, Stef, "
    "Fabin, Ear Kid, Fendi, Balmain, Codein, Vlone, Patek, Glock, MEI, drip, placo, "
    "cash, bitch."
)

PROMPT_VOCABULARIO_TRANSCRICAO = os.getenv(
    "PROMPT_VOCABULARIO_TRANSCRICAO",
    PROMPT_MUSICA if EH_MUSICA else PROMPT_CONVERSA
)

# ---------------------------------------------------------------------------
# CORREÇÕES DETERMINÍSTICAS DE TERMOS CONHECIDOS — aplicadas DEPOIS da
# correção por LLM, 100% locais (sem API externa), para os erros fonéticos
# e de espaçamento recorrentes desta gravação que nem o viés de vocabulário
# nem o LLM corrigiram de forma consistente. Chave = como costuma sair
# errado (sem diferenciar maiúsculas/minúsculas, respeitando limites de
# palavra e [AJUSTADO] tolerando pontuação entre as palavras da chave);
# valor = forma correta.
#
# ATENÇÃO (para o TCC): várias entradas abaixo foram calibradas olhando
# UMA gravação. Para medir a precisão de forma honesta, avalie em áudios
# que não foram usados para montar este dicionário.
# ---------------------------------------------------------------------------
CORRECOES_TERMOS_CONHECIDOS = {
    # Espaçamento residual que o LLM às vezes não corrige
    "enfrentoua": "enfrentou a",
    "eufalo": "eu falo",
    "danielvorcaro": "Daniel Vorcaro",
    # Entidades/termos trocados por proximidade fonética
    "próprios canos": "próprios escândalos",
    "calhado": "Caiado",
    "não vai mandar nenhum palmo": "não vai mandar em um palmo",
    "o maior inimigo do trabalho visou": "o maior inimigo do trabalhador avisou",
    "sai o governo, entra o governo": "Sai governo, entra governo",
}

# [NOVO] No perfil música, o dicionário político não se aplica.
CORRECOES_ATIVAS = {} if EH_MUSICA else CORRECOES_TERMOS_CONHECIDOS


def _preparar_audio_para_transcricao_groq(caminho_entrada):
    caminho_comprimido = caminho_entrada.rsplit('.', 1)[0] + '_groq.flac'
    try:
        (
            ffmpeg
            .input(caminho_entrada)
            .output(caminho_comprimido, ar=16000, ac=1, acodec='flac')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr)"
        print(f"    [FFMPEG ERROR] comprimir para envio à Groq {caminho_entrada}:\n{stderr_txt}", flush=True)
        raise
    return caminho_comprimido


def _dividir_audio_em_chunks(caminho_entrada, duracao_chunk_s=DURACAO_CHUNK_SEGUNDOS):
    """
    Divide um áudio em pedaços de duração fixa (o último fica com o resto),
    sem sobreposição. Usado só quando o arquivo comprimido ainda excede o
    limite de tamanho da Groq. Retorna lista de (caminho_chunk, offset_s).
    """
    duracao_total = librosa.get_duration(path=caminho_entrada)
    chunks = []
    base = caminho_entrada.rsplit('.', 1)[0]
    offset = 0.0
    indice = 0
    while offset < duracao_total:
        caminho_chunk = f"{base}_chunk{indice}.wav"
        duracao_deste_chunk = min(duracao_chunk_s, duracao_total - offset)
        try:
            (
                ffmpeg
                .input(caminho_entrada, ss=offset, t=duracao_deste_chunk)
                .output(caminho_chunk, acodec='pcm_s16le')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            stderr_txt = e.stderr.decode('utf-8', errors='replace') if e.stderr else "(sem stderr)"
            print(f"    [FFMPEG ERROR] cortar chunk {indice} de {caminho_entrada}:\n{stderr_txt}", flush=True)
            raise
        chunks.append((caminho_chunk, offset))
        offset += duracao_chunk_s
        indice += 1
    return chunks


def _normalizar_palavra(palavra, offset):
    """Converte a palavra (dict ou objeto do SDK) num dict simples, com
    timestamp já deslocado pelo offset do chunk — funciona igual para
    chunk único (offset=0) ou múltiplos chunks."""
    return {
        "word": _campo(palavra, "word", ""),
        "start": _campo(palavra, "start", 0.0) + offset,
        "end": _campo(palavra, "end", 0.0) + offset,
    }


def _normalizar_segmento_whisper(segmento, offset):
    """Mesma ideia, para os segmentos do Whisper usados na checagem de
    fala fantasma."""
    campos = {}
    for chave in ("text", "start", "end", "no_speech_prob", "avg_logprob", "compression_ratio"):
        valor = _campo(segmento, chave)
        if valor is not None:
            campos[chave] = valor
    if "start" in campos:
        campos["start"] += offset
    if "end" in campos:
        campos["end"] += offset
    return campos


def _chamar_groq_transcricao(caminho_arquivo, prompt_vocabulario=None):
    with open(caminho_arquivo, 'rb') as f:
        argumentos = dict(
            file=(os.path.basename(caminho_arquivo), f.read()),
            model=MODELO_TRANSCRICAO,          # [AJUSTADO] configurável via .env
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,                   # [NOVO] saída determinística
        )
        if IDIOMA_TRANSCRICAO:
            argumentos["language"] = IDIOMA_TRANSCRICAO   # [NOVO] idioma fixo
        if prompt_vocabulario:
            argumentos["prompt"] = prompt_vocabulario
        return client.audio.transcriptions.create(**argumentos)


def _transcrever_com_timestamps(audio_path, prompt_vocabulario=PROMPT_VOCABULARIO_TRANSCRICAO):
    """
    Transcreve `audio_path` (mono, já normalizado) via Groq. Se o arquivo
    comprimido (16kHz mono FLAC) couber no limite de 25MB, envia direto.
    Se exceder (áudios longos), divide em chunks de DURACAO_CHUNK_SEGUNDOS,
    transcreve cada um separadamente e junta os resultados, ajustando os
    timestamps de cada chunk para a linha do tempo GLOBAL do áudio original.

    `prompt_vocabulario` é repassado à Groq em CADA chunk (o contexto do
    prompt não "acumula" entre chamadas, então precisa ir em todas).
    """
    caminho_upload = _preparar_audio_para_transcricao_groq(audio_path)
    tamanho_mb = os.path.getsize(caminho_upload) / (1024 * 1024)
    print(f"    Tamanho comprimido para Groq: {tamanho_mb:.2f} MB (16kHz mono FLAC, "
          f"original: {os.path.getsize(audio_path) / (1024*1024):.2f} MB) | "
          f"modelo: {MODELO_TRANSCRICAO}, idioma: {IDIOMA_TRANSCRICAO or 'auto'}", flush=True)

    if tamanho_mb <= LIMITE_MB_GROQ:
        try:
            resposta = _chamar_groq_transcricao(caminho_upload, prompt_vocabulario=prompt_vocabulario)
        finally:
            if os.path.exists(caminho_upload):
                os.remove(caminho_upload)

        palavras = [_normalizar_palavra(p, 0.0) for p in (resposta.words or [])]
        segmentos = [_normalizar_segmento_whisper(s, 0.0)
                     for s in (getattr(resposta, "segments", None) or [])]
        return palavras, segmentos

    # --- CHUNKING ---
    print(f"    [AVISO] Arquivo comprimido ({tamanho_mb:.2f} MB) excede {LIMITE_MB_GROQ} MB. "
          f"Dividindo em chunks de {DURACAO_CHUNK_SEGUNDOS}s...", flush=True)
    if os.path.exists(caminho_upload):
        os.remove(caminho_upload)

    chunks = _dividir_audio_em_chunks(audio_path)
    todas_palavras = []
    todos_segmentos = []

    for indice, (caminho_chunk, offset) in enumerate(chunks):
        print(f"    [CHUNK {indice + 1}/{len(chunks)}] Transcrevendo trecho a partir de {offset:.1f}s...", flush=True)
        caminho_chunk_comprimido = _preparar_audio_para_transcricao_groq(caminho_chunk)
        try:
            tamanho_chunk_mb = os.path.getsize(caminho_chunk_comprimido) / (1024 * 1024)
            print(f"    [CHUNK {indice + 1}/{len(chunks)}] Enviando: {tamanho_chunk_mb:.2f} MB", flush=True)
            resposta = _chamar_groq_transcricao(caminho_chunk_comprimido, prompt_vocabulario=prompt_vocabulario)
        finally:
            if os.path.exists(caminho_chunk_comprimido):
                os.remove(caminho_chunk_comprimido)
            if os.path.exists(caminho_chunk):
                os.remove(caminho_chunk)

        todas_palavras.extend(_normalizar_palavra(p, offset) for p in (resposta.words or []))
        todos_segmentos.extend(
            _normalizar_segmento_whisper(s, offset)
            for s in (getattr(resposta, "segments", None) or [])
        )

    return todas_palavras, todos_segmentos


# ---------------------------------------------------------------------------
# [NOVO] ATRIBUIÇÃO DE FALANTE POR PALAVRA + REPARO DE FRONTEIRAS DE TURNO
# ---------------------------------------------------------------------------
def _falante_da_palavra(inicio_p, fim_p, segmentos_falantes):
    """
    Falante com maior sobreposição temporal com a palavra. Se a palavra não
    sobrepõe NENHUM segmento (ex: caiu no vão aberto pelo padding de
    fronteira), usa o segmento mais próximo em vez de "DESCONHECIDO".
    """
    melhor, maior = None, 0.0
    for seg in segmentos_falantes:
        sobre = min(fim_p, seg['fim']) - max(inicio_p, seg['inicio'])
        if sobre > maior:
            maior, melhor = sobre, seg['falante']
    if melhor is not None:
        return melhor
    if not segmentos_falantes:
        return "DESCONHECIDO"

    meio = (inicio_p + fim_p) / 2

    def distancia(seg):
        if seg['inicio'] <= meio <= seg['fim']:
            return 0.0
        return min(abs(meio - seg['inicio']), abs(meio - seg['fim']))

    return min(segmentos_falantes, key=distancia)['falante']


# Palavras que tipicamente FECHAM a fala de quem acabou de falar (tag
# questions). Se abrem um bloco de OUTRO falante, voltam para o anterior.
TAGS_FIM_DE_FRASE = {"ne", "certo", "ta", "entendeu"}
# Palavras que tipicamente ABREM a fala de quem vai falar. Se fecham um
# bloco logo antes da troca de falante, vão para o seguinte.
ABERTURAS_DE_FALA = {"bom", "bem", "olha", "entao"}
GAP_MAXIMO_FRONTEIRA_REGRA = 0.6   # segundos
GAP_MAXIMO_FRONTEIRA_LLM = 1.0     # segundos


def _reparar_fronteiras_por_regra(palavras, gap_max=GAP_MAXIMO_FRONTEIRA_REGRA):
    """
    Corrige, nos DOIS sentidos, palavras de fronteira que o diarizador
    deixou do lado errado: "Né?" abrindo o bloco de B pertence ao fim de A;
    "Bom," fechando o bloco de A pertence ao início de B. Só age quando a
    pausa entre as duas palavras é curta (fala colada). Edite os conjuntos
    acima para o seu tipo de conteúdo.
    """
    for i in range(1, len(palavras)):
        a, b = palavras[i - 1], palavras[i]
        if a["falante"] == b["falante"] or (b["inicio"] - a["fim"]) > gap_max:
            continue
        na = _normalizar_texto_comparacao(a["palavra"])
        nb = _normalizar_texto_comparacao(b["palavra"])
        if nb in TAGS_FIM_DE_FRASE:
            print(f"    [FRONTEIRA] '{b['palavra']}' ({b['inicio']}s) devolvida ao falante anterior", flush=True)
            b["falante"] = a["falante"]
        elif na in ABERTURAS_DE_FALA:
            print(f"    [FRONTEIRA] '{a['palavra']}' ({a['inicio']}s) movida para o falante seguinte", flush=True)
            a["falante"] = b["falante"]
    return palavras


def _extrair_json_objeto(texto):
    """Extrai o primeiro objeto {...} de uma resposta de LLM (tolera cercas de código)."""
    m = re.search(r'\{.*?\}', texto, flags=re.DOTALL)
    if not m:
        raise ValueError("resposta sem objeto JSON")
    return json.loads(m.group(0))


def _reparar_fronteiras_llm(palavras, gap_max=GAP_MAXIMO_FRONTEIRA_LLM, janela=6):
    """
    Em cada troca de falante com pausa curta, mostra ao LLM só o final do
    bloco A e o início do bloco B e pergunta quantas palavras (0 a 3) estão
    do lado errado. O LLM devolve apenas NÚMEROS — nunca reescreve texto —,
    então não há risco de alterar o conteúdo transcrito. Resolve casos que
    regra simples não cobre, como frase cortada no meio
    ("...que deve" | "ser resguardado").
    """
    i = 1
    while i < len(palavras):
        a, b = palavras[i - 1], palavras[i]
        if a["falante"] == b["falante"] or (b["inicio"] - a["fim"]) >= gap_max:
            i += 1
            continue

        ini_a = i
        while ini_a > 0 and palavras[ini_a - 1]["falante"] == a["falante"]:
            ini_a -= 1
        fim_b = i
        while fim_b < len(palavras) and palavras[fim_b]["falante"] == b["falante"]:
            fim_b += 1

        cauda = " ".join(p["palavra"] for p in palavras[max(ini_a, i - janela):i])
        cabeca = " ".join(p["palavra"] for p in palavras[i:min(fim_b, i + janela)])

        prompt = (
            "Um diarizador separou duas falas de pessoas diferentes (A e B) numa "
            "transcrição em português. Às vezes 1 a 3 palavras ficam do lado errado "
            "da fronteira.\n"
            f'Final de A: "{cauda}"\n'
            f'Início de B: "{cabeca}"\n'
            'Responda SOMENTE um JSON: {"a_para_b": n, "b_para_a": m}, com n e m entre '
            "0 e 3 (pelo menos um deles deve ser 0). Mova palavras apenas se uma frase "
            "ficou cortada no meio da fronteira ou se a palavra claramente pertence ao "
            "outro lado (ex: 'né?' no início de B pertence ao fim de A; 'bom,' no fim "
            "de A pertence ao início de B). Se a fronteira estiver correta, responda 0 e 0."
        )
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            dec = _extrair_json_objeto(resp.choices[0].message.content.strip())
            n = max(0, min(3, int(dec.get("a_para_b", 0)), i - ini_a - 1))
            m = max(0, min(3, int(dec.get("b_para_a", 0)), fim_b - i - 1))
            if n and m:
                n = m = 0  # resposta contraditória — ignora
            if n:
                print(f"    [FRONTEIRA-LLM] {n} palavra(s) movida(s) de A para B em {b['inicio']}s: "
                      f"'...{cauda}' | '{cabeca}...'", flush=True)
                for p in palavras[i - n:i]:
                    p["falante"] = b["falante"]
            elif m:
                print(f"    [FRONTEIRA-LLM] {m} palavra(s) movida(s) de B para A em {b['inicio']}s: "
                      f"'...{cauda}' | '{cabeca}...'", flush=True)
                for p in palavras[i:i + m]:
                    p["falante"] = a["falante"]
        except Exception as e:
            print(f"    [FRONTEIRA-LLM] ignorado em {b['inicio']}s: {e}", flush=True)

        i = fim_b
    return palavras


# ---------------------------------------------------------------------------
# COMBINAÇÃO: diarização + transcrição, por palavra
# ---------------------------------------------------------------------------
def _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao):
    palavras_com_falante = []
    for palavra in palavras_transcricao:
        inicio_p, fim_p = palavra['start'], palavra['end']
        falante = _falante_da_palavra(inicio_p, fim_p, segmentos_falantes)   # [AJUSTADO]
        palavras_com_falante.append({
            "inicio": round(inicio_p, 2), "fim": round(fim_p, 2),
            "falante": falante, "palavra": palavra['word'].strip()
        })

    if not palavras_com_falante:
        return []

    # [NOVO] Reparo das fronteiras de turno ANTES de agrupar em blocos
    palavras_com_falante = _reparar_fronteiras_por_regra(palavras_com_falante)
    if REPARAR_FRONTEIRAS_LLM:
        palavras_com_falante = _reparar_fronteiras_llm(palavras_com_falante)

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
def _processar_canal_audio(caminho_audio, intervalos_fala=None, sr=None, rotulo=""):
    """
    Roda diarização + transcrição + combinação para UM canal de áudio
    já normalizado (sem cortar silêncio — os timestamps continuam
    correspondendo ao áudio original, o que é necessário para a
    comparação entre canais funcionar corretamente).

    `intervalos_fala`/`sr` (do VAD já calculado sobre esse mesmo canal)
    são opcionais e usados só para cruzar com a checagem de "fala
    fantasma" — se não forem passados, a checagem cai para o modo mais
    conservador. `rotulo` identifica o canal nos logs de progresso da
    diarização (ex: "esquerdo", "direito", "canal0.wav").

    Retorna um dicionário com a lista de segmentos (blocos de fala)
    desse canal, no mesmo formato usado pelo caminho mono.
    """
    segmentos_falantes = _diarizar_audio(caminho_audio, rotulo=rotulo)
    palavras_transcricao, segmentos_whisper = _transcrever_com_timestamps(caminho_audio)
    palavras_transcricao = _filtrar_falas_fantasma(palavras_transcricao, segmentos_whisper, intervalos_fala, sr)
    segmentos = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)
    return {"segmentos": segmentos}


def _processar_um_canal_multicanal(caminho_canal, top_db):
    """
    Estágio "processamento" de UM canal do MULTICANAL (roda em paralelo
    para cada C0, C1, ..., CN): normaliza (two-pass), roda a análise de
    qualidade sobre o áudio normalizado, detecta intervalos de fala (usado
    tanto para log/diagnóstico quanto para a checagem de fala fantasma) e
    roda diarização + transcrição.
    Retorna os segmentos desse canal, o relatório de qualidade, o caminho
    do arquivo normalizado e os intervalos de fala (log/diagnóstico).
    """
    caminho_normalizado = caminho_canal.rsplit('.', 1)[0] + '_normalizado.wav'
    _normalizar_mono_two_pass(caminho_canal, caminho_normalizado)
    relatorio_qualidade = _analisar_qualidade_audio(caminho_normalizado, rotulo=os.path.basename(caminho_canal))
    intervalos_fala, sr = _detectar_intervalos_fala(caminho_normalizado, top_db)
    resultado_canal = _processar_canal_audio(
        caminho_normalizado, intervalos_fala, sr, rotulo=os.path.basename(caminho_canal)
    )
    return {
        "caminho_normalizado": caminho_normalizado,
        "intervalos_fala": intervalos_fala,
        "segmentos": resultado_canal["segmentos"],
        "qualidade": relatorio_qualidade,
    }


def _pipeline_multicanal(wav_lossless_path, canais, top_db):
    """
    Pipeline completo do MULTICANAL, em estágios explícitos:

        MULTICANAL
            |
      C0  C1  C2 ... CN        <- separar canais
            |
        processamento          <- por canal, em paralelo (normalizar + análise de
            |                       qualidade + VAD + diarizar/transcrever, já
            |                       filtrando fala fantasma)
            |
   comparação entre canais     <- todos os pares, sem canal fixo como referência
            |
    remover apenas duplicatas  <- mantém quem começou a falar primeiro
            |
       resultado final         <- correção de texto

    Retorna (resultado, qualidade_multicanal, canais_multicanal_paths,
    canais_normalizados_paths) — os dois últimos são devolvidos só para o
    chamador poder limpar os arquivos temporários depois.
    """
    rotulos_canais = ", ".join(f"C{i}" for i in range(canais))
    print(f"[5/10] [MULTICANAL] Separando em {canais} canais: {rotulos_canais}")
    canais_multicanal_paths = _separar_canais_multicanal(wav_lossless_path, canais)
    print(f"[5/10] [MULTICANAL] Canais separados: {canais_multicanal_paths}")

    print(f"[6-8/10] [MULTICANAL] Processando os {canais} canais em paralelo "
          f"(normalização two-pass + qualidade + VAD + diarização/transcrição)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=canais) as executor:
        futuros = [
            executor.submit(_processar_um_canal_multicanal, caminho, top_db)
            for caminho in canais_multicanal_paths
        ]
        processados = [futuro.result() for futuro in futuros]

    canais_normalizados_paths = [p["caminho_normalizado"] for p in processados]
    resultados_por_canal = [{"segmentos": p["segmentos"]} for p in processados]
    qualidade_multicanal = [p["qualidade"] for p in processados]

    for i, p in enumerate(processados):
        print(f"    C{i}: {len(p['intervalos_fala'])} intervalos de fala, "
              f"{len(p['segmentos'])} segmentos de fala transcritos.")

    print(f"[9/10] [MULTICANAL] Comparando todos os canais entre si "
          f"(sem canal fixo como referência)...")
    segmentos_comparados = _remover_duplicatas_multicanal(
        resultados_por_canal, limite_sobreposicao=0.70, limite_similaridade=0.80
    )

    resultado_combinado = [
        s for s in segmentos_comparados if not s.get("duplicado", False)
    ]
    total_duplicadas = len(segmentos_comparados) - len(resultado_combinado)
    print(f"[9/10] [MULTICANAL] Duplicatas removidas: {total_duplicadas} "
          f"(mantendo sempre quem começou a falar primeiro)")
    resultado_combinado.sort(key=lambda s: (s.get("inicio", 0), s.get("fim", 0)))

    print(f"[10/10] [MULTICANAL] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
    resultado = _corrigir_blocos(resultado_combinado)

    resultado = _atribuir_falantes_globais(resultado)
    resultado = _unificar_falantes_por_semelhanca(resultado)

    print("\n--- RESULTADO FINAL (MULTICANAL) ---")
    for item in resultado:
        print(f"[{item['inicio']}s - {item['fim']}s] canal={item['canal']} "
              f"{item['falante_global']} (local: {item['falante_local']}): {item['texto_corrigido']}")

    return resultado, qualidade_multicanal, canais_multicanal_paths, canais_normalizados_paths


# ---------------------------------------------------------------------------
# CORREÇÃO DE TEXTO (LLM via Groq) — pontuação, capitalização, termos técnicos
# ---------------------------------------------------------------------------
def _corrigir_texto(texto):
    prompt = (
        "Você é um corretor de transcrições de áudio geradas por reconhecimento "
        "de voz automático (Whisper). Corrija SOMENTE estes tipos de erro:\n"
        "1. Pontuação e capitalização.\n"
        "2. Espaços que ficaram colados entre duas palavras — um erro comum "
        "desse tipo de transcrição, não do texto original. Exemplos: "
        "'paradiversos' -> 'para diversos', 'ministrodo' -> 'ministro do', "
        "'agente' -> 'a gente' (quando o contexto for claramente esse). Sempre "
        "que perceber duas palavras grudadas sem espaço, separe-as.\n"
        "3. Termos, siglas ou nomes próprios que claramente saíram errados por "
        "confusão FONÉTICA do reconhecimento de voz (palavras que soam parecido "
        "mas não fazem sentido no contexto). Use como referência este "
        f"vocabulário do contexto do áudio, quando for relevante: "
        f"{PROMPT_VOCABULARIO_TRANSCRICAO}\n\n"
        "IMPORTANTE sobre o que NÃO fazer: NÃO adicione, remova ou reescreva "
        "o conteúdo. NÃO resuma. NÃO troque uma palavra só porque parece "
        "estranha — troque apenas quando tiver certeza de que é um erro de "
        "transcrição fonética, mantendo o sentido original da fala. E, acima "
        "de tudo, NUNCA junte duas palavras que já estavam separadas no texto "
        "original em uma só — a correção deve ter o MESMO número de palavras "
        "do original (ou mais, ao separar uma palavra grudada), nunca menos. "
        "Responda apenas com o texto corrigido, sem comentários.\n\n"
        f"Texto original: {texto}"
    )

    resposta = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip()


def _remover_artefatos_gagueira(texto):
    """
    [NOVO] Remove um artefato típico do ASR: uma vírgula colada (sem espaço)
    a uma letra isolada, como em 'que,e adoecer' -> 'que adoecer'. Em texto
    normal a vírgula é sempre seguida de espaço, então esse padrão é
    artefato de gagueira/hesitação. Só age entre LETRAS (não mexe em
    números como '1,5').
    """
    novo = re.sub(r'(?<=[^\W\d_]),[^\W\d_]\b', '', texto)
    if novo != texto:
        print(f"    [ARTEFATO] gagueira removida: '{texto[:60]}...' ", flush=True)
    return novo


def _reparar_espacamento_generico(texto):
    """
    Corrige um padrão específico de espaço perdido que às vezes passa
    despercebido mesmo depois da correção por LLM: duas palavras que
    viraram uma só porque a segunda começa com maiúscula — tipicamente
    nomes próprios grudados (ex: 'DanielVorcaro' -> 'Daniel Vorcaro').
    Insere um espaço sempre que uma letra minúscula for seguida
    diretamente por uma maiúscula, algo que não deveria ocorrer em
    português normal fora desse tipo de erro de transcrição.
    """
    return re.sub(r'(?<=[a-zà-üç])([A-ZÀ-Ü])', r' \1', texto)


def _padrao_tolerante(errado):
    """
    [NOVO] Monta um regex para a chave do dicionário de correções que
    ignora maiúsculas/minúsculas, respeita limites de palavra e TOLERA
    pontuação/espaços entre as palavras da chave (o LLM costuma inserir
    vírgulas que quebrariam uma busca literal).
    """
    partes = [re.escape(p) for p in re.split(r'[\s,.;:!?]+', errado.strip()) if p]
    return re.compile(r'\b' + r'[\s,.;:!?]+'.join(partes) + r'\b', re.IGNORECASE)


_PADROES_CORRECOES_TERMOS = None


def _aplicar_correcoes_termos(texto, correcoes):
    """
    Aplica um dicionário de correções determinísticas (find/replace, sem
    diferenciar maiúsculas/minúsculas, respeitando limites de palavra e
    tolerando pontuação entre as palavras da chave) DEPOIS da correção
    por LLM. Serve para os erros recorrentes e específicos desta gravação
    que o LLM não pegou de forma consistente — 100% local, sem depender de
    nenhuma API externa.
    """
    global _PADROES_CORRECOES_TERMOS
    if not correcoes:
        return texto
    if _PADROES_CORRECOES_TERMOS is None:
        _PADROES_CORRECOES_TERMOS = [(_padrao_tolerante(errado), certo) for errado, certo in correcoes.items()]
    for padrao, certo in _PADROES_CORRECOES_TERMOS:
        texto = padrao.sub(lambda m, certo=certo: certo, texto)
    return texto


def _texto_preserva_palavras(original, corrigido):
    """
    Confere se a correção por LLM manteve as MESMAS palavras do texto
    original (só mudando pontuação/capitalização/separação, como pedido
    no prompt). Compara as listas de palavras normalizadas (minúsculas,
    sem pontuação/acento) dos dois textos ignorando espaços — assim uma
    separação legítima de palavra grudada ('paradiversos' -> 'para
    diversos') não é penalizada, mas uma reescrita/perda de conteúdo é.
    """
    letras_originais = "".join(_tokenizar_comparacao(original))
    letras_corrigidas = "".join(_tokenizar_comparacao(corrigido))
    if not letras_originais:
        return True
    diferenca = SequenceMatcher(None, letras_originais, letras_corrigidas).ratio()
    return diferenca >= 0.90


def _capitalizar_texto_bruto(texto):
    """
    Fallback simples usado quando a correção por LLM falha na checagem
    de integridade de palavras (_texto_preserva_palavras): só capitaliza
    a primeira letra e garante pontuação final, sem tentar reescrever
    nada — melhor um texto sem pontuação refinada do que um texto com
    palavras alteradas/perdidas.
    """
    texto = texto.strip()
    if not texto:
        return texto
    texto = texto[0].upper() + texto[1:]
    if texto[-1] not in ".!?":
        texto += "."
    return texto


def _corrigir_blocos(blocos):
    for bloco in blocos:
        # [NOVO] remove artefatos de gagueira ANTES do LLM
        texto_bruto = _remover_artefatos_gagueira(bloco["texto"])

        if EH_MUSICA:
            # [NOVO] Perfil música: sem LLM (ele "normaliza" gírias para português padrão)
            texto_corrigido = _capitalizar_texto_bruto(texto_bruto)
        else:
            texto_corrigido = _corrigir_texto(texto_bruto)

            if not _texto_preserva_palavras(texto_bruto, texto_corrigido):
                print(f"    [CORREÇÃO] Aviso: correção por LLM alterou demais o "
                      f"conteúdo do bloco '{bloco['texto'][:60]}...' — revertendo "
                      f"para o texto original (com capitalização básica).", flush=True)
                texto_corrigido = _capitalizar_texto_bruto(texto_bruto)

        texto_corrigido = _reparar_espacamento_generico(texto_corrigido)
        texto_corrigido = _aplicar_correcoes_termos(texto_corrigido, CORRECOES_ATIVAS)

        bloco["texto_corrigido"] = texto_corrigido
    return blocos


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def transcrever_arquivo(caminho_original, top_db=40):
    """
    Recebe um caminho de arquivo já existente em disco e retorna um
    dicionário {"segmentos": [...], "qualidade": [...]} — "qualidade" traz
    um relatório de diagnóstico por canal normalizado (vazio quando o
    fluxo não passa por normalização, como no fallback de WAV puro).
    """
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
            print(f"[3/10] Verificando canais de áudio...")
            info_canais = _detectar_canais(tmp_path)
            canais = info_canais["canais"]
            tipo_canal = info_canais["tipo_canal"]  # "mono" | "estereo" | "multicanal"
            print(f"[3/10] Áudio detectado: {info_canais['tipo_audio']} "
                  f"(layout: {info_canais['layout']}, taxa: {info_canais['taxa_original']} Hz)")

            # [NOVO] Perfil música / FORCAR_MONO: downmix para mono na conversão
            # (evita C0/C1 redundantes com a mesma voz).
            if FORCAR_MONO and tipo_canal != "mono":
                print(f"[3/10] Perfil '{PERFIL_AUDIO}': forçando downmix para MONO "
                      f"(evita C0/C1 redundantes).", flush=True)
                canais = 1
                tipo_canal = "mono"

            print(f"[4/10] Convertendo {extensao} para WAV sem perda de dados (PCM 16-bit)...")
            wav_lossless_path = tmp_path.rsplit('.', 1)[0] + '_lossless.wav'
            informacoes_conversao = _converter_para_wav_lossless(tmp_path, wav_lossless_path, canais)
            print(f"[4/10] WAV lossless gerado: {wav_lossless_path}")

            # [NOVO] Reconhecimento na nuvem ANTES do Demucs (usa a mixagem completa)
            info_musica = None
            prompt_transcricao = PROMPT_VOCABULARIO_TRANSCRICAO
            if RECONHECER_MUSICA:
                print(f"[4/10] Reconhecendo a música na nuvem (AudD)...", flush=True)
                info_musica = _identificar_musica(tmp_path, wav_lossless_path, duracao_bruta)
                if info_musica and info_musica.get("titulo"):
                    prompt_transcricao = (
                        f'{PROMPT_VOCABULARIO_TRANSCRICAO} Faixa: "{info_musica["titulo"]}", '
                        f'de {info_musica.get("artista") or "artista desconhecido"}.'
                    )

            # [NOVO] Isolamento de vocais (Demucs) — opcional, pesado
            if ISOLAR_VOCAIS_DEMUCS:
                print(f"[4/10] Isolando vocais com Demucs (pode demorar)...", flush=True)
                pasta_demucs = tempfile.mkdtemp()
                try:
                    vocais = _isolar_vocais_demucs(wav_lossless_path, pasta_demucs)
                    if vocais is None:
                        print(f"[4/10] Demucs indisponível: seguindo com o áudio original "
                              f"(sem isolar vocais).", flush=True)
                    else:
                        wav_vocais_path = tmp_path.rsplit('.', 1)[0] + '_vocais.wav'
                        ffmpeg.input(vocais).output(
                            wav_vocais_path, acodec='pcm_s16le', ac=1
                        ).run(quiet=True, overwrite_output=True)
                        os.replace(wav_vocais_path, wav_lossless_path)
                        canais, tipo_canal = 1, "mono"
                finally:
                    shutil.rmtree(pasta_demucs, ignore_errors=True)

            if tipo_canal == "estereo":
                print(f"[5/10] Separando canais estéreo em duas vozes (esquerda/direita)...")
                canal_esq_path = tmp_path.rsplit('.', 1)[0] + '_esq.wav'
                canal_dir_path = tmp_path.rsplit('.', 1)[0] + '_dir.wav'
                _separar_canais_estereo(wav_lossless_path, canal_esq_path, canal_dir_path)
                print(f"[5/10] Canais separados: {canal_esq_path} | {canal_dir_path}")

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

                print(f"[6/10] Analisando qualidade dos canais normalizados...")
                relatorio_qualidade_esq = _analisar_qualidade_audio(normalizado_esq_path, rotulo="esquerdo")
                relatorio_qualidade_dir = _analisar_qualidade_audio(normalizado_dir_path, rotulo="direito")
                relatorios_qualidade_estereo = [relatorio_qualidade_esq, relatorio_qualidade_dir]

                print(f"[7/10] Detectando intervalos de fala nos dois canais em paralelo "
                      f"(VAD informativo, NÃO corta o áudio, top_db={top_db})...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futuro_vad_esq = executor.submit(_detectar_intervalos_fala, normalizado_esq_path, top_db)
                    futuro_vad_dir = executor.submit(_detectar_intervalos_fala, normalizado_dir_path, top_db)
                    intervalos_esq, sr_esq = futuro_vad_esq.result()
                    intervalos_dir, sr_dir = futuro_vad_dir.result()

                print(f"[7/10] VAD concluído: canal esquerdo com {len(intervalos_esq)} intervalos de fala, "
                      f"canal direito com {len(intervalos_dir)} intervalos de fala.")

                print(f"[8/10] Diarizando e transcrevendo os dois canais em paralelo...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futuro_esq = executor.submit(
                        _processar_canal_audio, normalizado_esq_path, intervalos_esq, sr_esq, "esquerdo"
                    )
                    futuro_dir = executor.submit(
                        _processar_canal_audio, normalizado_dir_path, intervalos_dir, sr_dir, "direito"
                    )
                    resultado_esq = futuro_esq.result()
                    resultado_dir = futuro_dir.result()

                segmentos_esq = resultado_esq["segmentos"]
                segmentos_dir = resultado_dir["segmentos"]
                print(f"[8/10] Segmentos encontrados — esquerdo: {len(segmentos_esq)}, direito: {len(segmentos_dir)}")

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

                print(f"[10/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
                resultado = _corrigir_blocos(segmentos_sem_duplicacao)

                resultado = _atribuir_falantes_globais(resultado)
                resultado = _unificar_falantes_por_semelhanca(resultado)

                print("\n--- RESULTADO FINAL (ESTÉREO) ---")
                for item in resultado:
                    print(f"[{item['inicio']}s - {item['fim']}s] canal={item['canal']} "
                          f"{item['falante_global']} (local: {item['falante_local']}): {item['texto_corrigido']}")

                return {
                    "segmentos": resultado,
                    "qualidade": relatorios_qualidade_estereo,
                    "informacoes_audio": informacoes_conversao,
                }

            elif tipo_canal == "multicanal":
                resultado, qualidade_multicanal, canais_multicanal_paths, canais_normalizados_paths = _pipeline_multicanal(
                    wav_lossless_path, canais, top_db
                )
                return {
                    "segmentos": resultado,
                    "qualidade": qualidade_multicanal,
                    "informacoes_audio": informacoes_conversao,
                }

            else:
                # --- MONO: fluxo original, sem separação de canais ---
                normalizado_path = tmp_path.rsplit('.', 1)[0] + '_normalizado.wav'

                print(f"[5/10] Normalizando loudness EBU R128 (single-pass)...")
                _normalizar_mono_single_pass(wav_lossless_path, normalizado_path)

                print(f"[5/10] WAV normalizado gerado: {normalizado_path}")

                print(f"[5/10] Analisando qualidade do áudio normalizado...")
                relatorio_qualidade_mono = _analisar_qualidade_audio(normalizado_path, rotulo="mono")

                if REMOVER_SILENCIO_MONO:
                    print(f"[6/10] Enquadrando e removendo trechos de silêncio (VAD por energia, 25ms/10ms, top_db={top_db})...")
                    sem_silencio_path = tmp_path.rsplit('.', 1)[0] + '_final.wav'
                    mapa_tempo_mono = _enquadrar_e_remover_silencio(normalizado_path, sem_silencio_path, top_db=top_db)
                    print(f"[6/10] Concluído: {sem_silencio_path} "
                          f"(mapa de tempo com {len(mapa_tempo_mono)} trechos mantidos)")
                    audio_path = sem_silencio_path
                else:
                    print(f"[6/10] Remoção de silêncio DESATIVADA "
                          f"(REMOVER_SILENCIO_MONO=false ou perfil música): "
                          f"usando o áudio normalizado inteiro.")
                    mapa_tempo_mono = None
                    audio_path = normalizado_path

                # [NOVO] Converte tempos do áudio (possivelmente cortado) -> tempo do áudio ORIGINAL
                converter_tempo_mono = _criar_conversor_tempo(mapa_tempo_mono)

                print(f"[7/10] Identificando falantes (diarização) e detectando intervalos de fala...")
                segmentos_falantes = _diarizar_audio(audio_path, rotulo="mono")
                intervalos_fala_mono, sr_mono = _detectar_intervalos_fala(audio_path, top_db)
                print(f"[7/10] {len(segmentos_falantes)} segmentos de fala identificados (diarização)")

                # [NOVO] Diarização de volta para o tempo ORIGINAL (o VAD acima continua no
                # tempo do áudio processado, pois é cruzado com os segmentos do Whisper,
                # que ainda estão nesse mesmo tempo, na checagem de fala fantasma).
                segmentos_falantes = _remapear_segmentos_diarizacao(segmentos_falantes, converter_tempo_mono)

                print(f"[8/10] Transcrevendo com timestamps por palavra (Groq)...")
                palavras_transcricao, segmentos_whisper = _transcrever_com_timestamps(
                    audio_path, prompt_vocabulario=prompt_transcricao
                )
                palavras_transcricao = _filtrar_falas_fantasma(
                    palavras_transcricao, segmentos_whisper, intervalos_fala_mono, sr_mono
                )

                # [NOVO] Palavras de volta para o tempo ORIGINAL (depois do filtro, que ainda
                # usa o tempo do áudio processado) — assim diarização e palavras ficam na
                # mesma linha do tempo e as pausas reais entre turnos são preservadas.
                palavras_transcricao = _remapear_palavras(palavras_transcricao, converter_tempo_mono)

                # [NOVO] Música identificada com letra: usa a letra como referência de texto
                if info_musica and info_musica.get("letra"):
                    palavras_transcricao = _alinhar_palavras_com_letra(
                        palavras_transcricao, info_musica["letra"]
                    )

                resultado = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)

                print(f"[9/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
                resultado = _corrigir_blocos(resultado)

                resultado = _atribuir_falantes_globais(resultado)
                resultado = _unificar_falantes_por_semelhanca(resultado)

                print("\n--- RESULTADO FINAL ---")
                for item in resultado:
                    print(f"[{item['inicio']}s - {item['fim']}s] canal={item['canal']} "
                          f"{item['falante_global']} (local: {item['falante_local']}): {item['texto_corrigido']}")

                return {
                    "segmentos": resultado,
                    "qualidade": [relatorio_qualidade_mono],
                    "informacoes_audio": informacoes_conversao,
                    "musica": _resumo_musica(info_musica),
                }

        # Caso o arquivo não seja .mp4/.mp3 (ex: .wav puro) — fluxo mínimo direto.
        print(f"[7/10] Identificando falantes (diarização) e detectando intervalos de fala...")
        segmentos_falantes = _diarizar_audio(audio_path, rotulo="mono")
        intervalos_fala_wav, sr_wav = _detectar_intervalos_fala(audio_path, top_db)
        print(f"[7/10] {len(segmentos_falantes)} segmentos de fala identificados")

        print(f"[8/10] Transcrevendo com timestamps por palavra (Groq)...")
        palavras_transcricao, segmentos_whisper = _transcrever_com_timestamps(audio_path)
        palavras_transcricao = _filtrar_falas_fantasma(
            palavras_transcricao, segmentos_whisper, intervalos_fala_wav, sr_wav
        )

        resultado = _combinar_diarizacao_transcricao(segmentos_falantes, palavras_transcricao)

        print(f"[9/10] Corrigindo pontuação, capitalização e termos técnicos (LLM)...")
        resultado = _corrigir_blocos(resultado)

        resultado = _atribuir_falantes_globais(resultado)
        resultado = _unificar_falantes_por_semelhanca(resultado)

        print("\n--- RESULTADO FINAL ---")
        for item in resultado:
            print(f"[{item['inicio']}s - {item['fim']}s] canal={item['canal']} "
                  f"{item['falante_global']} (local: {item['falante_local']}): {item['texto_corrigido']}")

        return {"segmentos": resultado, "qualidade": [], "informacoes_audio": None}

    finally:
        os.remove(tmp_path)
        if wav_lossless_path and os.path.exists(wav_lossless_path):
            os.remove(wav_lossless_path)
        if normalizado_path and os.path.exists(normalizado_path):
            os.remove(normalizado_path)
        if sem_silencio_path and os.path.exists(sem_silencio_path):
            os.remove(sem_silencio_path)
        if canal_esq_path and os.path.exists(canal_esq_path):
            os.remove(canal_esq_path)
        if canal_dir_path and os.path.exists(canal_dir_path):
            os.remove(canal_dir_path)
        if normalizado_esq_path and os.path.exists(normalizado_esq_path):
            os.remove(normalizado_esq_path)
        if normalizado_dir_path and os.path.exists(normalizado_dir_path):
            os.remove(normalizado_dir_path)
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
        return jsonify(resultado)  # já vem como {"segmentos": [...], "qualidade": [...]}
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:  # [NOVO] erros inesperados viram JSON em vez de página HTML de erro
        print(f"ERRO inesperado em /transcrever: {e}", flush=True)
        return jsonify({"erro": f"Falha interna ao transcrever: {e}"}), 500
    finally:
        os.remove(tmp_path)


if __name__ == '__main__':
    TESTE_LOCAL = True

    if TESTE_LOCAL:
        diretorio_script = os.path.dirname(os.path.abspath(__file__))
        pasta_testes = os.path.join(os.path.dirname(diretorio_script), "Teste Video")
        arquivos_teste = ["Teste1.mp4"]

        print(f"\n[DIAGNÓSTICO] Perfil de áudio: {PERFIL_AUDIO} "
              f"(forçar mono: {FORCAR_MONO}, demucs: {ISOLAR_VOCAIS_DEMUCS})")
        print(f"[DIAGNÓSTICO] Diretório do script: {diretorio_script}")
        print(f"[DIAGNÓSTICO] Pasta de testes esperada: {pasta_testes}")
        print(f"[DIAGNÓSTICO] Pasta existe? {os.path.isdir(pasta_testes)}")
        if os.path.isdir(pasta_testes):
            print(f"[DIAGNÓSTICO] Conteúdo da pasta: {os.listdir(pasta_testes)}")
        else:
            print(f"[DIAGNÓSTICO] Conteúdo do diretório do script: {os.listdir(diretorio_script)}")

        resultados = {}
        for nome_arquivo in arquivos_teste:
            caminho = os.path.join(pasta_testes, nome_arquivo)
            print(f"\n{'=' * 70}")
            print(f"  RODANDO PIPELINE: {nome_arquivo}")
            print(f"{'=' * 70}")
            try:
                resultados[nome_arquivo] = transcrever_arquivo(caminho)
            except Exception as e:
                print(f"  ERRO ao processar {nome_arquivo}: {e}")
                resultados[nome_arquivo] = None

        print(f"\n{'=' * 70}")
        print("  RESUMO DOS TESTES")
        print(f"{'=' * 70}")
        for nome_arquivo, resultado in resultados.items():
            if resultado is not None:
                total_alertas = sum(len(q.get('alertas', [])) for q in resultado.get('qualidade', []))
                status = f"OK ({len(resultado['segmentos'])} blocos, {total_alertas} alerta(s) de qualidade)"
            else:
                status = "FALHOU"
            print(f"  {nome_arquivo}: {status}")
    else:
        app.run(debug=True)