import sys

print("Importando Flask...", flush=True)
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Importando Groq...", flush=True)
from groq import Groq
from dotenv import load_dotenv

print("Importando librosa (pode demorar um pouco)...", flush=True)
import librosa

print("Importando ffmpeg...", flush=True)
import ffmpeg

print("Importando numpy/soundfile...", flush=True)
import os, tempfile, shutil, json
import numpy as np
import soundfile as sf

FFMPEG_BIN = r"C:\ffmpeg\ffmpeg-9.0.1-full_build-shared\bin"

if os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]
    print(f"FFmpeg Shared configurado: {FFMPEG_BIN}", flush=True)
else:
    print(f"AVISO: FFmpeg Shared não encontrado: {FFMPEG_BIN}", flush=True)

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

print(f"Carregando .env: {ENV_PATH}", flush=True)

load_dotenv(ENV_PATH, override=True)

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
LIMITE_SEGUNDOS = 3600
HF_TOKEN = os.getenv("HF_TOKEN")

_diarization_pipeline = None


# ---------------------------------------------------------------------------
# DETECÇÃO DE CANAIS (mono / estéreo / multicanal)
# ---------------------------------------------------------------------------
def _detectar_canais(caminho_entrada):
    """
    Inspeciona o arquivo via ffprobe (sem decodificar o áudio de verdade)
    e retorna um dicionário com o número de canais, layout e taxa de
    amostragem original da trilha de áudio.
    """
    probe = ffmpeg.probe(caminho_entrada)
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

    if not audio_streams:
        raise ValueError("Nenhuma trilha de áudio encontrada no arquivo.")

    canais = audio_streams[0].get('channels', 1)
    layout = audio_streams[0].get('channel_layout', 'desconhecido')
    taxa_original = audio_streams[0].get('sample_rate', 'desconhecida')
    tipo_audio = "mono" if canais == 1 else ("estéreo" if canais == 2 else f"multicanal ({canais} canais)")

    return {
        "canais": canais,
        "layout": layout,
        "taxa_original": taxa_original,
        "tipo_audio": tipo_audio
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
# NORMALIZAÇÃO EBU R128 — MEDIÇÃO (usada no two-pass do caminho multicanal)
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
# ENQUADRAMENTO + REMOÇÃO DE SILÊNCIO
# (roda sobre o áudio já convertido em WAV E normalizado)
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
            print(f"[3/10] Áudio detectado: {info_canais['tipo_audio']} "
                  f"(layout: {info_canais['layout']}, taxa: {info_canais['taxa_original']} Hz)")

            # --- Conversão lossless para WAV ---
            print(f"[4/10] Convertendo {extensao} para WAV sem perda de dados (PCM 16-bit)...")
            wav_lossless_path = tmp_path.rsplit('.', 1)[0] + '_lossless.wav'
            _converter_para_wav_lossless(tmp_path, wav_lossless_path, canais)
            print(f"[4/10] WAV lossless gerado: {wav_lossless_path}")

            # --- Normalização EBU R128 (diferenciada por canal) ---
            normalizado_path = tmp_path.rsplit('.', 1)[0] + '_normalizado.wav'

            if canais == 1:
                print(f"[5/10] Normalizando loudness EBU R128 (single-pass)...")
                (
                    ffmpeg.input(wav_lossless_path)
                    .output(normalizado_path, vn=None, acodec='pcm_s16le', ac=1,
                            af='loudnorm=I=-23:LRA=7:TP=-2')
                    .run(quiet=True, overwrite_output=True)
                )
            else:
                print(f"[5a/10] Medindo loudness (passada 1/2)...")
                medidas = _medir_loudness(wav_lossless_path)
                print(f"[5a/10] Medido: I={medidas['input_i']} LUFS, TP={medidas['input_tp']} dBTP")

                print(f"[5b/10] Reamostrando 16kHz + normalização precisa (passada 2/2)...")
                filtro_preciso = (
                    f"loudnorm=I=-23:LRA=7:TP=-2:"
                    f"measured_I={medidas['input_i']}:"
                    f"measured_LRA={medidas['input_lra']}:"
                    f"measured_TP={medidas['input_tp']}:"
                    f"measured_thresh={medidas['input_thresh']}:"
                    f"offset={medidas['target_offset']}:linear=true"
                )
                (
                    ffmpeg.input(wav_lossless_path)
                    .output(normalizado_path, vn=None, acodec='pcm_s16le', ac=canais,
                            ar=16000, af=filtro_preciso)
                    .run(quiet=True, overwrite_output=True)
                )

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

    finally:
        os.remove(tmp_path)
        if wav_lossless_path and os.path.exists(wav_lossless_path):
            os.remove(wav_lossless_path)
        if normalizado_path and os.path.exists(normalizado_path):
            os.remove(normalizado_path)
        if sem_silencio_path and os.path.exists(sem_silencio_path):
            os.remove(sem_silencio_path)


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
        pasta_testes = os.path.join(os.path.dirname(diretorio_script), "Teste Video")
        arquivos_teste = ["Teste1.mp4"]

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