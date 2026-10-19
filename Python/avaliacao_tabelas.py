"""
avaliacao_tabelas.py - rota GET /avaliacao/tabelas (usada pela página Histórico).

Lê os CSVs gerados por avaliar_corpus.py e avaliar_diarizacao.py e devolve em JSON.
Coloque este arquivo na MESMA pasta do app.py e registre o blueprint (2 linhas no app.py):

    from avaliacao_tabelas import avaliacao_bp                  # junto dos outros imports
    app.register_blueprint(avaliacao_bp)                        # logo depois de app = Flask(...) / CORS(app)

Pasta dos CSVs: variável AVALIACAO_DIR do .env; se não existir, usa ./saida ao lado da pasta do app
(a mesma pasta "saida" que os scripts de avaliação criam quando rodados a partir da raiz do projeto).
"""
import csv
import os

from flask import Blueprint, jsonify

avaliacao_bp = Blueprint("avaliacao", __name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVOS_AVALIACAO = {
    "tabela3": "tabela3_por_trecho.csv",
    "tabela4": "tabela4_pareada.csv",
    "tabela5": "tabela5_diarizacao.csv",
}


def _pasta_avaliacao():
    # Lido a cada requisição: o .env já foi carregado pelo app.py quando a rota roda.
    return os.getenv("AVALIACAO_DIR") or os.path.join(_BASE_DIR, "saida")


@avaliacao_bp.route("/avaliacao/tabelas", methods=["GET"])
def tabelas_avaliacao():
    """Devolve as Tabelas 3, 4 e 5 (CSV -> JSON). None = tabela ainda não gerada."""
    pasta = _pasta_avaliacao()
    resposta = {}
    for chave, nome in ARQUIVOS_AVALIACAO.items():
        caminho = os.path.join(pasta, nome)
        if not os.path.exists(caminho):
            resposta[chave] = None
            continue
        with open(caminho, encoding="utf-8-sig", newline="") as f:
            linhas = list(csv.reader(f))
        resposta[chave] = {
            "cabecalho": linhas[0],
            "linhas": linhas[1:],
            "atualizado_em": int(os.path.getmtime(caminho) * 1000),
        } if linhas else None
    return jsonify(resposta)