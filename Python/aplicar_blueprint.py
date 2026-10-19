"""
aplicar_blueprint.py - instala a rota /avaliacao/tabelas no seu app.py (com backup).

Uso (a partir da pasta onde estão estes arquivos):
    python aplicar_blueprint.py /caminho/para/app.py

O que faz:
  1. copia avaliacao_tabelas.py para a pasta do app.py (se ainda não estiver lá, ou se mudou);
  2. cria app.py.bak;
  3. insere, logo depois da linha `CORS(app)`, o import e o register_blueprint (só uma vez).
Rodar de novo é seguro: se já estiver aplicado, não altera nada.
"""
import os
import re
import shutil
import sys

TRECHO = (
    "# [AVALIAÇÃO] rota usada pela página Histórico (Tabelas 3, 4 e 5)",
    "from avaliacao_tabelas import avaliacao_bp",
    "app.register_blueprint(avaliacao_bp)",
)


def main():
    if len(sys.argv) != 2:
        sys.exit("Uso: python aplicar_blueprint.py /caminho/para/app.py")
    app_py = os.path.abspath(sys.argv[1])
    if not os.path.isfile(app_py):
        sys.exit(f"Não encontrei: {app_py}")

    aqui = os.path.dirname(os.path.abspath(__file__))
    origem = os.path.join(aqui, "avaliacao_tabelas.py")
    destino = os.path.join(os.path.dirname(app_py), "avaliacao_tabelas.py")
    mudou = (not os.path.exists(destino)) or open(origem, "rb").read() != open(destino, "rb").read()
    if mudou:
        shutil.copyfile(origem, destino)
        print(f"Copiado: {destino}")

    with open(app_py, encoding="utf-8", newline="") as f:
        src = f.read()
    if "avaliacao_bp" in src:
        print("app.py já tem o blueprint registrado. Nada a fazer.")
        return

    m = re.search(r"^CORS\(app\)[ \t]*(?=\r?$)", src, re.M)
    if not m:
        print("Não achei a linha 'CORS(app)' no app.py. Cole manualmente, depois de criar o app Flask:\n")
        print("\n".join(TRECHO))
        sys.exit(1)

    nl = "\r\n" if "\r\n" in src else "\n"
    shutil.copyfile(app_py, app_py + ".bak")
    insercao = nl + nl + nl.join(TRECHO)
    novo = src[:m.end()] + insercao + src[m.end():]
    with open(app_py, "w", encoding="utf-8", newline="") as f:
        f.write(novo)
    print(f"Blueprint registrado em {app_py} (backup: {app_py}.bak)")


if __name__ == "__main__":
    main()