require("dotenv").config();

const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const multer = require("multer");

const app = express();

// O limite padrão do express.json() é 100kb. Uma transcrição de 40 min (texto +
// segmentos + palavras de fronteira) passa disso fácil e o POST /transcricoes
// falharia com 413 antes de chegar na rota. Por isso o limite global é 25mb.
app.use(express.json({ limit: "25mb" }));
app.use(cors());

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

const JWT_SECRET = process.env.JWT_SECRET || "chave_secreta_padrao";

// Mesmo limite do Flask (500 MB) e do front (TAMANHO_MAXIMO_MB em ground.js).
// ATENÇÃO: o arquivo é guardado como BYTEA no Neon e fica inteiro na memória
// durante o upload; arquivos muito grandes consomem RAM e o espaço do plano.
const TAMANHO_MAXIMO_MB = 500;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: TAMANHO_MAXIMO_MB * 1024 * 1024 }
});

function autenticarToken(req, res, next) {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ erro: "Acesso negado. Token não fornecido." });
  }

  jwt.verify(token, JWT_SECRET, (err, usuario) => {
    if (err) {
      return res.status(403).json({ erro: "Token inválido ou expirado." });
    }

    req.usuario = usuario;
    next();
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   TABELAS
   ══════════════════════════════════════════════════════════════════════════ */
async function prepararTabelaUploads() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS uploads_usuario (
      id SERIAL PRIMARY KEY,
      usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
      nome_arquivo VARCHAR(255) NOT NULL,
      tipo_mime VARCHAR(150),
      tamanho BIGINT NOT NULL,
      arquivo BYTEA NOT NULL,
      criado_em TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_uploads_usuario
    ON uploads_usuario(usuario_id)
  `);
}

prepararTabelaUploads().catch(err => {
  console.error("ERRO AO PREPARAR UPLOADS:", err);
});

async function prepararTabelaTranscricoes() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS transcricoes (
      id                 BIGSERIAL PRIMARY KEY,
      usuario_id         TEXT        NOT NULL,
      client_id          TEXT        NOT NULL,
      origem             TEXT        NOT NULL DEFAULT 'arquivo',
      titulo             TEXT,
      nome_arquivo       TEXT,
      texto              TEXT        NOT NULL DEFAULT '',
      hipotese           TEXT        NOT NULL DEFAULT '',
      segmentos          JSONB       NOT NULL DEFAULT '[]'::jsonb,
      palavras_fronteira JSONB       NOT NULL DEFAULT '[]'::jsonb,
      metadados          JSONB       NOT NULL DEFAULT '{}'::jsonb,
      referencia         TEXT,
      metricas           JSONB,
      criado_em          TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `);

  await pool.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS transcricoes_usuario_client_idx
    ON transcricoes (usuario_id, client_id)
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS transcricoes_usuario_criado_idx
    ON transcricoes (usuario_id, criado_em DESC, id DESC)
  `);
}

prepararTabelaTranscricoes().catch(err => {
  console.error("ERRO AO PREPARAR TRANSCRIÇÕES:", err);
});

/* ══════════════════════════════════════════════════════════════════════════
   TESTE / CADASTRO / LOGIN / PERFIL
   ══════════════════════════════════════════════════════════════════════════ */
app.get("/teste-db", async (req, res) => {
  try {
    const resultado = await pool.query("SELECT NOW()");

    res.json({
      sucesso: true,
      mensagem: "PostgreSQL conectado!",
      horario: resultado.rows[0]
    });
  } catch (err) {
    res.status(500).json({
      sucesso: false,
      erro: err.message
    });
  }
});

app.post("/cadastro", async (req, res) => {
  const { nome, email, senha } = req.body;

  try {
    const hash = await bcrypt.hash(senha, 10);

    await pool.query(
      "INSERT INTO usuarios (nome, email, senha) VALUES ($1, $2, $3)",
      [nome, email, hash]
    );

    res.json({ sucesso: true });
  } catch (err) {
    console.error("ERRO NO CADASTRO:", err);
    res.status(500).json({ erro: "Erro ao cadastrar usuário." });
  }
});

app.post("/login", async (req, res) => {
  const { email, senha } = req.body;

  try {
    const result = await pool.query(
      "SELECT * FROM usuarios WHERE email = $1",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(400).json({ erro: "Usuário não encontrado" });
    }

    const user = result.rows[0];
    const valido = await bcrypt.compare(senha, user.senha);

    if (!valido) {
      return res.status(400).json({ erro: "Senha incorreta" });
    }

    const token = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: "8h" }
    );

    res.json({
      sucesso: true,
      token,
      usuario: {
        id: user.id,
        nome: user.nome,
        email: user.email
      }
    });
  } catch (err) {
    console.error("ERRO NO LOGIN:", err);
    res.status(500).json({ erro: "Erro interno do servidor" });
  }
});

app.get("/me", autenticarToken, async (req, res) => {
  const usuarioId = req.usuario.userId;

  try {
    const resultado = await pool.query(
      "SELECT id, nome, email FROM usuarios WHERE id = $1",
      [usuarioId]
    );

    if (resultado.rows.length === 0) {
      return res.status(404).json({ erro: "Usuário não encontrado" });
    }

    res.json({
      sucesso: true,
      usuario: resultado.rows[0]
    });
  } catch (err) {
    console.error("ERRO AO BUSCAR USUÁRIO:", err);
    res.status(500).json({ erro: "Erro interno do servidor" });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   ARQUIVOS (upload do áudio/vídeo na conta do usuário)
   ══════════════════════════════════════════════════════════════════════════ */
app.post(
  "/arquivos/upload",
  autenticarToken,
  upload.single("file"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ erro: "Nenhum arquivo enviado." });
    }

    const extensao = req.file.originalname.toLowerCase().split(".").pop();

    if (!["mp3", "mp4", "m4a", "wav", "webm"].includes(extensao)) {
      return res.status(400).json({ erro: "Formato não suportado." });
    }

    try {
      const resultado = await pool.query(
        `INSERT INTO uploads_usuario
          (usuario_id, nome_arquivo, tipo_mime, tamanho, arquivo)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id, nome_arquivo, tipo_mime, tamanho, criado_em`,
        [
          req.usuario.userId,
          req.file.originalname,
          req.file.mimetype,
          req.file.size,
          req.file.buffer
        ]
      );

      res.status(201).json({
        sucesso: true,
        arquivo: {
          id: resultado.rows[0].id,
          nome_original: resultado.rows[0].nome_arquivo,
          tipo_mime: resultado.rows[0].tipo_mime,
          tamanho: resultado.rows[0].tamanho,
          criado_em: resultado.rows[0].criado_em
        }
      });
    } catch (err) {
      console.error("ERRO NO UPLOAD:", err);
      res.status(500).json({ erro: "Erro ao salvar arquivo no banco." });
    }
  }
);

app.get("/arquivos", autenticarToken, async (req, res) => {
  try {
    const resultado = await pool.query(
      `SELECT id, nome_arquivo, tipo_mime, tamanho, criado_em
       FROM uploads_usuario
       WHERE usuario_id = $1
       ORDER BY criado_em DESC`,
      [req.usuario.userId]
    );

    res.json({
      sucesso: true,
      arquivos: resultado.rows
    });
  } catch (err) {
    console.error("ERRO AO LISTAR ARQUIVOS:", err);
    res.status(500).json({ erro: "Erro ao listar arquivos." });
  }
});

app.get("/arquivos/:id", autenticarToken, async (req, res) => {
  try {
    const resultado = await pool.query(
      `SELECT nome_arquivo, tipo_mime, arquivo
       FROM uploads_usuario
       WHERE id = $1 AND usuario_id = $2`,
      [req.params.id, req.usuario.userId]
    );

    if (resultado.rows.length === 0) {
      return res.status(404).json({ erro: "Arquivo não encontrado." });
    }

    const item = resultado.rows[0];

    res.setHeader("Content-Type", item.tipo_mime || "application/octet-stream");
    res.setHeader(
      "Content-Disposition",
      `inline; filename*=UTF-8''${encodeURIComponent(item.nome_arquivo)}`
    );

    res.send(item.arquivo);
  } catch (err) {
    console.error("ERRO AO BUSCAR ARQUIVO:", err);
    res.status(500).json({ erro: "Erro ao buscar arquivo." });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   METADADOS
   ══════════════════════════════════════════════════════════════════════════ */
app.post("/metadados", autenticarToken, async (req, res) => {
  const { titulo, descricao } = req.body;
  const usuarioId = req.usuario.userId;

  try {
    const resultado = await pool.query(
      `INSERT INTO metadados (usuario_id, titulo, descricao)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [usuarioId, titulo, descricao]
    );

    res.json({
      sucesso: true,
      metadado: resultado.rows[0]
    });
  } catch (err) {
    console.error("ERRO AO SALVAR METADADOS:", err);
    res.status(500).json({
      sucesso: false,
      erro: "Erro ao salvar os dados"
    });
  }
});

app.get("/metadados", autenticarToken, async (req, res) => {
  const usuarioId = req.usuario.userId;

  try {
    const resultado = await pool.query(
      `SELECT * FROM metadados
       WHERE usuario_id = $1
       ORDER BY criado_em DESC`,
      [usuarioId]
    );

    res.json({
      sucesso: true,
      metadados: resultado.rows
    });
  } catch (err) {
    console.error("ERRO AO BUSCAR METADADOS:", err);
    res.status(500).json({
      sucesso: false,
      erro: "Erro ao buscar os dados"
    });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   HISTÓRICO DE TRANSCRIÇÕES (antes ficava em transcricoes.js)
   ══════════════════════════════════════════════════════════════════════════ */

// LISTAR — pilha LIFO: a mais recente primeiro
app.get("/transcricoes", autenticarToken, async (req, res) => {
  try {
    const { rows } = await pool.query(
      `SELECT id, client_id, origem, titulo, nome_arquivo, texto, hipotese,
              segmentos, palavras_fronteira, metadados, referencia, metricas, criado_em
         FROM transcricoes
        WHERE usuario_id = $1
        ORDER BY criado_em DESC, id DESC`,
      [String(req.usuario.userId)]
    );

    res.json({ sucesso: true, transcricoes: rows });
  } catch (err) {
    console.error("ERRO AO LISTAR TRANSCRIÇÕES:", err);
    res.status(500).json({ sucesso: false, erro: "Erro ao listar transcrições." });
  }
});

// EMPILHAR — sempre insere uma NOVA linha (nunca substitui)
app.post("/transcricoes", autenticarToken, async (req, res) => {
  const b = req.body || {};

  if (!b.client_id || (!b.texto && !b.hipotese)) {
    return res.status(400).json({ sucesso: false, erro: "client_id e texto são obrigatórios." });
  }

  try {
    const { rows } = await pool.query(
      `INSERT INTO transcricoes
         (usuario_id, client_id, origem, titulo, nome_arquivo, texto, hipotese,
          segmentos, palavras_fronteira, metadados)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb)
       ON CONFLICT (usuario_id, client_id) DO NOTHING
       RETURNING id, client_id, criado_em`,
      [
        String(req.usuario.userId),
        String(b.client_id),
        b.origem || "arquivo",
        b.titulo || null,
        b.nome_arquivo || null,
        b.texto || "",
        b.hipotese || "",
        JSON.stringify(b.segmentos || []),
        JSON.stringify(b.palavras_fronteira || []),
        JSON.stringify(b.metadados || {})
      ]
    );

    // rows vazio = já existia (reenvio de pendente): continua sendo sucesso
    res.status(201).json({ sucesso: true, transcricao: rows[0] || null });
  } catch (err) {
    console.error("ERRO AO SALVAR TRANSCRIÇÃO:", err);
    res.status(500).json({ sucesso: false, erro: "Erro ao salvar transcrição." });
  }
});

// Guarda o texto de referência (ground truth) e as métricas WER/CER calculadas
app.patch("/transcricoes/:id/referencia", autenticarToken, async (req, res) => {
  const { referencia, metricas } = req.body || {};

  if (!referencia || !metricas) {
    return res.status(400).json({ sucesso: false, erro: "Envie referencia e metricas." });
  }

  // id é BIGSERIAL: um valor não numérico faria o Postgres lançar erro (500)
  if (!/^\d+$/.test(req.params.id)) {
    return res.status(400).json({ sucesso: false, erro: "ID inválido." });
  }

  try {
    const { rowCount } = await pool.query(
      `UPDATE transcricoes
          SET referencia = $1, metricas = $2::jsonb
        WHERE id = $3 AND usuario_id = $4`,
      [referencia, JSON.stringify(metricas), req.params.id, String(req.usuario.userId)]
    );

    if (!rowCount) {
      return res.status(404).json({ sucesso: false, erro: "Transcrição não encontrada." });
    }

    res.json({ sucesso: true });
  } catch (err) {
    console.error("ERRO AO SALVAR MÉTRICAS:", err);
    res.status(500).json({ sucesso: false, erro: "Erro ao salvar métricas." });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   GROUND TRUTH (resumo/lista)
   ══════════════════════════════════════════════════════════════════════════ */
app.get("/gt/listar", autenticarToken, async (req, res) => {
  const usuarioId = req.usuario.userId;

  try {
    const resultado = await pool.query(
      "SELECT * FROM metadados WHERE usuario_id = $1 ORDER BY criado_em DESC",
      [usuarioId]
    );

    res.json(resultado.rows);
  } catch (err) {
    console.error("ERRO EM /gt/listar:", err);
    res.status(500).json({ erro: "Erro ao listar dados." });
  }
});

app.get("/gt/resumo", autenticarToken, async (req, res) => {
  const usuarioId = req.usuario.userId;

  try {
    const resultado = await pool.query(
      "SELECT COUNT(*) AS total FROM metadados WHERE usuario_id = $1",
      [usuarioId]
    );

    res.json({
      total: parseInt(resultado.rows[0].total)
    });
  } catch (err) {
    console.error("ERRO EM /gt/resumo:", err);
    res.status(500).json({ erro: "Erro ao gerar resumo." });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   TRATAMENTO DE ERROS — SEMPRE depois de todas as rotas
   ══════════════════════════════════════════════════════════════════════════ */
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      erro: `Arquivo muito grande. Limite máximo: ${TAMANHO_MAXIMO_MB}MB.`
    });
  }

  // JSON maior que o limite do express.json
  if (err && err.type === "entity.too.large") {
    return res.status(413).json({ erro: "Conteúdo muito grande para ser salvo." });
  }

  // JSON malformado
  if (err && err.type === "entity.parse.failed") {
    return res.status(400).json({ erro: "JSON inválido." });
  }

  if (err) {
    console.error(err);
    return res.status(500).json({ erro: "Erro interno do servidor." });
  }

  next();
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Servidor rodando na porta ${PORT}`);
});