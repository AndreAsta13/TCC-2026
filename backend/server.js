require("dotenv").config();

const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const multer = require("multer");

const app = express();

app.use(express.json());
app.use(cors());

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

const JWT_SECRET = process.env.JWT_SECRET || "chave_secreta_padrao";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }
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

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      erro: "Arquivo muito grande. Limite máximo: 100MB."
    });
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
