const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const bcrypt = require("bcrypt");

const app = express();
app.use(express.json());
app.use(cors());

// 🔗 CONEXÃO COM POSTGRESQL
const pool = new Pool({
  user: "postgres",
  host: "localhost",
  database: "tcc",
  password: "", 
  port: 5432,
});

// 📌 CADASTRO
app.post("/cadastro", async (req, res) => {
  const { nome, email, senha } = req.body;

  console.log("Recebido:", nome, email, senha); 

  try {
    const hash = await bcrypt.hash(senha, 10);

    await pool.query(
      "INSERT INTO usuarios (nome, email, senha) VALUES ($1, $2, $3)",
      [nome, email, hash]
    );

    res.json({ sucesso: true });
  } catch (err) {
  console.log("ERRO REAL:", err); // 👈 MOSTRA O ERRO VERDADEIRO
  res.json({ erro: err.message }); // 👈 MOSTRA NO FRONT
}
});

// 📌 LOGIN
app.post("/login", async (req, res) => {
  const { email, senha } = req.body;

  const result = await pool.query(
    "SELECT * FROM usuarios WHERE email = $1",
    [email]
  );

  if (result.rows.length === 0) {
    return res.json({ erro: "Usuário não encontrado" });
  }

  const user = result.rows[0];

  const valido = await bcrypt.compare(senha, user.senha);

  if (!valido) {
    return res.json({ erro: "Senha incorreta" });
  }

  res.json({ sucesso: true });
});

// 🚀 INICIAR SERVIDOR
app.listen(3000, () => {
  console.log("Servidor rodando em http://localhost:3000");
});