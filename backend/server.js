const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const bcrypt = require("bcrypt");

const app = express();

app.use(express.json());
app.use(cors());
// CONEXÃO COM POSTGRESQL
const pool = new Pool({
    user: "postgres",
    host: "localhost",
    database: "tcc",
    password: "1234",
    port: 5432,
});


// TESTE DO BANCO
app.get("/teste-db", async (req, res) => {
    try {
        const resultado = await pool.query("SELECT NOW()");

        res.json({
            sucesso: true,
            mensagem: "PostgreSQL conectado!",
            horario: resultado.rows[0]
        });

    } catch (err) {
        console.error("ERRO NO BANCO:", err);

        res.status(500).json({
            sucesso: false,
            erro: err.message
        });
    }
});


// CADASTRO
app.post("/cadastro", async (req, res) => {
    const { nome, email, senha } = req.body;

    console.log("Recebido:", nome, email);

    try {

        const hash = await bcrypt.hash(senha, 10);

        await pool.query(
            "INSERT INTO usuarios (nome, email, senha) VALUES ($1, $2, $3)",
            [nome, email, hash]
        );

        res.json({
            sucesso: true
        });

    } catch (err) {

        console.log("ERRO REAL:", err);

        res.status(500).json({
            erro: err.message
        });
    }
});


// LOGIN
app.post("/login", async (req, res) => {
    const { email, senha } = req.body;

    try {

        const result = await pool.query(
            "SELECT * FROM usuarios WHERE email = $1",
            [email]
        );

        if (result.rows.length === 0) {
            return res.json({
                erro: "Usuário não encontrado"
            });
        }

        const user = result.rows[0];

        const valido = await bcrypt.compare(
            senha,
            user.senha
        );

        if (!valido) {
            return res.json({
                erro: "Senha incorreta"
            });
        }

        res.json({
            sucesso: true,
            usuario: {
                id: user.id,
                nome: user.nome,
                email: user.email
            }
        });

    } catch (err) {

        console.error("ERRO NO LOGIN:", err);

        res.status(500).json({
            erro: "Erro interno do servidor"
        });
    }
});


// INICIAR SERVIDOR
app.listen(3000, () => {
    console.log("Servidor rodando em http://localhost:3000");
});