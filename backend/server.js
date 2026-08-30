require("dotenv").config();

const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const bcrypt = require("bcrypt");
//const jwt = require("jsonwebtoken");

const app = express();

app.use(express.json());
app.use(cors());

console.log("DATABASE_URL carregada:", !!process.env.DATABASE_URL);

const pool = new Pool({
    connectionString: process.env.DATABASE_URL
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

// SALVAR METADADOS DO USUÁRIO
app.post("/metadados", autenticarToken, async (req, res) => {

    const { titulo, descricao } = req.body;

    const usuarioId = req.usuario.userId;

    try {

        const resultado = await pool.query(
            `INSERT INTO metadados
            (usuario_id, titulo, descricao)
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
// BUSCAR METADADOS DO USUÁRIO LOGADO
app.get("/metadados", autenticarToken, async (req, res) => {

    const usuarioId = req.usuario.userId;

    try {

        const resultado = await pool.query(
            `SELECT *
             FROM metadados
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
// DADOS DO USUÁRIO LOGADO
app.get("/me", autenticarToken, async (req, res) => {

    const usuarioId = req.usuario.userId;

    try {

        const resultado = await pool.query(
            `SELECT id, nome, email
             FROM usuarios
             WHERE id = $1`,
            [usuarioId]
        );

        if (resultado.rows.length === 0) {
            return res.status(404).json({
                erro: "Usuário não encontrado"
            });
        }

        res.json({
            sucesso: true,
            usuario: resultado.rows[0]
        });

    } catch (err) {

        console.error("ERRO AO BUSCAR USUÁRIO:", err);

        res.status(500).json({
            erro: "Erro interno do servidor"
        });
    }
});
// INICIAR SERVIDOR
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`Servidor rodando na porta ${PORT}`);
});