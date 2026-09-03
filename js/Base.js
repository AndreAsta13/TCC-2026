const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');

function setStatus(texto, tipo) {
  statusEl.textContent = texto;
  const classes = {
    aguardando: 'text-sm px-3 py-1 rounded-full bg-gray-100 text-gray-600',
    processando: 'text-sm px-3 py-1 rounded-full bg-yellow-100 text-yellow-700',
    sucesso:     'text-sm px-3 py-1 rounded-full bg-green-100 text-green-700',
    erro:        'text-sm px-3 py-1 rounded-full bg-red-100 text-red-700',
  };
  statusEl.className = classes[tipo] || classes.aguardando;
}

async function enviarArquivoNeon(arquivo) {
  const token = localStorage.getItem("token");

  if (!token) {
    window.location.href = "cadastro/login.html";
    return;
  }

  const formData = new FormData();
  formData.append("file", arquivo);

  try {
    setStatus("Enviando...", "processando");
    transcriptEl.textContent = "Enviando arquivo...";

    const response = await fetch("http://localhost:3000/arquivos/upload", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`
      },
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.sucesso) {
      throw new Error(data.erro || "Erro no upload.");
    }

    setStatus("Upload concluído!", "sucesso");
    transcriptEl.textContent =
      "Arquivo enviado com sucesso.\n\n" +
      data.arquivo.nome_original;
  } catch (err) {
    console.error("Erro:", err);
    setStatus("Erro no upload", "erro");
    transcriptEl.textContent = err.message;
  }
}
function arquivoSelecionado(event) {
  const input = event.target;
  const arquivo = input.files[0];

  if (!arquivo) return;

  const extensao = arquivo.name.toLowerCase().split('.').pop();
  if (!['mp3', 'mp4', 'm4a', 'wav', 'webm'].includes(extensao)) {
    alert('Formato não suportado. Use MP3, MP4, M4A, WAV ou WebM.');
    input.value = '';
    return;
  }

  if (arquivo.size > 100 * 1024 * 1024) {
    alert('Arquivo muito grande. Limite máximo: 100MB.');
    input.value = '';
    return;
  }

  enviarArquivoNeon(arquivo);
}

function abrirYoutube() {
  const url = prompt('Cole o link do YouTube:');
  if (url) alert('Necessário entrar  na conta.');
}

// ===== MICROFONE =====

let reconhecendo = false;
let textoFinal = '';

const SpeechRecognition =
window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {

const recognition = new SpeechRecognition();

recognition.lang = "pt-BR";
recognition.continuous = true;
recognition.interimResults = true;

recognition.onstart = () => {

setStatus("Ouvindo...", "processando");

document.getElementById("micIcon").style.animation = "pulse 1s infinite";

};

recognition.onresult = (event) => {

let textoTemp = "";

for (let i = event.resultIndex; i < event.results.length; i++) {

const transcript = event.results[i][0].transcript;

if (event.results[i].isFinal) {

textoFinal += transcript + " ";

} else {

textoTemp += transcript;

}

}

transcriptEl.textContent = corrigirTexto(textoFinal + textoTemp);

};

recognition.onend = () => {

setStatus("Microfone parado", "aguardando");

document.getElementById("micIcon").style.animation = "none";

reconhecendo = false;

};

document.getElementById("startBtn").onclick = () => {

if (!reconhecendo) {

recognition.start();

reconhecendo = true;

}

};

document.getElementById("stopBtn").onclick = () => {

recognition.stop();

};

}


// ===== DOWNLOAD TRANSCRIÇÃO =====

document.getElementById("downloadBtn").onclick = () => {

const texto = transcriptEl.textContent;

const blob = new Blob([texto], { type: "text/plain" });

const link = document.createElement("a");

link.href = URL.createObjectURL(blob);

link.download = "transcricao.txt";

link.click();

};

function corrigirTexto(texto){

return texto
.replace(/\s+/g," ")
.replace(/\bvc\b/g,"você")
.replace(/\btd\b/g,"tudo")
.replace(/\bq\b/g,"que")
.replace(/\bblz\b/g,"beleza")

}
function verificarLogin(event) {
  if (!usuarioLogado) { // substitua pela sua verificação real
    event.preventDefault();
    alert("⚠️ Você precisa estar logado para enviar arquivos.");
  }
}