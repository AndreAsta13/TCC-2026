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

async function transcreverComGroq(arquivo) {
  const formData = new FormData();
  formData.append('file', arquivo);

  try {
    setStatus('Processando...', 'processando');
    transcriptEl.textContent = 'Aguarde...';

    const response = await fetch('http://localhost:5000/transcrever', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.erro) {
      setStatus(data.erro, 'erro');
      transcriptEl.textContent = '';
      return;
    }

    transcriptEl.textContent = data.texto.trim();
    setStatus('Transcrição concluída!', 'sucesso');

  } catch (err) {
    console.error('Erro:', err);
    setStatus('Erro ao transcrever: ' + err.message, 'erro');
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

  transcreverComGroq(arquivo);
}

function abrirYoutube() {
  const url = prompt('Cole o link do YouTube:');
  if (url) alert('Funcionalidade em desenvolvimento.');
}

document.getElementById('startBtn').addEventListener('click', () => {
  alert('Microfone em desenvolvimento.');
});