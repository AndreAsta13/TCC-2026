// Função de transcrição com Groq Whisper (large-v3-turbo)
async function transcreverComGroq(arquivo) {
  // SUA CHAVE REAL AQUI (para teste local apenas!)
  // NUNCA suba isso para GitHub/Vercel/Netlify público — use backend ou .env depois
  

  const formData = new FormData();
  formData.append('file', arquivo);
  formData.append('model', 'whisper-large-v3-turbo');
  formData.append('response_format', 'text'); // 'verbose_json' se quiser timestamps/word-level

  // Opcional: força detecção de idioma (comente se quiser auto-detect)
  // formData.append('language', 'pt');       // para português brasileiro
  // formData.append('language', 'en');       // para inglês
  // formData.append('prompt', 'Transcrição de aula acadêmica em português brasileiro.');

  try {
    const response = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`  // ← CORREÇÃO AQUI! Use a variável
      },
      body: formData
    });

    if (!response.ok) {
      let errorMsg = 'Erro na API Groq';
      try {
        const errorData = await response.json();
        errorMsg += `: ${errorData.error?.message || response.statusText}`;
      } catch {}
      throw new Error(errorMsg);
    }

    const texto = await response.text();
    document.getElementById('transcript').textContent = texto.trim(); // remove espaços extras
    document.getElementById('status').textContent = 'Transcrição concluída com sucesso!';

  } catch (err) {
    console.error('Erro na transcrição Groq:', err);
    document.getElementById('status').textContent = 'Erro ao transcrever: ' + err.message;
  }
}

// Função chamada ao selecionar arquivo (integre com sua validação anterior)
function arquivoSelecionado(event) {
  const input = event.target;
  const arquivo = input.files[0];

  if (!arquivo) return;

  // Validação simples (adicione sua versão completa se quiser)
  const extensao = arquivo.name.toLowerCase().split('.').pop();
  if (!['mp3', 'mp4', 'm4a', 'wav', 'webm'].includes(extensao)) {
    alert('Formato não suportado. Use MP3, MP4, M4A, WAV ou WebM.');
    input.value = '';
    return;
  }

  if (arquivo.size > 100 * 1024 * 1024) { // limite exemplo ~100MB (Groq aceita até ~25-100MB dependendo do plano)
    alert('Arquivo muito grande. Limite recomendado: 100MB.');
    input.value = '';
    return;
  }

  document.getElementById('status').textContent = `Transcrevendo ${arquivo.name} com Groq Whisper... (pode demorar alguns segundos)`;
  document.getElementById('transcript').textContent = 'Processando... Aguarde.';

  transcreverComGroq(arquivo);

  // Opcional: limpa o input após envio
  // input.value = '';
}
export default {
  content: ['./*.html', './js/*.js'],  // ajuste caminhos
}