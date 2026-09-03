const API = 'http://localhost:3000';

/* ══════════════════════════════════════════════════════════════════════════
   Variáveis globais que serão preenchidas quando o DOM estiver pronto
   (mantidas fora do DOMContentLoaded para que as funções abaixo, chamadas
   por onclick="" no HTML, consigam acessá-las)
   ══════════════════════════════════════════════════════════════════════════ */
let statusEl, transcriptEl, startBtn, stopBtn, micIcon, downloadBtn;
let menuDropdown, menuBtn, dz, gtTextarea;
let datasetCache = [];
let arquivoGtSelecionado = null;
let usuarioLogado = false; // troque por uma checagem real de sessão
let reconhecendo = false;
let textoFinal = '';

/* ══ Tabs ══════════════════════════════════════════════════════════════════ */
function showTab(id, btn) {
  const content = document.getElementById('tab-' + id);
  if (!content) return;
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  content.classList.add('active');
  btn.classList.add('active');
  if (id === 'wer') renderWerChart();
}

/* ══ Toast ══════════════════════════════════════════════════════════════════ */
function toast(msg, tipo = 'success') {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.className = 'show ' + tipo;
  setTimeout(() => { t.className = ''; }, 3200);
}

/* ══ Cores WER ══════════════════════════════════════════════════════════════ */
function werColor(v) {
  if (v === null || v === undefined) return 'var(--gray-400)';
  if (v < 7)  return 'var(--emerald)';
  if (v < 12) return 'var(--amber)';
  return 'var(--red)';
}
function statusBadge(s) {
  const map = { validado: 'badge-green', revisão: 'badge-amber', pendente: 'badge-indigo' };
  return `<span class="badge ${map[s] || 'badge-indigo'}">${s}</span>`;
}
function storageBadge(s) {
  const icons = { local:'💾', s3:'🪣', gcs:'☁️' };
  return `<span class="source-chip">${icons[s] || ''} ${s || 'local'}</span>`;
}

/* ══ Carregar dados ════════════════════════════════════════════════════════ */
async function carregarDados() {
  const token = localStorage.getItem("token");

  if (!token) {
    console.error("Token de autenticação não encontrado.");
    return;
  }

  try {
    // 1. Busca o resumo incluindo o token
    const resResumo = await fetch("http://localhost:3000/gt/resumo", {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${token}`
      }
    });

    // 2. Busca a lista incluindo o token
    const resListar = await fetch("http://localhost:3000/gt/listar", {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${token}`
      }
    });

    if (resResumo.ok && resListar.ok) {
      const resumo = await resResumo.json();
      const lista = await resListar.json();

      // Continue o seu código utilizando 'resumo' e 'lista'
      console.log("Resumo:", resumo);
      console.log("Lista:", lista);
    } else {
      console.error("Erro na resposta do servidor:", resResumo.status, resListar.status);
    }
  } catch (erro) {
    console.error("Erro ao carregar dados do Ground Truth:", erro);
  }
}

function usarDadosDemo() {
  datasetCache = [
    { id:'a1', filename:'entrevista_cv_01.mp3',  fonte:'Common Voice',      palavras_gt:312, wer:5.2,  cer:2.1, storage:'s3',   status:'validado' },
    { id:'a2', filename:'palestra_ia_02.wav',     fonte:'YouTube',           palavras_gt:604, wer:11.7, cer:5.3, storage:'s3',   status:'validado' },
    { id:'a3', filename:'reuniao_03.m4a',         fonte:'Gravação própria',  palavras_gt:781, wer:7.9,  cer:3.4, storage:'local',status:'validado' },
    { id:'a4', filename:'aula_usp_04.mp3',        fonte:'Podcast Acadêmico', palavras_gt:556, wer:4.1,  cer:1.8, storage:'gcs',  status:'validado' },
    { id:'a5', filename:'seminario_05.wav',       fonte:'Common Voice',      palavras_gt:344, wer:9.3,  cer:4.1, storage:'s3',   status:'revisão'  },
    { id:'a6', filename:'debate_tecnico_06.mp3',  fonte:'Podcast Acadêmico', palavras_gt:499, wer:13.8, cer:6.2, storage:'local',status:'revisão'  },
    { id:'a7', filename:'conferencia_07.m4a',     fonte:'YouTube',           palavras_gt:751, wer:6.6,  cer:2.9, storage:'gcs',  status:'validado' },
  ];
  const wers = datasetCache.map(d => d.wer);
  atualizarMetricas({
    total: datasetCache.length,
    validados: 5, pendentes: 0,
    wer_medio: (wers.reduce((a,b)=>a+b,0)/wers.length).toFixed(2),
    wer_melhor: Math.min(...wers).toFixed(2),
    wer_pior:   Math.max(...wers).toFixed(2),
    palavras_total: datasetCache.reduce((a,d)=>a+d.palavras_gt,0),
    storage_backend: 'demo',
  });
  renderDataset(datasetCache);
}

function atualizarMetricas(r) {
  const setText = (id, valor) => { const el = document.getElementById(id); if (el) el.textContent = valor; };
  setText('m-total', r.total ?? '—');
  const wmEl = document.getElementById('m-wer');
  if (wmEl) {
    wmEl.textContent = r.wer_medio != null ? r.wer_medio + '%' : '—';
    wmEl.className = 'metric-value ' + (r.wer_medio < 7 ? 'green' : r.wer_medio < 12 ? 'amber' : 'red');
  }
  const wbEl = document.getElementById('m-best');
  if (wbEl) wbEl.textContent = r.wer_melhor != null ? r.wer_melhor + '%' : '—';
  const wwEl = document.getElementById('m-worst');
  if (wwEl) {
    wwEl.textContent = r.wer_pior != null ? r.wer_pior + '%' : '—';
    wwEl.className = 'metric-value ' + (r.wer_pior >= 12 ? 'red' : 'amber');
  }
  setText('m-words', (r.palavras_total ?? 0).toLocaleString('pt-BR'));
  setText('m-valid', r.validados ?? '—');

  const label = document.getElementById('storageLabel');
  const pill  = document.getElementById('storageStatus');
  const be    = r.storage_backend || 'local';
  if (label) label.textContent = be.toUpperCase();
  if (pill && be === 'demo') {
    pill.style.background = '#fef3c7'; pill.style.color = '#92400e'; pill.style.borderColor = '#fcd34d';
  }
  document.querySelectorAll('.storage-card').forEach(c => c.classList.remove('active'));
  const sc = document.getElementById('card-' + be);
  if (sc) sc.classList.add('active');
}

function renderDataset(lista) {
  const tbody = document.getElementById('datasetBody');
  if (!tbody) return;
  if (!lista.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--gray-400);padding:32px;">Nenhum arquivo no dataset ainda.</td></tr>';
    return;
  }
  const maxWer = Math.max(...lista.map(d => d.wer || 0), 1);
  tbody.innerHTML = lista.map((d, i) => `
    <tr>
      <td style="color:var(--gray-400); font-size:12px;">${i + 1}</td>
      <td style="font-weight:500; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${d.filename}">${d.filename}</td>
      <td><span class="source-chip">${d.fonte || '—'}</span></td>
      <td>${(d.palavras_gt || 0).toLocaleString('pt-BR')}</td>
      <td>
        <div class="wer-cell">
          <span style="font-weight:700; color:${werColor(d.wer)}; min-width:40px; font-size:13px;">${d.wer != null ? d.wer.toFixed(1)+'%' : '—'}</span>
          <div class="wer-bar-bg">
            <div class="wer-bar-fill" style="width:${d.wer != null ? Math.min((d.wer/maxWer)*100,100).toFixed(1) : 0}%; background:${werColor(d.wer)};"></div>
          </div>
        </div>
      </td>
      <td style="color:var(--gray-600);">${d.cer != null ? d.cer.toFixed(1)+'%' : '—'}</td>
      <td>${storageBadge(d.storage)}</td>
      <td>${statusBadge(d.status)}</td>
    </tr>
  `).join('');
}

function renderWerChart() {
  const el = document.getElementById('werChart');
  if (!el) return;
  if (!datasetCache.length) { el.innerHTML = '<p style="color:var(--gray-400);text-align:center;">Nenhum dado disponível.</p>'; return; }
  const max = Math.max(...datasetCache.map(d => d.wer || 0), 1);
  el.innerHTML = datasetCache.map(d => `
    <div class="wer-row">
      <div class="wer-filename" title="${d.filename}">${d.filename}</div>
      <div class="wer-bar-bg" style="flex:1;">
        <div class="wer-bar-fill" style="width:${d.wer != null ? ((d.wer/max)*100).toFixed(1) : 0}%; background:${werColor(d.wer)};"></div>
      </div>
      <span style="font-size:13px;font-weight:700;color:${werColor(d.wer)};min-width:44px;text-align:right;">${d.wer != null ? d.wer.toFixed(1)+'%' : '—'}</span>
      <span style="font-size:11px;color:var(--gray-400);min-width:80px;text-align:right;">CER: ${d.cer != null ? d.cer.toFixed(1)+'%' : '—'}</span>
    </div>
  `).join('');
}

function arquivoEscolhido(input) {
  if (input.files[0]) definirArquivo(input.files[0]);
}
function definirArquivo(f) {
  arquivoGtSelecionado = f;
  const dropText = document.getElementById('dropText');
  if (dropText) dropText.textContent = '✔ ' + f.name;
  if (dz) { dz.style.borderColor = 'var(--emerald)'; dz.style.background = 'var(--emerald-light)'; dz.style.color = 'var(--emerald-dark)'; }
}

async function enviarParaServidor() {
  const gtEl    = document.getElementById('gtTextarea');
  const fonteEl = document.getElementById('fonteSelect');
  const gt    = gtEl ? gtEl.value.trim() : '';
  const fonte = fonteEl ? fonteEl.value : '';

  if (!arquivoGtSelecionado) { toast('Selecione um arquivo de áudio.', 'error'); return; }
  if (!gt)    { toast('Digite a transcrição ground truth.', 'error'); return; }
  if (!fonte) { toast('Selecione a fonte do áudio.', 'error'); return; }

  const form = new FormData();
  form.append('file', arquivoGtSelecionado);
  form.append('transcricao', gt);
  form.append('fonte', fonte);

  const prog  = document.getElementById('uploadProgress');
  const pBar  = document.getElementById('progressBar');
  const pText = document.getElementById('progressText');
  if (prog) prog.style.display = 'block';
  if (pBar) pBar.style.width   = '30%';
  if (pText) pText.textContent  = 'Enviando arquivo...';

  try {
    const res = await fetch(`${API}/gt/adicionar`, { method: 'POST', body: form });
    if (pBar) pBar.style.width = '80%';
    if (pText) pText.textContent = 'Transcrevendo e calculando WER...';
    const data = await res.json();
    if (pBar) pBar.style.width = '100%';

    if (data.sucesso) {
      toast(`Adicionado! WER: ${data.entrada.wer != null ? data.entrada.wer + '%' : 'pendente'}`, 'success');
      limparFormulario();
      await carregarDados();
    } else {
      toast(data.erro || 'Erro ao adicionar.', 'error');
    }
  } catch {
    toast('Servidor offline. Verifique se ground_truth_server.py está rodando.', 'error');
  } finally {
    setTimeout(() => { if (prog) prog.style.display = 'none'; if (pBar) pBar.style.width = '0%'; }, 1200);
  }
}

function limparFormulario() {
  arquivoGtSelecionado = null;
  const gtEl2       = document.getElementById('gtTextarea');
  const fonteEl2    = document.getElementById('fonteSelect');
  const audioInput  = document.getElementById('audioInput');
  const dropText    = document.getElementById('dropText');
  const wordCount   = document.getElementById('wordCount');
  if (gtEl2) gtEl2.value = '';
  if (fonteEl2) fonteEl2.value = '';
  if (audioInput) audioInput.value = '';
  if (dropText) dropText.textContent = 'Clique ou arraste o arquivo aqui';
  if (wordCount) wordCount.textContent = '0';
  if (dz) { dz.style.borderColor = ''; dz.style.background = ''; dz.style.color = ''; }
}

function exportarDataset() {
  const blob = new Blob([JSON.stringify(datasetCache, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ground_truth_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  toast('Dataset exportado!', 'success');
}

function setStatus(texto, tipo) {
  if (!statusEl) return;
  statusEl.textContent = texto;
  const classes = {
    aguardando:  'text-sm px-3 py-1 rounded-full bg-gray-100 text-gray-600',
    processando: 'text-sm px-3 py-1 rounded-full bg-yellow-100 text-yellow-700',
    sucesso:     'text-sm px-3 py-1 rounded-full bg-green-100 text-green-700',
    erro:        'text-sm px-3 py-1 rounded-full bg-red-100 text-red-700',
  };
  statusEl.className = classes[tipo] || classes.aguardando;
}

async function enviarArquivoNeon(arquivo) {
  const token = localStorage.getItem("token");

  if (!token) {
    window.location.href = "../cadastro/login.html";
    return;
  }

  const formData = new FormData();
  formData.append("file", arquivo);

  try {
    setStatus("Enviando...", "processando");

    if (transcriptEl) {
      transcriptEl.textContent = "Enviando arquivo...";
    }

    const response = await fetch(`${API}/arquivos/upload`, {
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

    if (transcriptEl) {
      transcriptEl.textContent =
        "Arquivo enviado com sucesso.\n\n" +
        data.arquivo.nome_original;
    }

    toast(`"${data.arquivo.nome_original}" salvo na sua conta.`, "success");
  } catch (err) {
    console.error("Erro:", err);
    setStatus("Erro no upload", "erro");

    if (transcriptEl) {
      transcriptEl.textContent = err.message;
    }
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
  const url = prompt('Cole o link do vídeo do YouTube:');
  if (!url) return;
  importarYoutube(url.trim());
}

async function importarYoutube(url) {
  toast('Baixando áudio do YouTube e enviando para o Google Cloud...', 'processando');

  try {
    const res = await fetch(`${API}/gt/youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, fonte: 'YouTube' }),
    });
    const data = await res.json();

    if (!res.ok || !data.sucesso) {
      toast(data.erro || 'Erro ao importar vídeo do YouTube.', 'error');
      return;
    }

    toast(`"${data.entrada.titulo_youtube}" importado! Falta adicionar a transcrição manual.`, 'success');
    await carregarDados();
  } catch {
    toast('Servidor offline.', 'error');
  }
}

function verificarLogin(event) {
  if (!usuarioLogado) {
    event.preventDefault();
    alert('⚠️ Você precisa estar logado para enviar arquivos.');
  }
}

function corrigirTexto(texto) {
  return texto
    .replace(/\s+/g, ' ')
    .replace(/\bvc\b/g, 'você')
    .replace(/\btd\b/g, 'tudo')
    .replace(/\bq\b/g, 'que')
    .replace(/\bblz\b/g, 'beleza');
}

/* ══════════════════════════════════════════════════════════════════════════
   Tudo que DEPENDE do DOM já estar carregado (buscar elementos, ligar
   eventos) fica aqui dentro. É isso que resolve o "clico e não acontece nada".
   ══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

  /* ── referências de elementos ── */
  statusEl     = document.getElementById('status');
  transcriptEl = document.getElementById('transcript');
  startBtn     = document.getElementById('startBtn');
  stopBtn      = document.getElementById('stopBtn');
  micIcon      = document.getElementById('micIcon');
  downloadBtn  = document.getElementById('downloadBtn');
  menuDropdown = document.getElementById('menuDropdown');
  menuBtn      = document.getElementById('menuBtn');
  dz           = document.getElementById('dropZone');
  gtTextarea   = document.getElementById('gtTextarea');

  /* ── slides (se existirem nessa página) ── */
  const sections = Array.from(document.querySelectorAll('main.slides > section'));
  if (sections.length) {
    const dots    = Array.from(document.querySelectorAll('.slide-dot'));
    const prevBtn = document.getElementById('slidePrev');
    const nextBtn = document.getElementById('slideNext');
    let current = 0;
    function goTo(index) {
      index = Math.max(0, Math.min(sections.length - 1, index));
      sections.forEach((s, i) => s.classList.toggle('active', i === index));
      dots.forEach((d, i) => d.classList.toggle('active', i === index));
      if (prevBtn) prevBtn.disabled = index === 0;
      if (nextBtn) nextBtn.disabled = index === sections.length - 1;
      current = index;
    }
    dots.forEach((dot, i) => dot.addEventListener('click', () => goTo(i)));
    if (prevBtn) prevBtn.addEventListener('click', () => goTo(current - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => goTo(current + 1));
    goTo(0);
  }

  /* ── menu dropdown do header ── */
  if (menuBtn && menuDropdown) {
    menuBtn.addEventListener('click', (e) => { e.preventDefault(); menuDropdown.classList.toggle('open'); });
    document.addEventListener('click', (e) => { if (!menuDropdown.contains(e.target)) menuDropdown.classList.remove('open'); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') menuDropdown.classList.remove('open'); });
  }

  /* ── drop zone (aba "Adicionar áudio") ── */
  if (dz) {
    dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', e => {
      e.preventDefault(); dz.classList.remove('over');
      const f = e.dataTransfer.files[0];
      if (f) definirArquivo(f);
    });
  }

  if (gtTextarea) {
    gtTextarea.addEventListener('input', function() {
      const n = this.value.trim().split(/\s+/).filter(Boolean).length;
      const wc = document.getElementById('wordCount');
      if (wc) wc.textContent = n.toLocaleString('pt-BR');
    });
  }

  /* ── microfone (Web Speech API) ── */
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition && startBtn && stopBtn) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'pt-BR';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setStatus('Ouvindo...', 'processando');
      if (micIcon) micIcon.style.animation = 'pulse 1s infinite';
    };

    recognition.onresult = (event) => {
      let textoTemp = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) textoFinal += transcript + ' ';
        else textoTemp += transcript;
      }
      if (transcriptEl) transcriptEl.textContent = corrigirTexto(textoFinal + textoTemp);
    };

    recognition.onerror = (event) => {
      console.error('Erro no reconhecimento:', event.error);
      if (event.error === 'not-allowed' || event.error === 'permission-denied') {
        toast('Permissão de microfone negada. Libere o microfone nas configurações do navegador.', 'error');
      }
      setStatus('Erro no microfone', 'aguardando');
      reconhecendo = false;
    };

    recognition.onend = () => {
      setStatus('Microfone parado', 'aguardando');
      if (micIcon) micIcon.style.animation = 'none';
      reconhecendo = false;
    };

    startBtn.onclick = () => {
      if (!reconhecendo) {
        recognition.start();
        reconhecendo = true;
      }
    };

    stopBtn.onclick = () => { recognition.stop(); };

  } else if (startBtn) {
    startBtn.addEventListener('click', () => toast('Seu navegador não suporta reconhecimento de voz. Tente no Chrome ou Edge.', 'error'));
  }

  /* ── download da transcrição ── */
  if (downloadBtn) {
    downloadBtn.onclick = () => {
      const texto = transcriptEl ? transcriptEl.textContent.trim() : '';
      if (!texto) { toast('Nenhuma transcrição disponível ainda.', 'error'); return; }
      const blob = new Blob([texto], { type: 'text/plain' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'transcricao.txt';
      link.click();
    };
  }

  /* ── carrega o dataset, se essa página tiver um ── */
  carregarDados();
});