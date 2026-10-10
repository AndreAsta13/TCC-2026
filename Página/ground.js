/* ══════════════════════════════════════════════════════════════════════════
   ENDEREÇOS DOS BACKENDS
   - API:             Node (login, /me, /arquivos/upload, /gt/..., /transcricoes)
   - API_TRANSCRICAO: Flask (pipeline de transcrição: /transcrever e /status)
   ══════════════════════════════════════════════════════════════════════════ */
const API = 'http://localhost:3000';
const API_TRANSCRICAO = 'http://localhost:5000';

// Salva também o arquivo enviado na conta do usuário (Neon, via Node)?
const SALVAR_ARQUIVO_NA_CONTA = true;

// Intervalo entre as consultas de andamento ao Flask
const INTERVALO_POLLING_MS = 2000;

// Tamanho máximo do arquivo enviado (deve ser <= MAX_CONTENT_LENGTH do Flask: 500 MB)
const TAMANHO_MAXIMO_MB = 500;

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
let emProcessamento = false;   // evita duas transcrições de arquivo ao mesmo tempo
let ultimaTranscricao = '';    // texto "limpo" da última transcrição de arquivo (para o download)
let toastTimer = null;

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

/* ══ Toast (usa classes Tailwind, então funciona no tema escuro) ═══════════ */
function toast(msg, tipo = 'success') {
  const t = document.getElementById('toast');
  if (!t) return;
  const cores = {
    success: 'bg-emerald-600',
    error: 'bg-red-600',
    processando: 'bg-indigo-600',
  };
  t.innerHTML = '';
  const el = document.createElement('div');
  el.className = `${cores[tipo] || cores.success} text-white text-sm font-semibold px-4 py-3 rounded-xl shadow-lg max-w-sm`;
  el.textContent = msg;
  t.appendChild(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.innerHTML = ''; }, 3200);
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

/* ══ Badge de status (classes pensadas para o tema escuro) ═════════════════ */
function setStatus(texto, tipo) {
  if (!statusEl) return;
  statusEl.textContent = texto;
  const base = 'text-xs font-semibold px-3 py-1 rounded-full border ';
  const classes = {
    aguardando:  base + 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
    processando: base + 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    sucesso:     base + 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    erro:        base + 'bg-red-500/20 text-red-300 border-red-500/30',
  };
  statusEl.className = classes[tipo] || classes.aguardando;
}

/* ══════════════════════════════════════════════════════════════════════════
   TRANSCRIÇÃO DE ARQUIVO (Flask)
   Fluxo: POST /transcrever -> recebe job_id -> consulta /status/<job_id>
   a cada 2s até "concluido" ou "erro".
   ══════════════════════════════════════════════════════════════════════════ */
function formatarTempo(segundos) {
  const t = Math.max(0, Math.floor(segundos || 0));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/* Mostra texto na caixa de transcrição e volta a rolagem para o topo
   (a caixa tem altura máxima e rola por dentro quando o texto é grande,
   ex: transcrições de 40 min). */
function mostrarNoTranscript(texto) {
  if (transcriptEl) {
    transcriptEl.textContent = texto;
    transcriptEl.scrollTop = 0;
  }
}

async function iniciarJobTranscricao(arquivo) {
  const form = new FormData();
  form.append('file', arquivo);   // mesmo nome lido por request.files.get('file') no Flask

  let resp;
  try {
    resp = await fetch(`${API_TRANSCRICAO}/transcrever`, { method: 'POST', body: form });
  } catch {
    throw new Error('Servidor de transcrição offline. Verifique se o Flask está rodando na porta 5000.');
  }

  const dados = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(dados.erro || `Erro ${resp.status} ao enviar o arquivo.`);
  return dados.job_id;
}

async function aguardarJob(jobId) {
  const inicio = Date.now();

  while (true) {
    await new Promise(r => setTimeout(r, INTERVALO_POLLING_MS));

    let resp;
    try {
      resp = await fetch(`${API_TRANSCRICAO}/status/${jobId}`);
    } catch {
      throw new Error('Perdi a conexão com o servidor de transcrição.');
    }
    if (!resp.ok) throw new Error('Não foi possível consultar o andamento da transcrição.');

    const job = await resp.json();
    const decorrido = formatarTempo((Date.now() - inicio) / 1000);

    if (job.estado === 'concluido') return job.resultado;
    if (job.estado === 'erro') throw new Error(job.erro || 'Falha na transcrição.');

    if (job.estado === 'na_fila') {
      setStatus(`Na fila (${decorrido})`, 'processando');
      mostrarNoTranscript(`Aguardando a vez na fila de processamento...\nTempo: ${decorrido}`);
    } else {
      setStatus(`Processando (${decorrido})`, 'processando');
      mostrarNoTranscript(
        `Transcrevendo e identificando falantes...\nTempo: ${decorrido}\n\n` +
        `Áudios longos podem levar alguns minutos. Você pode deixar esta aba aberta.`
      );
    }
  }
}

function renderizarResultado(resultado) {
  const segmentos = (resultado && resultado.segmentos) || [];

  if (!segmentos.length) {
    ultimaTranscricao = '';
    mostrarNoTranscript('Nenhuma fala foi detectada neste arquivo.');
    return;
  }

  const linhas = segmentos.map(s => {
    const texto = s.texto_corrigido || s.texto || '';
    const falante = s.falante_global || 'Falante';
    return `[${formatarTempo(s.inicio)} – ${formatarTempo(s.fim)}] ${falante}: ${texto}`;
  });

  // Texto "limpo" usado no botão de download
  ultimaTranscricao = linhas.join('\n\n');

  let saida = ultimaTranscricao;

  const musica = resultado.musica;
  if (musica && musica.titulo) {
    saida = `♪ ${musica.artista || 'Artista desconhecido'} — ${musica.titulo}\n\n` + saida;
  }

  const alertas = (resultado.qualidade || []).flatMap(q =>
    (q.alertas || []).map(a => `• ${q.canal ? q.canal + ': ' : ''}${a}`)
  );
  if (alertas.length) {
    saida += '\n\n⚠ Alertas de qualidade do áudio:\n' + alertas.join('\n');
  }

  mostrarNoTranscript(saida);   // textContent: seguro contra HTML vindo da transcrição
}

/* Salva o arquivo na conta do usuário (Node + Neon). Não bloqueia a
   transcrição: se falhar, só avisa por toast. */
async function enviarArquivoNeon(arquivo) {
  const token = localStorage.getItem("token");

  if (!token) {
    window.location.href = "../cadastro/login.html";
    return false;
  }

  const formData = new FormData();
  formData.append("file", arquivo);

  try {
    const response = await fetch(`${API}/arquivos/upload`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}` },
      body: formData
    });

    const data = await response.json();
    if (!response.ok || !data.sucesso) {
      throw new Error(data.erro || "Erro no upload.");
    }

    toast(`"${data.arquivo.nome_original}" salvo na sua conta.`, "success");
    return true;
  } catch (err) {
    console.error("Erro ao salvar arquivo na conta:", err);
    toast('Não consegui salvar o arquivo na sua conta (a transcrição continua).', 'error');
    return false;
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   HISTÓRICO (pilha LIFO) — salva cada transcrição no Neon (via Node)
   Nunca substitui: cada transcrição vira uma NOVA entrada. Se o servidor
   falhar, fica em localStorage e é sincronizada ao abrir historico.html.
   ══════════════════════════════════════════════════════════════════════════ */
const CHAVE_PENDENTES = 'transcricoes_pendentes';

function novoClientId() {
  return (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : 'id-' + Date.now() + '-' + Math.random().toString(16).slice(2);
}

function montarEntradaDoResultado(arquivo, resultado) {
  const segmentos = (resultado && resultado.segmentos) || [];

  const hipotese = segmentos
    .map(s => s.texto_corrigido || s.texto || '')
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  const palavrasFronteira = segmentos.flatMap(s =>
    (s.palavras_fronteira || []).map(p => ({
      palavra: p.palavra,
      inicio: p.inicio,
      fim: p.fim,
      motivo: p.motivo,
      de: p.de,
      para: p.para,
      falante: s.falante_global || null,
    }))
  );

  return {
    client_id: novoClientId(),
    origem: 'arquivo',
    titulo: arquivo.name,
    nome_arquivo: arquivo.name,
    texto: ultimaTranscricao,
    hipotese,
    segmentos: segmentos.map(s => ({
      inicio: s.inicio,
      fim: s.fim,
      falante: s.falante_global || null,
      texto: s.texto_corrigido || s.texto || '',
    })),
    palavras_fronteira: palavrasFronteira,
    metadados: {
      duracao_audio_s: resultado.duracao_audio_s ?? null,
      tempo_processamento_s: resultado.tempo_processamento_s ?? null,
      fator_tempo_real: resultado.fator_tempo_real ?? null,
      configuracao: resultado.configuracao || null,
      perfil: (resultado.perfil && resultado.perfil.perfil) || null,
    },
  };
}

function montarEntradaDoMicrofone(texto) {
  const limpo = (texto || '').replace(/\s+/g, ' ').trim();
  return {
    client_id: novoClientId(),
    origem: 'microfone',
    titulo: 'Gravação ao vivo — ' + new Date().toLocaleString('pt-BR'),
    nome_arquivo: null,
    texto: limpo,
    hipotese: limpo,
    segmentos: [],
    palavras_fronteira: [],
    metadados: {},
  };
}

async function salvarNoHistorico(entrada) {
  const token = localStorage.getItem('token');
  let salvo = false;

  if (token) {
    try {
      const resp = await fetch(`${API}/transcricoes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(entrada),
      });
      const dados = await resp.json().catch(() => ({}));
      salvo = resp.ok && dados.sucesso;
    } catch (e) {
      console.error('Erro ao salvar no histórico:', e);
    }
  }

  if (!salvo) {
    try {
      const pendentes = JSON.parse(localStorage.getItem(CHAVE_PENDENTES) || '[]');
      pendentes.push({ ...entrada, criado_em: new Date().toISOString() });
      localStorage.setItem(CHAVE_PENDENTES, JSON.stringify(pendentes));
    } catch (e) {
      console.error('Erro ao guardar localmente:', e);
    }
    toast('Histórico guardado só neste navegador (servidor indisponível).', 'error');
  } else {
    toast('Transcrição adicionada ao histórico.', 'success');
  }
  return salvo;
}

async function processarArquivo(arquivo) {
  if (emProcessamento) {
    toast('Já existe uma transcrição em andamento. Aguarde terminar.', 'error');
    return;
  }
  emProcessamento = true;
  ultimaTranscricao = '';

  // Em paralelo: guarda o arquivo na conta enquanto a transcrição roda
  const salvando = SALVAR_ARQUIVO_NA_CONTA ? enviarArquivoNeon(arquivo) : Promise.resolve(true);

  try {
    setStatus('Enviando...', 'processando');
    mostrarNoTranscript(`Enviando "${arquivo.name}" para transcrição...`);

    const jobId = await iniciarJobTranscricao(arquivo);
    const resultado = await aguardarJob(jobId);

    renderizarResultado(resultado);
    setStatus('Concluído', 'sucesso');

    // Empilha no histórico (Neon) — só se houve fala detectada
    if (ultimaTranscricao) {
      await salvarNoHistorico(montarEntradaDoResultado(arquivo, resultado));
    } else {
      toast('Transcrição concluída!', 'success');
    }
  } catch (err) {
    console.error('Erro na transcrição:', err);
    setStatus('Erro na transcrição', 'erro');
    mostrarNoTranscript(err.message);
    toast(err.message, 'error');
  } finally {
    emProcessamento = false;
    await salvando;
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
  if (arquivo.size > TAMANHO_MAXIMO_MB * 1024 * 1024) {
    alert(`Arquivo muito grande. Limite máximo: ${TAMANHO_MAXIMO_MB}MB.`);
    input.value = '';
    return;
  }

  input.value = '';          // permite escolher o mesmo arquivo de novo depois
  processarArquivo(arquivo);
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

  /* ── menu dropdown do header ──
     O painel no HTML usa a classe Tailwind "hidden", então é ela que
     precisa ser alternada (antes o código alternava uma classe "open"
     que não tinha efeito). */
  const menuPanel = document.getElementById('menuPanel');
  if (menuBtn && menuPanel) {
    menuBtn.addEventListener('click', (e) => { e.preventDefault(); menuPanel.classList.toggle('hidden'); });
    document.addEventListener('click', (e) => { if (menuDropdown && !menuDropdown.contains(e.target)) menuPanel.classList.add('hidden'); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') menuPanel.classList.add('hidden'); });
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
      ultimaTranscricao = '';   // a partir daqui o download usa o texto do microfone
      let textoTemp = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) textoFinal += transcript + ' ';
        else textoTemp += transcript;
      }
      if (transcriptEl) {
        transcriptEl.textContent = corrigirTexto(textoFinal + textoTemp);
        // Na gravação ao vivo, acompanha o texto novo rolando para o final
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      }
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

      // Cada gravação vira UMA nova entrada na pilha do histórico
      const texto = corrigirTexto(textoFinal).trim();
      if (texto) {
        salvarNoHistorico(montarEntradaDoMicrofone(texto));
      }
      textoFinal = '';
    };

    startBtn.onclick = () => {
      if (!reconhecendo) {
        textoFinal = '';   // nova gravação = nova entrada
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
      const texto = ultimaTranscricao || (transcriptEl ? transcriptEl.textContent.trim() : '');
      if (!texto || texto.startsWith('O texto transcrito aparecerá aqui')) {
        toast('Nenhuma transcrição disponível ainda.', 'error');
        return;
      }
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