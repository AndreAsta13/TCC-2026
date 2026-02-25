const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const transcriptDiv = document.getElementById('transcript');
            const status = document.getElementById('status');

            let mediaRecorder;
            let socket; // para WebSocket da API escolhida

            startBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                // Exemplo com WebSocket para API de streaming (AssemblyAI, Deepgram, etc.)
                socket = new WebSocket('wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000'); // ← exemplo

                // Autenticação (substitua pela sua chave real)
                socket.onopen = () => {
                socket.send(JSON.stringify({ token: 'SUA_CHAVE_API_AQUI' }));
                status.textContent = 'Gravando e transcrevendo...';
                startBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                };

                socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.message_type === 'FinalTranscript' || data.message_type === 'PartialTranscript') {
                    if (data.text) {
                    transcriptDiv.textContent += data.text + ' ';
                    transcriptDiv.scrollTop = transcriptDiv.scrollHeight; // auto-scroll
                    }
                }
                };

                mediaRecorder.ondataavailable = (event) => {
                if (socket.readyState === WebSocket.OPEN && event.data.size > 0) {
                    socket.send(event.data); // envia pedaços de áudio
                }
                };

                mediaRecorder.start(250); // envia a cada 250ms (bom para streaming)

            } catch (err) {
                console.error(err);
                status.textContent = 'Erro ao acessar microfone: ' + err.message;
            }
            });

            stopBtn.addEventListener('click', () => {
                if (mediaRecorder) mediaRecorder.stop();
                if (socket) socket.close();
                status.textContent = 'Parado.';
                startBtn.classList.remove('hidden');
                stopBtn.classList.add('hidden');
            });
        