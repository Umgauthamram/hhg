/**
 * Voice-Enabled Sub-200ms RAG - Universal Audio & Voice Streamer.
 * Features:
 * - Studio-Grade Audio Preprocessing: DynamicsCompressor + Gain Booster (3.0x) for quiet/soft voice clarity.
 * - MediaRecorder with normalized WebM/Opus encoding.
 * - Sub-100ms Groq Whisper LPU speech transcription.
 * - Real-time token streaming and microsecond latency telemetry.
 */

let ws = null;
let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let analyserNode = null;
let animationFrameId = null;

// DOM Elements
const micBtn = document.getElementById("micBtn");
const micWrapper = document.getElementById("micWrapper");
const voiceStatus = document.getElementById("voiceStatus");
const waveformCanvas = document.getElementById("waveformCanvas");
const canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;

const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const answerText = document.getElementById("answerText");
const cursorBlink = document.getElementById("cursorBlink");

// Latency Badges
const badgeStt = document.getElementById("badgeStt");
const badgeRetrieval = document.getElementById("badgeRetrieval");
const badgeTtft = document.getElementById("badgeTtft");
const badgeTotal = document.getElementById("badgeTotal");

// Benchmark Gauges
const gaugeP50 = document.getElementById("gaugeP50");
const gaugeP70 = document.getElementById("gaugeP70");
const gaugeP90 = document.getElementById("gaugeP90");
const gaugeP100 = document.getElementById("gaugeP100");
const sourcesList = document.getElementById("sourcesList");

// Initialize WebSocket Connection
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/voice-rag`;
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WS] Connected to Voice-RAG server");
    document.getElementById("serverStatus").textContent = "Connected (Sub-200ms Active)";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleServerEvent(data);
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  ws.onclose = () => {
    document.getElementById("serverStatus").textContent = "Reconnecting...";
    setTimeout(initWebSocket, 2000);
  };
}

function handleServerEvent(data) {
  const event = data.event;

  if (event === "token") {
    cursorBlink.style.display = "inline-block";
    answerText.textContent += data.token;
    updateLatencyBadges(data.latency);
  } else if (event === "complete") {
    cursorBlink.style.display = "none";
    if (data.full_answer) {
      answerText.textContent = data.full_answer;
    }
    updateLatencyBadges(data.latency);
    renderSources(data.sources || []);
    fetchBenchmarkStats();
  } else if (event === "grounding_rejection") {
    cursorBlink.style.display = "none";
    answerText.textContent = data.token;
    answerText.style.color = "#fbbf24";
    updateLatencyBadges(data.latency);
  } else if (event === "error") {
    cursorBlink.style.display = "none";
    answerText.textContent = data.token || data.message || "An error occurred.";
    answerText.style.color = "#f87171";
    if (data.latency) updateLatencyBadges(data.latency);
  } else if (event === "transcript_final") {
    voiceStatus.textContent = `Transcribed: "${data.text}"`;
    textInput.value = data.text;
  }
}

function updateLatencyBadges(latency) {
  if (!latency) return;

  const stt = latency.stt_ms || 0;
  const retrieval = latency.retrieval_ms || 0;
  const ttft = latency.llm_ttft_ms || 0;
  const total = latency.total_pipeline_ms || 0;

  badgeStt.textContent = `${stt.toFixed(1)} ms`;
  badgeRetrieval.textContent = `${retrieval.toFixed(1)} ms`;
  badgeTtft.textContent = `${ttft.toFixed(1)} ms`;
  badgeTotal.textContent = `${total.toFixed(1)} ms`;

  if (total <= 200.0) {
    badgeTotal.className = "total-badge";
    badgeTotal.textContent = `${total.toFixed(1)} ms ✅`;
  } else {
    badgeTotal.className = "total-badge over-target";
  }
}

function renderSources(sources) {
  if (!sources || sources.length === 0) {
    sourcesList.innerHTML = `<div class="source-item"><span class="source-text" style="color:var(--text-muted);">No documents retrieved or ungrounded query.</span></div>`;
    return;
  }

  sourcesList.innerHTML = sources.map((s, idx) => `
    <div class="source-item">
      <div class="source-meta">
        <span>#${idx+1} [${s.language || 'en'}] ${s.doc_id || 'doc'}</span>
        <span>Score: ${(s.score || s.fused_score || 0).toFixed(4)}</span>
      </div>
      <div class="source-text">${s.text || 'Grounded context match.'}</div>
    </div>
  `).join("");
}

// MediaRecorder Audio Capture with Studio Gain & Compressor
async function toggleRecording() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    audioChunks = [];
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });
    
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // 1. Dynamics Compressor (Boosts soft whispers while preventing loud clipping)
    const compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.setValueAtTime(-36, audioContext.currentTime);
    compressor.knee.setValueAtTime(20, audioContext.currentTime);
    compressor.ratio.setValueAtTime(12, audioContext.currentTime);
    compressor.attack.setValueAtTime(0.003, audioContext.currentTime);
    compressor.release.setValueAtTime(0.25, audioContext.currentTime);

    // 2. Gain Booster Node (2.8x amplification for low-volume microphones)
    const gainNode = audioContext.createGain();
    gainNode.gain.setValueAtTime(2.8, audioContext.currentTime);

    // 3. Analyser Node for canvas waveform
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 64;

    // 4. MediaStreamDestination to feed boosted audio into MediaRecorder
    const destinationNode = audioContext.createMediaStreamDestination();

    // Connect audio processing graph:
    // Source -> Compressor -> Gain -> Destination & Analyser
    sourceNode.connect(compressor);
    compressor.connect(gainNode);
    gainNode.connect(destinationNode);
    gainNode.connect(analyserNode);

    // Pick best supported MIME format
    let mimeType = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/ogg";
      }
    }

    mediaRecorder = new MediaRecorder(destinationNode.stream, {
      mimeType: mimeType,
      audioBitsPerSecond: 128000
    });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      await processVoiceAudio(audioBlob);
    };

    mediaRecorder.start(100);
    isRecording = true;
    micBtn.classList.add("recording");
    micWrapper.classList.add("recording");
    voiceStatus.textContent = "🎙️ Listening... (Voice Booster Active - Click mic when done)";
    answerText.textContent = "";
    answerText.style.color = "#f1f5f9";
    cursorBlink.style.display = "inline-block";

    drawWaveform();

  } catch (err) {
    console.error("Microphone access error:", err);
    voiceStatus.textContent = "Microphone access denied. You can type in the box below.";
  }
}

function stopRecording() {
  isRecording = false;
  micBtn.classList.remove("recording");
  micWrapper.classList.remove("recording");
  voiceStatus.textContent = "Transcribing boosted voice with Groq Whisper LPU...";

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close();
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
  clearCanvas();
}

async function processVoiceAudio(audioBlob) {
  try {
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");
    formData.append("language", "en");

    const res = await fetch("/api/voice-query", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    textInput.value = data.query || "";
    voiceStatus.textContent = `Transcribed: "${data.query}"`;
    cursorBlink.style.display = "none";
    answerText.textContent = data.answer;
    updateLatencyBadges(data.latency);
    renderSources(data.sources || []);
    fetchBenchmarkStats();

  } catch (err) {
    console.error("Voice query failed:", err);
    voiceStatus.textContent = "Could not process audio. Please try again or type below.";
    cursorBlink.style.display = "none";
  }
}

function drawWaveform() {
  if (!isRecording || !analyserNode || !canvasCtx) return;

  animationFrameId = requestAnimationFrame(drawWaveform);
  const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(dataArray);

  canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  const barWidth = (waveformCanvas.width / dataArray.length) * 2.2;
  let x = 0;

  for (let i = 0; i < dataArray.length; i++) {
    const barHeight = (dataArray[i] / 255) * waveformCanvas.height * 0.95;
    
    const gradient = canvasCtx.createLinearGradient(0, waveformCanvas.height, 0, 0);
    gradient.addColorStop(0, "#6366f1");
    gradient.addColorStop(1, "#d946ef");

    canvasCtx.fillStyle = gradient;
    canvasCtx.fillRect(x, waveformCanvas.height - barHeight, barWidth - 2, barHeight);
    x += barWidth;
  }
}

function clearCanvas() {
  if (canvasCtx && waveformCanvas) {
    canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }
}

// Text Query Submission
function sendQuery(text) {
  const query = text || textInput.value.trim();
  if (!query) return;

  answerText.textContent = "";
  answerText.style.color = "#f1f5f9";
  cursorBlink.style.display = "inline-block";
  voiceStatus.textContent = `Query: "${query}"`;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "query", text: query }));
  } else {
    fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query })
    })
    .then(res => res.json())
    .then(data => {
      cursorBlink.style.display = "none";
      answerText.textContent = data.answer;
      updateLatencyBadges(data.latency);
      renderSources(data.sources);
    });
  }
}

// Fetch Benchmark Percentile Statistics
async function fetchBenchmarkStats() {
  try {
    const res = await fetch("/api/benchmark/stats");
    if (!res.ok) return;
    const data = await res.json();

    if (data.P50 && data.P50.total_ms > 0) {
      gaugeP50.textContent = `${data.P50.total_ms.toFixed(1)} ms`;
      gaugeP70.textContent = `${data.P70.total_ms.toFixed(1)} ms`;
      gaugeP90.textContent = `${data.P90.total_ms.toFixed(1)} ms`;
      gaugeP100.textContent = `${data.P100.total_ms.toFixed(1)} ms`;
    }
  } catch (e) {
    console.warn("Could not fetch benchmark stats:", e);
  }
}

// Quick Sample Clicks
function setupQuickSamples() {
  document.querySelectorAll(".sample-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      textInput.value = q;
      sendQuery(q);
    });
  });
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  setupQuickSamples();
  fetchBenchmarkStats();

  micBtn.addEventListener("click", toggleRecording);
  sendBtn.addEventListener("click", () => sendQuery());
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuery();
  });

  const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");
  if (runBenchmarkBtn) {
    runBenchmarkBtn.addEventListener("click", async () => {
      runBenchmarkBtn.textContent = "Running Benchmark...";
      runBenchmarkBtn.disabled = true;
      try {
        await fetch("/api/benchmark/stats");
        await fetchBenchmarkStats();
      } finally {
        runBenchmarkBtn.textContent = "Refresh Benchmark Gauges";
        runBenchmarkBtn.disabled = false;
      }
    });
  }
});
