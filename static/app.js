/**
 * Voice-Enabled Sub-200ms RAG - Frontend Application Logic.
 * Web Audio API Analyser + WebSockets + Real-time Latency Waterfall Visualization.
 */

let ws = null;
let audioContext = null;
let mediaStream = null;
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
    console.log("[WS] Disconnected. Reconnecting in 2s...");
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
  } else if (event === "transcript_partial") {
    voiceStatus.textContent = `Listening: "${data.text}"...`;
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
        <span>Score: ${(s.score || 0).toFixed(4)}</span>
      </div>
      <div class="source-text">${s.text || 'Grounded context match.'}</div>
    </div>
  `).join("");
}

// Audio Recording & Web Audio API Visualizer
async function toggleRecording() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 64;
    source.connect(analyserNode);

    isRecording = true;
    micBtn.classList.add("recording");
    micWrapper.classList.add("recording");
    voiceStatus.textContent = "Listening... Speak now";
    answerText.textContent = "";
    answerText.style.color = "#f1f5f9";
    cursorBlink.style.display = "inline-block";

    drawWaveform();

    // Setup ScriptProcessor for audio chunk streaming
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      // Convert float32 to int16 PCM
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm16.buffer);
      }
    };

  } catch (err) {
    console.error("Microphone access denied:", err);
    voiceStatus.textContent = "Microphone access denied. You can use keyboard input.";
  }
}

function stopRecording() {
  isRecording = false;
  micBtn.classList.remove("recording");
  micWrapper.classList.remove("recording");
  voiceStatus.textContent = "Processing audio stream...";

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

function drawWaveform() {
  if (!isRecording || !analyserNode || !canvasCtx) return;

  animationFrameId = requestAnimationFrame(drawWaveform);
  const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(dataArray);

  canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  const barWidth = (waveformCanvas.width / dataArray.length) * 2.2;
  let x = 0;

  for (let i = 0; i < dataArray.length; i++) {
    const barHeight = (dataArray[i] / 255) * waveformCanvas.height * 0.9;
    
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
    // Fallback REST call if WebSocket is connecting
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

  textInput.value = "";
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
