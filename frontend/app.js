// Neuvera MVP frontend
const API = "";

const READING_SENTENCE =
  "The rainbow is a division of white light into many beautiful colors that stretch across the sky.";

const STEPS = [
  {
    key: "voice",
    title: "Step 1 of 3 — Voice Test",
    desc: "Read the sentence below aloud, clearly and at a natural pace.",
    media: { audio: true, video: false },
    minSeconds: 5,
    reading: true,
  },
  {
    key: "hand",
    title: "Step 2 of 3 — Hand Tremor Test",
    desc: "Show one hand to the camera, palm out, and hold it as steady as you can for ~8 seconds.",
    media: { audio: false, video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } } },
    minSeconds: 6,
    overlay: "hands",
  },
  {
    key: "movement",
    title: "Step 3 of 3 — Movement & Posture Test",
    desc: "Stand back so head and upper body are visible. Stand upright, then slowly raise both arms out to your sides and lower them, twice. Keep your head still.",
    media: { audio: false, video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } } },
    minSeconds: 6,
    overlay: "pose",
  },
];

const state = {
  stepIndex: 0,
  recordings: {},
  stream: null,
  recorder: null,
  chunks: [],
  startedAt: 0,
  timerId: null,
  overlayRunning: false,
  overlayRAF: null,
  hands: null,
  pose: null,
};

// ----- Navigation -----
function show(viewName) {
  document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
  document.querySelector(`[data-view="${viewName}"]`).classList.remove("hidden");
  if (viewName === "history") loadHistory();
  if (viewName === "test") {
    state.stepIndex = 0;
    state.recordings = {};
    setupStep();
  }
  if (viewName === "home") teardownStream();
}

document.querySelectorAll("[data-nav]").forEach(el => {
  el.addEventListener("click", () => show(el.dataset.nav));
});

// ----- Capture -----
async function setupStep() {
  teardownStream();
  const step = STEPS[state.stepIndex];
  document.getElementById("step-title").textContent = step.title;
  document.getElementById("step-desc").textContent = step.desc;
  document.getElementById("status").textContent = "";
  document.getElementById("btn-start").classList.remove("hidden");
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").classList.add("hidden");
  document.getElementById("timer").textContent = "0s";

  const readingBox = document.getElementById("reading-box");
  if (step.reading) {
    document.getElementById("reading-sentence").textContent = READING_SENTENCE;
    readingBox.classList.remove("hidden");
  } else {
    readingBox.classList.add("hidden");
  }

  const preview = document.getElementById("preview");
  const overlay = document.getElementById("overlay");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia(step.media);
    if (step.media.video) {
      preview.srcObject = state.stream;
      preview.classList.remove("hidden");
      await preview.play().catch(() => {});
      if (step.overlay) {
        overlay.classList.remove("hidden");
        startOverlay(step.overlay);
      } else {
        overlay.classList.add("hidden");
      }
    } else {
      preview.classList.add("hidden");
      overlay.classList.add("hidden");
    }
  } catch (e) {
    document.getElementById("status").textContent =
      "Could not access camera/microphone: " + e.message;
  }
}

function teardownStream() {
  stopOverlay();
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
}

// ----- Live landmark overlay -----
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe";

function syncCanvasSize() {
  const preview = document.getElementById("preview");
  const overlay = document.getElementById("overlay");
  const w = preview.videoWidth || preview.clientWidth;
  const h = preview.videoHeight || preview.clientHeight;
  if (w && h && (overlay.width !== w || overlay.height !== h)) {
    overlay.width = w;
    overlay.height = h;
  }
}

function drawHandsResults(results) {
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  syncCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!results.multiHandLandmarks) return;
  for (const landmarks of results.multiHandLandmarks) {
    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, { color: "#4f8cff", lineWidth: 3 });
    }
    if (window.drawLandmarks) {
      window.drawLandmarks(ctx, landmarks, { color: "#22c55e", lineWidth: 1, radius: 3 });
    }
  }
}

function drawPoseResults(results) {
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  syncCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!results.poseLandmarks) return;
  if (window.drawConnectors && window.POSE_CONNECTIONS) {
    window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, { color: "#4f8cff", lineWidth: 3 });
  }
  if (window.drawLandmarks) {
    window.drawLandmarks(ctx, results.poseLandmarks, { color: "#22c55e", lineWidth: 1, radius: 3 });
  }
}

async function startOverlay(kind) {
  const preview = document.getElementById("preview");
  if (kind === "hands") {
    if (!state.hands) {
      state.hands = new window.Hands({ locateFile: f => `${MP_CDN}/hands/${f}` });
      state.hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      state.hands.onResults(drawHandsResults);
    }
  } else if (kind === "pose") {
    if (!state.pose) {
      state.pose = new window.Pose({ locateFile: f => `${MP_CDN}/pose/${f}` });
      state.pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      state.pose.onResults(drawPoseResults);
    }
  }

  state.overlayRunning = true;
  const tick = async () => {
    if (!state.overlayRunning) return;
    if (preview.readyState >= 2) {
      try {
        if (kind === "hands") await state.hands.send({ image: preview });
        else if (kind === "pose") await state.pose.send({ image: preview });
      } catch (e) { /* ignore single frame errors */ }
    }
    state.overlayRAF = requestAnimationFrame(tick);
  };
  tick();
}

function stopOverlay() {
  state.overlayRunning = false;
  if (state.overlayRAF) {
    cancelAnimationFrame(state.overlayRAF);
    state.overlayRAF = null;
  }
  const canvas = document.getElementById("overlay");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx && ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.add("hidden");
  }
}

// ----- Recording -----
document.getElementById("btn-start").addEventListener("click", () => {
  if (!state.stream) return;
  const step = STEPS[state.stepIndex];
  const mimeType = step.media.video
    ? (MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus") ? "video/webm;codecs=vp9,opus" : "video/webm")
    : (MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm");

  state.chunks = [];
  state.recorder = new MediaRecorder(state.stream, { mimeType });
  state.recorder.ondataavailable = e => { if (e.data.size > 0) state.chunks.push(e.data); };
  state.recorder.onstop = onRecordStop;
  state.recorder.start();
  state.startedAt = Date.now();
  state.timerId = setInterval(() => {
    const s = Math.floor((Date.now() - state.startedAt) / 1000);
    document.getElementById("timer").textContent = s + "s";
  }, 200);

  document.getElementById("btn-start").classList.add("hidden");
  document.getElementById("btn-stop").classList.remove("hidden");
  document.getElementById("status").textContent = "Recording…";
});

document.getElementById("btn-stop").addEventListener("click", () => {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
});

async function onRecordStop() {
  clearInterval(state.timerId);
  state.timerId = null;
  const step = STEPS[state.stepIndex];
  const blob = new Blob(state.chunks, { type: state.recorder.mimeType });
  const seconds = (Date.now() - state.startedAt) / 1000;

  document.getElementById("btn-stop").classList.add("hidden");

  if (seconds < step.minSeconds) {
    document.getElementById("status").textContent =
      `Recording too short (${seconds.toFixed(1)}s). Need at least ${step.minSeconds}s. Try again.`;
    document.getElementById("btn-start").classList.remove("hidden");
    return;
  }

  state.recordings[step.key] = blob;
  document.getElementById("status").textContent = "Saved. Moving on…";

  state.stepIndex++;
  if (state.stepIndex < STEPS.length) {
    setTimeout(setupStep, 500);
  } else {
    teardownStream();
    await submitSession();
  }
}

async function submitSession() {
  show("result");
  document.getElementById("result-content").innerHTML =
    '<p class="status">Analyzing… this can take a few seconds.</p>';

  const fd = new FormData();
  if (state.recordings.voice)    fd.append("voice",    state.recordings.voice,    "voice.webm");
  if (state.recordings.hand)     fd.append("hand",     state.recordings.hand,     "hand.webm");
  if (state.recordings.movement) fd.append("movement", state.recordings.movement, "movement.webm");

  try {
    const r = await fetch(API + "/analyze/session", { method: "POST", body: fd });
    const data = await r.json();
    renderResult(data);
  } catch (e) {
    document.getElementById("result-content").innerHTML =
      `<p class="status">Analysis failed: ${e.message}</p>`;
  }
}

// ----- Result rendering -----
function scoreClass(s) {
  if (s == null) return "na";
  if (s >= 0.66) return "high";
  if (s >= 0.33) return "med";
  return "low";
}

function fmtScore(s) {
  return s == null ? "N/A" : (s * 100).toFixed(0);
}

function renderResult(data) {
  const root = document.getElementById("result-content");
  if (data.status === "failed" || data.detail) {
    root.innerHTML = `<p class="status">Could not analyze: ${data.detail || data.summary || "unknown error"}</p>`;
    return;
  }

  const overall = data.overall_score;
  const voice = (data.voice || {}).score;
  const tremor = (data.tremor || {}).score;
  const movement = (data.movement || {}).score;

  const tiles = `
    <div class="score-card">
      <div class="score-tile ${scoreClass(overall)}">
        <div class="label">Overall risk signal</div>
        <div class="value">${fmtScore(overall)}</div>
      </div>
      <div class="score-tile ${scoreClass(voice)}">
        <div class="label">Voice</div>
        <div class="value">${fmtScore(voice)}</div>
      </div>
      <div class="score-tile ${scoreClass(tremor)}">
        <div class="label">Hand tremor</div>
        <div class="value">${fmtScore(tremor)}</div>
      </div>
      <div class="score-tile ${scoreClass(movement)}">
        <div class="label">Movement</div>
        <div class="value">${fmtScore(movement)}</div>
      </div>
    </div>`;

  const summary = data.summary
    ? `<div class="summary"><strong>Summary:</strong> ${data.summary}</div>`
    : "";

  let signals = "";
  const details = data.signal_details || [];
  if (details.length) {
    signals = `<div class="signal-list"><h3>Detected signals</h3><ul>` +
      details.map(s => `<li><strong>${s.label || s.signal}</strong> — ${s.description || ""}</li>`).join("") +
      `</ul></div>`;
  } else if ((data.signals || []).length) {
    signals = `<div class="signal-list"><h3>Detected signals</h3><ul>` +
      data.signals.map(s => `<li>${s}</li>`).join("") + `</ul></div>`;
  }

  let llm = "";
  const exp = data.llm_explanation;
  if (exp && typeof exp === "object") {
    const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
    const sections = [];
    if (exp.overview) sections.push(`<p>${esc(exp.overview)}</p>`);
    for (const k of ["voice", "tremor", "movement"]) {
      if (exp[k]) sections.push(`<p><strong>${k[0].toUpperCase()+k.slice(1)}:</strong> ${esc(exp[k])}</p>`);
    }
    if (Array.isArray(exp.next_steps) && exp.next_steps.length) {
      sections.push(`<p><strong>Suggested next steps:</strong></p><ul>` +
        exp.next_steps.map(s => `<li>${esc(s)}</li>`).join("") + `</ul>`);
    }
    if (exp.caution) sections.push(`<p class="caution-line"><em>${esc(exp.caution)}</em></p>`);
    if (sections.length) {
      llm = `<div class="llm-block"><h3>AI interpretation</h3>${sections.join("")}</div>`;
    }
  }

  root.innerHTML = tiles + summary + llm + signals;
}

// ----- History -----
async function loadHistory() {
  const root = document.getElementById("history-list");
  root.textContent = "Loading…";
  try {
    const r = await fetch(API + "/history");
    const data = await r.json();
    if (!data.items.length) {
      root.innerHTML = "<p class='status'>No tests yet. Take your first one!</p>";
      return;
    }
    root.innerHTML = data.items.map(i => `
      <div class="history-row">
        <div>
          <div><strong>${i.kind}</strong> — score ${fmtScore(i.overall_score)}</div>
          <div class="when">${new Date(i.created_at).toLocaleString()}</div>
        </div>
        <div>${i.status}</div>
      </div>
    `).join("");
  } catch (e) {
    root.textContent = "Could not load history: " + e.message;
  }
}

// initial
show("home");
