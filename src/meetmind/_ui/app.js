// MeetMind overlay client — editorial transcript renderer.
//
// Renders transcript as contiguous *blocks* per speaker (a court-reporter
// style block) rather than per-segment rows. Partials show with a blinking
// caret. Speaker changes get a colored vertical rule in the margin.
//
// SSE protocol unchanged: connects to /v1/transcripts/live, parses
// `event:`/`data:` frames. Auto-handshake when served same-origin from
// the API.

// ─────────── elements ───────────
const $ = (sel) => document.querySelector(sel);
const captions = $("#captions");
const hint = $("#hint");
const dot = $("#status-dot");
const endpointInput = $("#endpoint");
const tokenInput = $("#token");
const connectBtn = $("#connect");
const sessionState = $("#session-state");
const sessionTimer = $("#session-timer");
const coachEl = $("#coach");
const coachText = $("#coach-text");
const coachCat = $("#coach-cat");
const coachClose = $("#coach-close");

// ─────────── speaker palette assignment ───────────
// Deterministic — same speaker id -> same color index, across sessions.
const PALETTE_SIZE = 8;
const speakerIdx = new Map();

function indexFor(speakerKey) {
  if (!speakerKey) return 0;
  if (speakerIdx.has(speakerKey)) return speakerIdx.get(speakerKey);
  // FNV-1a 32-bit hash modulo palette size.
  let h = 0x811c9dc5;
  for (let i = 0; i < speakerKey.length; i++) {
    h ^= speakerKey.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  const idx = h % PALETTE_SIZE;
  speakerIdx.set(speakerKey, idx);
  return idx;
}

// ─────────── transcript block management ───────────
// We render contiguous speaker runs as a single "block":
//   <div class="block" data-spk-idx="N">
//     <div class="block-head">…label · time</div>
//     <div class="block-body">…inline segments…</div>
//   </div>

let currentBlock = null;
let currentSpeakerKey = null;
let partialSegment = null;
let sessionStartMs = null;

function clearHint() {
  if (hint && hint.parentElement) hint.remove();
}

function formatClock(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function speakerLabel(speakerKey, stream) {
  if (!speakerKey || speakerKey === "·") {
    return stream === "loopback" ? "loopback" : stream === "mic" ? "you" : "unknown";
  }
  // cluster_id form like "spk-0", "spk-1" → render as "spk 0"
  if (/^spk[-_]?\d+$/i.test(speakerKey)) {
    return speakerKey.replace(/[-_]/, " ");
  }
  return speakerKey;
}

function ensureBlock(speakerKey, stream, startMs) {
  if (currentBlock && currentSpeakerKey === speakerKey) return currentBlock;

  clearHint();

  const block = document.createElement("div");
  block.className = "block";
  if (speakerKey === "system" || speakerKey === "error") {
    block.classList.add("block-system");
  } else {
    block.dataset.spkIdx = String(indexFor(speakerKey));
  }

  const head = document.createElement("div");
  head.className = "block-head";
  const name = document.createElement("div");
  name.className = "block-name";
  name.textContent = speakerLabel(speakerKey, stream);
  const time = document.createElement("div");
  time.className = "block-time";
  const elapsed = sessionStartMs == null ? startMs : Math.max(0, startMs);
  time.textContent = formatClock(elapsed);
  head.appendChild(name);
  head.appendChild(time);

  const body = document.createElement("div");
  body.className = "block-body";

  block.appendChild(head);
  block.appendChild(body);
  captions.appendChild(block);

  currentBlock = block;
  currentSpeakerKey = speakerKey;
  partialSegment = null;
  return block;
}

function appendSegment({ speakerKey, stream, text, partial, startMs }) {
  const block = ensureBlock(speakerKey, stream, startMs);
  const body = block.querySelector(".block-body");

  if (partial) {
    if (partialSegment && partialSegment.parentElement === body) {
      partialSegment.textContent = text;
    } else {
      partialSegment = document.createElement("span");
      partialSegment.className = "segment partial";
      partialSegment.textContent = text;
      body.appendChild(partialSegment);
    }
  } else {
    if (partialSegment && partialSegment.parentElement === body) {
      partialSegment.classList.remove("partial");
      partialSegment.textContent = text;
      partialSegment = null;
    } else {
      const seg = document.createElement("span");
      seg.className = "segment";
      seg.textContent = text;
      body.appendChild(seg);
    }
  }

  // auto-scroll only if user is near the bottom (don't fight them mid-read)
  const nearBottom =
    captions.scrollTop + captions.clientHeight >= captions.scrollHeight - 80;
  if (nearBottom) captions.scrollTop = captions.scrollHeight;
}

// ─────────── coach card ───────────

let coachTimer = null;
function showCoachTip(evt) {
  if (!coachEl) return;
  coachText.textContent = evt.tip;
  coachCat.textContent = evt.category || "tip";
  coachEl.hidden = false;
  coachEl.style.animation = "none";
  void coachEl.offsetWidth;
  coachEl.style.animation = "";
  if (coachTimer) clearTimeout(coachTimer);
  coachTimer = setTimeout(() => {
    coachEl.hidden = true;
  }, 12000);
}
if (coachClose) {
  coachClose.addEventListener("click", () => {
    coachEl.hidden = true;
    if (coachTimer) clearTimeout(coachTimer);
  });
}

// ─────────── status & session ticker ───────────

function setStatus(state) {
  dot.classList.remove("connected", "connecting");
  if (state === "connected") dot.classList.add("connected");
  if (state === "connecting") dot.classList.add("connecting");

  if (state === "connected") {
    document.body.classList.add("live");
    sessionState.textContent = "recording";
  } else if (state === "connecting") {
    sessionState.textContent = "connecting";
    document.body.classList.remove("live");
  } else {
    document.body.classList.remove("live");
    sessionState.textContent = "awaiting signal";
  }
}

let tickerInterval = null;
function startSession() {
  sessionStartMs = Date.now();
  if (tickerInterval) clearInterval(tickerInterval);
  tickerInterval = setInterval(() => {
    if (sessionStartMs == null) return;
    sessionTimer.textContent = formatClock(Date.now() - sessionStartMs);
  }, 1000);
}
function stopSession() {
  if (tickerInterval) {
    clearInterval(tickerInterval);
    tickerInterval = null;
  }
}

// ─────────── event dispatch ───────────

function applyEvent(evt) {
  switch (evt.kind) {
    case "partial": {
      const speakerKey = evt.speaker_id || evt.cluster_id || evt.stream || "·";
      appendSegment({
        speakerKey,
        stream: evt.stream,
        text: evt.text,
        partial: true,
        startMs: evt.start_ms || 0,
      });
      break;
    }
    case "final": {
      const speakerKey = evt.speaker_id || evt.cluster_id || evt.stream || "·";
      appendSegment({
        speakerKey,
        stream: evt.stream,
        text: evt.text,
        partial: false,
        startMs: evt.start_ms || 0,
      });
      break;
    }
    case "speaker": {
      const speakerKey = evt.speaker_id || evt.cluster_id || "·";
      appendSegment({
        speakerKey,
        stream: evt.stream,
        text: evt.text,
        partial: false,
        startMs: evt.start_ms || 0,
      });
      break;
    }
    case "diar":
      // diar events without text are silent — they just shift the speaker
      // context for the next segment.
      currentSpeakerKey = evt.cluster_id || currentSpeakerKey;
      break;
    case "meta":
      if (evt.event === "session_started") {
        setStatus("connected");
        if (sessionStartMs == null) startSession();
      } else if (evt.event === "session_stopped") {
        setStatus("disconnected");
        stopSession();
      } else if (evt.event === "error") {
        appendSegment({
          speakerKey: "error",
          stream: "system",
          text: evt.detail || "unknown error",
          partial: false,
          startMs: 0,
        });
      }
      break;
    case "coach_tip":
      showCoachTip(evt);
      break;
  }
}

// ─────────── SSE connection ───────────

let abort = null;

async function connect() {
  if (abort) abort.abort();
  const endpoint = endpointInput.value.replace(/\/+$/, "");
  const token = tokenInput.value.trim();
  if (!endpoint || !token) {
    appendSegment({
      speakerKey: "error",
      stream: "system",
      text: "endpoint and token required",
      partial: false,
      startMs: 0,
    });
    return;
  }
  localStorage.setItem("meetmind", JSON.stringify({ endpoint, token }));
  setStatus("connecting");

  abort = new AbortController();
  let response;
  try {
    response = await fetch(`${endpoint}/v1/transcripts/live`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: abort.signal,
    });
  } catch (e) {
    setStatus("disconnected");
    appendSegment({
      speakerKey: "error",
      stream: "system",
      text: `fetch failed — ${e}`,
      partial: false,
      startMs: 0,
    });
    return;
  }
  if (!response.ok) {
    setStatus("disconnected");
    let body = "";
    try {
      body = await response.text();
    } catch (_) {}
    appendSegment({
      speakerKey: "error",
      stream: "system",
      text: `HTTP ${response.status} — ${body}`,
      partial: false,
      startMs: 0,
    });
    return;
  }
  setStatus("connected");
  if (sessionStartMs == null) startSession();

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      break;
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            applyEvent(JSON.parse(line.slice(6)));
          } catch (_) {}
        }
      }
    }
  }
  setStatus("disconnected");
  stopSession();
}

// ─────────── bootstrap ───────────

const sameOrigin =
  location.protocol === "http:" || location.protocol === "https:";
if (sameOrigin) {
  endpointInput.value = location.origin;
}
const persisted = JSON.parse(localStorage.getItem("meetmind") || "{}");
if (!sameOrigin && persisted.endpoint) endpointInput.value = persisted.endpoint;
if (!sameOrigin && persisted.token) tokenInput.value = persisted.token;

connectBtn.addEventListener("click", connect);

$("#close").addEventListener("click", () => {
  if (abort) abort.abort();
  if (window.__TAURI_INTERNALS__) {
    window.__TAURI_INTERNALS__.invoke("plugin:window|hide", {}).catch(() => {});
  } else {
    document.body.style.opacity = "0.1";
  }
});

$("#toggle-mode").addEventListener("click", () => {
  const next = document.body.dataset.mode === "overlay" ? "dashboard" : "overlay";
  document.body.dataset.mode = next;
  // Notify dashboard.js it should refresh / hide.
  window.dispatchEvent(new CustomEvent("meetmind:mode", { detail: next }));
});

async function autoHandshake() {
  if (!sameOrigin) return false;
  try {
    const r = await fetch(`${location.origin}/v1/auth/handshake`);
    if (!r.ok) return false;
    const body = await r.json();
    if (body.token) {
      tokenInput.value = body.token;
      return true;
    }
  } catch (_) {}
  return false;
}

(async () => {
  if (await autoHandshake()) {
    await connect();
    return;
  }
  if (persisted.endpoint && persisted.token) {
    await connect();
  }
})();

// Expose a tiny helper for dashboard.js to grab the active token.
window.__meetmindToken = () => tokenInput.value.trim();
