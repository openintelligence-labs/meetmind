// MeetMind dashboard — archive view + search.
// Activates when body[data-mode="dashboard"]. Reads from the same FastAPI
// origin as the SSE client (app.js). Falls back gracefully if endpoints
// are missing.

const $ = (sel) => document.querySelector(sel);
const meetingList = $("#meeting-list");
const dashMain = $("#dash-main");
const dashMeta = $("#dash-meta");
const searchInput = $("#search-input");

let currentMeetingId = null;
let listCache = null;
let searchDebounce = null;
let scrollToSegmentMs = null; // set when user clicks a search hit

// ─────────── speaker palette (matches app.js) ───────────
const PALETTE_SIZE = 8;
const speakerIdx = new Map();
function indexFor(speakerKey) {
  if (!speakerKey) return 0;
  if (speakerIdx.has(speakerKey)) return speakerIdx.get(speakerKey);
  let h = 0x811c9dc5;
  for (let i = 0; i < speakerKey.length; i++) {
    h ^= speakerKey.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  const idx = h % PALETTE_SIZE;
  speakerIdx.set(speakerKey, idx);
  return idx;
}

// ─────────── auth-bearing fetch ───────────
function token() {
  return (window.__meetmindToken && window.__meetmindToken()) || "";
}
async function api(path, init = {}) {
  const tok = token();
  const url = `${location.origin}${path}`;
  const headers = {
    ...(init.headers || {}),
    ...(tok ? { Authorization: `Bearer ${tok}` } : {}),
  };
  if (init.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const r = await fetch(url, { ...init, headers });
  if (!r.ok) {
    const err = new Error(`HTTP ${r.status}`);
    err.status = r.status;
    try {
      const body = await r.json();
      err.detail = body.detail || body;
    } catch (_) {}
    throw err;
  }
  return r.json();
}

// ─────────── formatting helpers ───────────
function formatDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const opts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
    return d.toLocaleString(undefined, opts);
  } catch {
    return iso;
  }
}
function formatDuration(seconds) {
  if (seconds == null) return "—";
  const total = Math.floor(seconds);
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
function formatMs(ms) {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function speakerLabel(key) {
  if (!key) return "unknown";
  if (/^spk[-_]?\d+$/i.test(key)) return key.replace(/[-_]/, " ");
  if (key === "loopback") return "loopback";
  if (key === "mic") return "you";
  return key;
}

// ─────────── meeting list ───────────
async function loadMeetings() {
  meetingList.innerHTML = `
    <div class="skeleton-list">
      <div class="skel-line"></div>
      <div class="skel-line short"></div>
      <div class="skel-line"></div>
      <div class="skel-line medium"></div>
      <div class="skel-line"></div>
    </div>`;
  try {
    const data = await api("/v1/meetings");
    listCache = data.meetings || [];
    renderMeetingList(listCache);
  } catch (e) {
    meetingList.innerHTML = `<div class="meetings-empty">no archive endpoint — start a recording to populate</div>`;
  }
}

function renderMeetingList(items) {
  if (!items || !items.length) {
    meetingList.innerHTML = `<div class="meetings-empty">no meetings yet</div>`;
    return;
  }
  meetingList.innerHTML = "";
  for (const m of items) {
    const row = document.createElement("div");
    row.className = "meeting-row";
    if (m.id === currentMeetingId) row.classList.add("active");
    row.dataset.id = m.id;
    const title = m.title || m.id;
    const when = formatDate(m.started_at || m.created_at);
    const dur = formatDuration(m.duration_seconds);
    const segs = m.segment_count != null ? `${m.segment_count} segs` : "";
    row.innerHTML = `
      <div class="row-title">${escapeHtml(title)}</div>
      <div class="row-meta">
        ${when ? `<span>${escapeHtml(when)}</span><span class="dot-sep">·</span>` : ""}
        <span>${escapeHtml(dur)}</span>
        ${segs ? `<span class="dot-sep">·</span><span>${escapeHtml(segs)}</span>` : ""}
      </div>`;
    row.addEventListener("click", () => selectMeeting(m.id));
    meetingList.appendChild(row);
  }
}

// ─────────── meeting detail ───────────
async function selectMeeting(id) {
  currentMeetingId = id;
  document
    .querySelectorAll(".meeting-row")
    .forEach((r) => r.classList.toggle("active", r.dataset.id === id));

  dashMain.innerHTML = `
    <div class="skeleton-list" style="padding:36px 0">
      <div class="skel-line"></div>
      <div class="skel-line short"></div>
      <div class="skel-line"></div>
    </div>`;
  dashMeta.hidden = true;

  try {
    const data = await api(`/v1/meeting/${encodeURIComponent(id)}`);
    renderMeetingDetail(data);
  } catch (e) {
    dashMain.innerHTML = `<div class="dash-empty"><p>could not load meeting (${e.status || e.message})</p></div>`;
  }
}

function renderMeetingDetail(data) {
  const m = data.meeting || {};
  const segments = data.segments || [];
  const summary = data.summary;

  // ── HEAD ──
  const head = document.createElement("div");
  head.className = "detail-head";
  const title = m.title || m.id || "Untitled meeting";
  const when = formatDate(m.started_at || m.created_at);
  const segCount = segments.length;
  const dur = formatDuration(m.duration_seconds);
  head.innerHTML = `
    <h1 class="detail-title">${escapeHtml(title)}</h1>
    <div class="detail-meta">
      ${when ? `<span>${escapeHtml(when)}</span>` : ""}
      <span class="meta-strong">${escapeHtml(dur)}</span>
      <span>${segCount} segments</span>
      ${summary?.model ? `<span class="meta-model">summary · ${escapeHtml(summary.model)}</span>` : ""}
    </div>`;

  // ── SUMMARY CARD (only if persisted) ──
  let summaryCard = null;
  if (summary && summary.tl_dr) {
    summaryCard = document.createElement("div");
    summaryCard.className = "summary-card";
    const topicChips = (summary.topics || [])
      .map((t) => `<span class="topic-chip">${escapeHtml(t)}</span>`)
      .join("");
    summaryCard.innerHTML = `
      <div class="summary-label">TL;DR</div>
      <p class="summary-text">${escapeHtml(summary.tl_dr)}</p>
      ${topicChips ? `<div class="summary-topics">${topicChips}</div>` : ""}
    `;
  }

  // ── TRANSCRIPT: group by contiguous speaker (same as overlay) ──
  const transcript = document.createElement("div");
  transcript.className = "transcript";
  transcript.style.padding = "0";
  let block = null;
  let lastKey = null;
  for (const s of segments) {
    const key = s.speaker_id || s.cluster_id || s.speaker || s.stream || "·";
    if (key !== lastKey) {
      block = document.createElement("div");
      block.className = "block";
      block.dataset.spkIdx = String(indexFor(key));
      const headBox = document.createElement("div");
      headBox.className = "block-head";
      headBox.innerHTML = `
        <div class="block-name">${escapeHtml(speakerLabel(key))}</div>
        <div class="block-time">${formatMs(s.start_ms || 0)}</div>`;
      const body = document.createElement("div");
      body.className = "block-body";
      block.appendChild(headBox);
      block.appendChild(body);
      transcript.appendChild(block);
      lastKey = key;
    }
    const seg = document.createElement("span");
    seg.className = "segment";
    seg.dataset.startMs = String(s.start_ms || 0);
    seg.textContent = s.text || "";
    block.querySelector(".block-body").appendChild(seg);
  }

  // ── AUDIO PLAYER (only if persisted) ──
  let audioPlayer = null;
  const audioPaths = [];
  if (m.audio_path_mic) audioPaths.push({ stream: "mic", path: m.audio_path_mic });
  if (m.audio_path_loopback)
    audioPaths.push({ stream: "loopback", path: m.audio_path_loopback });
  if (audioPaths.length) {
    audioPlayer = document.createElement("div");
    audioPlayer.className = "audio-player";
    audioPlayer.innerHTML = audioPaths
      .map(
        ({ stream }) => `
        <div class="audio-stream">
          <label>${escapeHtml(stream === "mic" ? "you (mic)" : "system (loopback)")}</label>
          <audio
            controls
            preload="metadata"
            src="/v1/meeting/${encodeURIComponent(m.id)}/audio/${encodeURIComponent(stream)}?t=${encodeURIComponent(token())}"
            data-stream="${escapeHtml(stream)}"
          ></audio>
        </div>`
      )
      .join("");
  }

  dashMain.innerHTML = "";
  dashMain.appendChild(head);
  if (audioPlayer) dashMain.appendChild(audioPlayer);
  if (summaryCard) dashMain.appendChild(summaryCard);
  if (!segments.length) {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.style.height = "200px";
    empty.innerHTML = `<p>no transcript segments stored for this meeting</p>`;
    dashMain.appendChild(empty);
  } else {
    dashMain.appendChild(transcript);
  }

  // ── META PANEL: actions / decisions / attendees ──
  renderMetaPanel(data, segments);
  dashMeta.hidden = false;

  // ── search jump-to-segment ──
  // If the user clicked a search hit before this render, scroll to the
  // matching segment + flash a highlight. Match by closest start_ms ≤
  // requested — the indexer rounds slightly, and the dashboard groups
  // contiguous-speaker segments, so the search hit may not land on an
  // exact start_ms boundary.
  if (scrollToSegmentMs != null) {
    requestAnimationFrame(() => {
      const target = scrollToSegmentMs;
      scrollToSegmentMs = null;
      const candidates = dashMain.querySelectorAll(".segment[data-start-ms]");
      let best = null;
      let bestDiff = Infinity;
      for (const c of candidates) {
        const ms = Number(c.dataset.startMs || 0);
        const diff = Math.abs(ms - target);
        if (diff < bestDiff) {
          best = c;
          bestDiff = diff;
        }
      }
      if (best) {
        best.scrollIntoView({ behavior: "smooth", block: "center" });
        best.classList.add("segment-flash");
        setTimeout(() => best.classList.remove("segment-flash"), 1800);
      }
    });
  }
}

function renderMetaPanel(data, segments) {
  // actions
  const actionsBox = dashMeta.querySelector("#meta-actions");
  const actions = data.actions || [];
  actionsBox.innerHTML = `<h3>Action items</h3>` +
    (actions.length
      ? actions
          .map((a) => {
            const owner = a.owner || "unassigned";
            const due = a.due ? ` · due ${escapeHtml(a.due)}` : "";
            const status = a.status ? ` · ${escapeHtml(a.status)}` : "";
            return `<div class="meta-item">${escapeHtml(a.description || "")}
              <span class="meta-sub">${escapeHtml(owner)}${due}${status}</span>
            </div>`;
          })
          .join("")
      : `<div class="meta-empty">None extracted yet. Run <code>meetmind summarize ${escapeHtml(currentMeetingId || "ID")}</code>.</div>`);

  // decisions
  const decisionsBox = dashMeta.querySelector("#meta-decisions");
  const decisions = data.decisions || [];
  decisionsBox.innerHTML = `<h3>Decisions</h3>` +
    (decisions.length
      ? decisions
          .map((d) => {
            const rationale = d.rationale
              ? `<span class="meta-sub">${escapeHtml(d.rationale)}</span>`
              : "";
            return `<div class="meta-item">${escapeHtml(d.decision || "")}${rationale}</div>`;
          })
          .join("")
      : `<div class="meta-empty">None recorded.</div>`);

  // attendees (computed from segments)
  const attendBox = dashMeta.querySelector("#meta-attendees");
  const talkTime = new Map();
  for (const s of segments) {
    const key = s.speaker_id || s.cluster_id || s.speaker || s.stream || "·";
    const ms = (s.end_ms || 0) - (s.start_ms || 0);
    talkTime.set(key, (talkTime.get(key) || 0) + Math.max(0, ms));
  }
  const sorted = [...talkTime.entries()].sort((a, b) => b[1] - a[1]);
  attendBox.innerHTML = `<h3>Voices heard</h3>` +
    (sorted.length
      ? sorted
          .map(
            ([key, ms]) =>
              `<div class="meta-speaker" data-spk-idx="${indexFor(key)}">
                <span class="meta-speaker-mark"></span>
                <span class="meta-speaker-label">${escapeHtml(speakerLabel(key))}</span>
                <span class="meta-speaker-time">${formatMs(ms)}</span>
              </div>`
          )
          .join("")
      : `<div class="meta-empty">No speakers identified.</div>`);
}

// ─────────── search ───────────
async function runSearch(query) {
  query = query.trim();
  if (!query) {
    if (currentMeetingId) {
      // restore detail view
      selectMeeting(currentMeetingId);
    } else {
      dashMain.innerHTML = `<div class="dash-empty">
        <div class="empty-mark">M·M</div>
        <p>Select a meeting from the left, or search the archive.</p>
      </div>`;
      dashMeta.hidden = true;
    }
    return;
  }
  dashMain.innerHTML = `
    <div class="detail-head">
      <h1 class="detail-title">Searching “${escapeHtml(query)}”</h1>
      <div class="detail-meta"><span>indexing…</span></div>
    </div>
    <div class="skeleton-list" style="padding:24px 0">
      <div class="skel-line"></div>
      <div class="skel-line short"></div>
      <div class="skel-line medium"></div>
    </div>`;
  dashMeta.hidden = true;

  try {
    const data = await api(`/v1/search?q=${encodeURIComponent(query)}&limit=20`);
    renderSearchResults(query, data.hits || []);
  } catch (e) {
    dashMain.innerHTML = `<div class="dash-empty"><p>search unavailable (${e.status || e.message})</p></div>`;
  }
}

function renderSearchResults(query, hits) {
  const head = `
    <div class="detail-head">
      <h1 class="detail-title">“${escapeHtml(query)}”</h1>
      <div class="detail-meta"><span class="meta-strong">${hits.length}</span><span>hits</span></div>
    </div>`;
  if (!hits.length) {
    dashMain.innerHTML = head + `<div class="dash-empty" style="height:200px"><p>no matches in archive</p></div>`;
    return;
  }
  const rx = new RegExp(`(${query.split(/\s+/).map(escapeRegex).filter(Boolean).join("|")})`, "gi");
  dashMain.innerHTML =
    head +
    hits
      .map((h) => {
        const text = (h.text || "").replace(rx, "<mark>$1</mark>");
        const when = h.meeting_title || h.meeting_id || "unknown";
        const t = formatMs(h.start_ms || 0);
        const score = h.score != null ? `score ${h.score.toFixed(3)}` : "";
        return `<div class="search-hit"
          data-mid="${escapeHtml(h.meeting_id || "")}"
          data-start="${Number(h.start_ms || 0)}">
          <div class="hit-meta">
            <span>${escapeHtml(when)}</span>
            <span>${escapeHtml(t)}</span>
            ${score ? `<span class="hit-score">${score}</span>` : ""}
          </div>
          <div class="hit-text">${text}</div>
        </div>`;
      })
      .join("");
  for (const node of dashMain.querySelectorAll(".search-hit")) {
    node.addEventListener("click", () => {
      const mid = node.dataset.mid;
      const start = Number(node.dataset.start || 0);
      if (mid) {
        scrollToSegmentMs = start;
        selectMeeting(mid);
      }
    });
  }
}

// ─────────── tiny utils ───────────
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ─────────── wire up ───────────
if (searchInput) {
  searchInput.addEventListener("input", (e) => {
    if (searchDebounce) clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => runSearch(e.target.value), 220);
  });
}

// ─────────── recorder controls ───────────
const recToggle = $("#rec-toggle");
const recTitle = $("#rec-title");
const recStream = $("#rec-stream");
const recStatus = $("#rec-status");
const recStatusText = $("#rec-status-text");
const recElapsed = $("#rec-elapsed");

let recIsActive = false;
let recElapsedTimer = null;
let recStartedAt = null;

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function setRecorderUI(active, info = {}) {
  recIsActive = !!active;
  if (active) {
    recToggle.classList.add("recording");
    recToggle.querySelector(".rec-label").textContent = "stop";
    recStatus.hidden = false;
    recStatusText.textContent =
      "recording · " + (info.stream || recStream.value || "both");
    recStream.disabled = true;
    recTitle.disabled = true;
    if (info.started_at) recStartedAt = new Date(info.started_at).getTime();
    else if (recStartedAt == null) recStartedAt = Date.now();
    if (recElapsedTimer) clearInterval(recElapsedTimer);
    recElapsedTimer = setInterval(() => {
      recElapsed.textContent = formatElapsed(Date.now() - recStartedAt);
    }, 1000);
    recElapsed.textContent = formatElapsed(Date.now() - recStartedAt);
  } else {
    recToggle.classList.remove("recording");
    recToggle.querySelector(".rec-label").textContent = "record";
    recStatus.hidden = true;
    recStream.disabled = false;
    recTitle.disabled = false;
    recStartedAt = null;
    if (recElapsedTimer) {
      clearInterval(recElapsedTimer);
      recElapsedTimer = null;
    }
  }
}

async function pollRecordingStatus() {
  try {
    const r = await api("/v1/recording/status");
    setRecorderUI(r.recording, r);
  } catch (_) {
    setRecorderUI(false);
  }
}

async function startRecording() {
  const title = recTitle.value.trim() || null;
  const stream = recStream.value || "both";
  try {
    const r = await api("/v1/recording/start", {
      method: "POST",
      body: JSON.stringify({ title, stream }),
    });
    setRecorderUI(true, r);
    // Refresh list so the new meeting appears
    setTimeout(loadMeetings, 800);
  } catch (e) {
    recStatusText.textContent = `start failed: ${e.detail || e.message}`;
    recStatus.hidden = false;
  }
}

async function stopRecording() {
  recToggle.disabled = true;
  recStatusText.textContent = "stopping…";
  try {
    const r = await api("/v1/recording/stop", { method: "POST" });
    setRecorderUI(false);
    if (r.meeting_id) {
      // Refresh + jump to the just-finished meeting
      await loadMeetings();
      selectMeeting(r.meeting_id);
    }
  } catch (e) {
    recStatusText.textContent = `stop failed: ${e.detail || e.message}`;
  } finally {
    recToggle.disabled = false;
  }
}

if (recToggle) {
  recToggle.addEventListener("click", () => {
    if (recIsActive) stopRecording();
    else startRecording();
  });
}

// ─────────── summarize control ───────────
async function summarizeCurrentMeeting() {
  if (!currentMeetingId) return;
  const btn = $("#btn-summarize");
  const hint = $("#summarize-hint");
  if (!btn || !hint) return;
  btn.disabled = true;
  hint.className = "tool-hint";
  hint.textContent = "running gemma4 locally · this can take ~30s";
  try {
    const r = await api(
      `/v1/meeting/${encodeURIComponent(currentMeetingId)}/summarize`,
      { method: "POST" }
    );
    hint.className = "tool-hint success";
    hint.textContent = `done · ${r.actions_accepted} actions, ${r.decisions_accepted} decisions (${r.model})`;
    await selectMeeting(currentMeetingId);
  } catch (e) {
    hint.className = "tool-hint error";
    hint.textContent = `failed: ${e.detail || e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ─────────── compliance panel ───────────
async function loadCompliance() {
  const box = document.querySelector("#meta-compliance");
  if (!box) return;
  try {
    const r = await api("/v1/compliance/status");
    const enc = r.encryption || {};
    const ret = r.retention || {};
    const c = r.counts || {};
    const isEncrypted = enc.mode === "encrypted";
    box.innerHTML = `
      <h3>Compliance</h3>
      <div class="meta-item">
        <span class="compliance-pill ${isEncrypted ? "good" : "warn"}">${escapeHtml(enc.mode || "?")}</span>
        <span class="meta-sub">${escapeHtml(enc.driver || "")}</span>
      </div>
      <div class="meta-item">
        <span class="meta-sub-strong">retention</span>
        <span class="meta-sub">
          meetings ${ret.meetings_days ?? "—"}d · voiceprints ${ret.voiceprints_days ?? "—"}d
        </span>
      </div>
      <div class="meta-item">
        <span class="meta-sub-strong">archive</span>
        <span class="meta-sub">
          ${c.meetings ?? 0} meetings · ${c.speakers ?? 0} speakers · ${c.consent_events ?? 0} consent events
        </span>
      </div>
    `;
  } catch (e) {
    box.innerHTML = `<h3>Compliance</h3><div class="meta-empty">unavailable (${e.status || e.message})</div>`;
  }
}

// ─────────── rename / delete / export / diarize ───────────
async function renameCurrentMeeting() {
  if (!currentMeetingId) return;
  const cur = document.querySelector(".detail-title")?.textContent || "";
  const next = window.prompt("New title:", cur);
  if (next == null) return;
  const title = next.trim();
  if (!title) return;
  const hint = $("#manage-hint");
  hint.className = "tool-hint";
  hint.textContent = "renaming…";
  try {
    await api(`/v1/meeting/${encodeURIComponent(currentMeetingId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    hint.className = "tool-hint success";
    hint.textContent = "renamed";
    await loadMeetings();
    await selectMeeting(currentMeetingId);
  } catch (e) {
    hint.className = "tool-hint error";
    hint.textContent = `failed: ${e.detail || e.message}`;
  }
}

async function deleteCurrentMeeting() {
  if (!currentMeetingId) return;
  const confirmed = window.confirm(
    "Delete this meeting? This removes transcripts, summary, actions, decisions, and any persisted audio. Cannot be undone."
  );
  if (!confirmed) return;
  const hint = $("#manage-hint");
  hint.className = "tool-hint";
  hint.textContent = "deleting…";
  try {
    await api(`/v1/meeting/${encodeURIComponent(currentMeetingId)}`, {
      method: "DELETE",
    });
    const deletedId = currentMeetingId;
    currentMeetingId = null;
    await loadMeetings();
    dashMain.innerHTML = `<div class="dash-empty"><p>deleted ${escapeHtml(deletedId)}</p></div>`;
    dashMeta.hidden = true;
  } catch (e) {
    hint.className = "tool-hint error";
    hint.textContent = `failed: ${e.detail || e.message}`;
  }
}

async function exportTo(target) {
  if (!currentMeetingId) return;
  const hint = $("#export-hint");
  hint.className = "tool-hint";
  let body = {};
  if (target === "obsidian") {
    const vault = window.prompt(
      "Obsidian vault path (absolute):",
      localStorage.getItem("meetmind:obsidian-vault") || ""
    );
    if (!vault) return;
    localStorage.setItem("meetmind:obsidian-vault", vault);
    body = { vault_path: vault };
  } else if (target === "github") {
    const repo = window.prompt(
      "GitHub repo (owner/name):",
      localStorage.getItem("meetmind:github-repo") || ""
    );
    if (!repo) return;
    localStorage.setItem("meetmind:github-repo", repo);
    body = { repo, dry_run: false };
  } else if (target === "slack") {
    const webhook = window.prompt(
      "Slack webhook (blank to use $SLACK_WEBHOOK_URL):",
      localStorage.getItem("meetmind:slack-webhook") || ""
    );
    if (webhook == null) return;
    if (webhook) {
      localStorage.setItem("meetmind:slack-webhook", webhook);
      body.webhook_url = webhook;
    }
  } else {
    return;
  }
  hint.textContent = `exporting to ${target}…`;
  try {
    const r = await api(
      `/v1/meeting/${encodeURIComponent(currentMeetingId)}/export/${target}`,
      { method: "POST", body: JSON.stringify(body) }
    );
    if (r.ok === false) {
      hint.className = "tool-hint error";
      hint.textContent = `failed: ${r.error || JSON.stringify(r)}`;
      return;
    }
    hint.className = "tool-hint success";
    if (target === "obsidian") hint.textContent = `wrote ${r.path}`;
    else if (target === "github") hint.textContent = `opened ${r.issues.length} issues`;
    else hint.textContent = `posted to slack`;
  } catch (e) {
    hint.className = "tool-hint error";
    hint.textContent = `failed: ${e.detail || e.message}`;
  }
}

async function diarizeCurrentMeeting() {
  if (!currentMeetingId) return;
  const hint = $("#diarize-hint");
  hint.className = "tool-hint";
  hint.textContent = "running diarizer…";
  try {
    const r = await api(
      `/v1/meeting/${encodeURIComponent(currentMeetingId)}/diarize`,
      { method: "POST" }
    );
    if (r.ok === false) {
      hint.className = "tool-hint error";
      hint.textContent = `failed: ${r.error || JSON.stringify(r)}`;
      return;
    }
    hint.className = "tool-hint success";
    hint.textContent = "diarized · refreshing";
    await selectMeeting(currentMeetingId);
  } catch (e) {
    hint.className = "tool-hint error";
    hint.textContent = `failed: ${e.detail || e.message}`;
  }
}

// Re-bind the summarize button each time we render a meeting detail.
const _origRenderMeetingDetail = renderMeetingDetail;
function _bindSummarizeBtn() {
  const btn = document.querySelector("#btn-summarize");
  if (btn && !btn.dataset.bound) {
    btn.dataset.bound = "1";
    btn.addEventListener("click", summarizeCurrentMeeting);
  }
  const renameBtn = document.querySelector("#btn-rename");
  if (renameBtn && !renameBtn.dataset.bound) {
    renameBtn.dataset.bound = "1";
    renameBtn.addEventListener("click", renameCurrentMeeting);
  }
  const delBtn = document.querySelector("#btn-delete");
  if (delBtn && !delBtn.dataset.bound) {
    delBtn.dataset.bound = "1";
    delBtn.addEventListener("click", deleteCurrentMeeting);
  }
  const obsBtn = document.querySelector("#btn-export-obsidian");
  if (obsBtn && !obsBtn.dataset.bound) {
    obsBtn.dataset.bound = "1";
    obsBtn.addEventListener("click", () => exportTo("obsidian"));
  }
  const ghBtn = document.querySelector("#btn-export-github");
  if (ghBtn && !ghBtn.dataset.bound) {
    ghBtn.dataset.bound = "1";
    ghBtn.addEventListener("click", () => exportTo("github"));
  }
  const slackBtn = document.querySelector("#btn-export-slack");
  if (slackBtn && !slackBtn.dataset.bound) {
    slackBtn.dataset.bound = "1";
    slackBtn.addEventListener("click", () => exportTo("slack"));
  }
  const diarBtn = document.querySelector("#btn-diarize");
  if (diarBtn && !diarBtn.dataset.bound) {
    diarBtn.dataset.bound = "1";
    diarBtn.addEventListener("click", diarizeCurrentMeeting);
  }
}
// dashboard.js doesn't bind directly to currentMeetingId mutation; the
// meta panel is re-rendered on every detail load, so we wire here.
const _dashMetaObs = new MutationObserver(_bindSummarizeBtn);
const _dashMetaEl = document.querySelector("#dash-meta");
if (_dashMetaEl) _dashMetaObs.observe(_dashMetaEl, { childList: true, subtree: true });

window.addEventListener("meetmind:mode", (e) => {
  if (e.detail === "dashboard") {
    loadMeetings();
    pollRecordingStatus();
    loadCompliance();
  }
});

// initial load + status sync
if (document.body.dataset.mode === "dashboard") {
  loadMeetings();
  pollRecordingStatus();
  loadCompliance();
}
// keep status synced for the case where another tab started a recording
setInterval(() => {
  if (document.body.dataset.mode === "dashboard") pollRecordingStatus();
}, 10000);
