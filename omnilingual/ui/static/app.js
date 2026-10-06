// A thin renderer over the server's messages. There is no transcription logic
// here and no formatting of the transcript: the saved Markdown file is the
// artefact, written by the same renderer the CLI uses, and this only displays
// what the pipeline produced.
const TOKEN_HEADER = "X-Omnilingual-Token";

// The three routes that answer 200 {"started": true} and finish on a detached
// worker thread, each with the button that reaches it. They report on the
// WebSocket log channel, so the page never waits on one for a result and never
// reads a body that is not there. /api/setup/apply is the Setup screen's route
// and phase 1 has no button for it, so the wiring skips a path whose button is
// absent rather than reaching for a null.
const DETACHED = { "/api/audio/setup": "setup-audio",
                   "/api/audio/restart-daemon": "restart-daemon",
                   "/api/setup/apply":"setup-apply" };

// Run states, each with its own wording. 'done' is deliberately bare: it covers
// an exit code of 2 as well as 0, and renderEnd is the only place allowed to
// say which one happened.
const RUN_STATE = {
  idle: "idle — no run",
  running: "running",
  stopping: "stopping — finishing the chunk in flight",
  done: "done",
  failed: "failed",
  halted: "halted — capture continued, the API calls stopped",
};

// 'stopping' holds the window like 'running' does; the three terminal states do
// not. 'halted' is the surprising one — its capture thread is already gone by
// the time the run reports it — so the window is free and recovery is offered.
const BUSY = new Set(["running", "stopping"]);

// A bounded chain of re-reads, never a busy-wait: the detached routes have no
// completion event of their own, so readiness is re-read until it settles or
// the budget runs out.
const REPOLL_ATTEMPTS = 90;
const REPOLL_DELAY_MS = 2000;

// Reconnection is bounded and delayed rather than immediate: a server that has
// gone away must not be hammered, and a window left open overnight must not spin.
const RECONNECT_ATTEMPTS = 20;

// How /api/defaults carries each panel field. The three lists together must
// cover every field but mode, langs, stt and mt, which are filled by their own
// lines; the test file checks that span against the store, so a fourth field
// added to it fails here rather than silently keeping a stale value.
const TEXT_FIELDS = ["out", "source", "stt_model", "mt_model", "device",
                     "work_dir"];
const NUMBER_FIELDS = ["num_speakers", "target_s", "max_chunk_s", "min_chunk_s",
                       "noise_db", "stt_workers", "max_cost"];
const CHECK_FIELDS = ["diarize", "mic_only", "english_only"];

let token = null;
let socket = null;
let reconnects = 0;
let repolls = 0;
let defaults = { default_stt_models: {}, default_mt_models: {} };

const $ = (id) => document.getElementById(id);

// --- requests --------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      [TOKEN_HEADER]: token,
      ...(options.headers || {}),
    },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // One exception type stands behind every 400, so its own sentence is the
    // whole answer and is shown verbatim. The status travels with it for the
    // one case that means something different.
    const err = new Error(body.error || `request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return body;
}

function showBanner(messages, level = "warn") {
  const banner = $("banner");
  banner.textContent = "";
  for (const message of messages) {
    const line = document.createElement("div");
    line.textContent = message;
    banner.appendChild(line);
  }
  banner.className = `banner ${level}`;
}

function clearBanner() {
  $("banner").textContent = "";
  $("banner").className = "banner hidden";
}

function logLine(message, level = "info") {
  const line = document.createElement("div");
  line.className = level;
  line.textContent = message;
  const strip = $("log");
  strip.appendChild(line);
  strip.scrollTop = strip.scrollHeight;
}

// --- the panel -------------------------------------------------------------

function fillSelect(id, values, current) {
  const select = $(id);
  select.textContent = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === current;
    select.appendChild(option);
  }
}

function modelPlaceholder(provider, field, which) {
  // The provider default is the placeholder, never the value: an empty field
  // means "whatever the provider's own default is today".
  $(field).placeholder = defaults[which][$(provider).value]
    || "provider default";
}

async function loadDefaults() {
  const d = await api("/api/defaults");
  defaults = d;
  fillSelect("stt", d.stt_providers, d.stt);
  fillSelect("mt", d.mt_providers, d.mt);
  modelPlaceholder("stt", "stt_model", "default_stt_models");
  modelPlaceholder("mt", "mt_model", "default_mt_models");
  for (const key of TEXT_FIELDS) $(key).value = d[key] ?? "";
  // langs is a list in the store and one comma-separated field here. Blank is
  // the empty list, which is what "auto-detect" means to the pipeline.
  $("langs").value = (d.langs || []).join(",");
  for (const key of NUMBER_FIELDS) {
    if (d[key] !== null && d[key] !== undefined) $(key).value = d[key];
  }
  for (const key of CHECK_FIELDS) $(key).checked = Boolean(d[key]);
  // A relative out path resolves against the repository, which is not where a
  // window launched from Finder starts.
  if (d.last_output_dir) $("out").placeholder = d.last_output_dir;
  const radio = document.querySelector(`input[name=mode][value="${d.mode}"]`);
  if (radio) radio.checked = true;
  toggleMode();
}

async function loadKeys() {
  // Presence only: GET /api/keys answers booleans and a stored value never
  // leaves the server process, so there is nothing here to render but a word.
  const present = await api("/api/keys");
  const box = $("keys");
  box.textContent = "";
  for (const [name, isSet] of Object.entries(present)) {
    const row = document.createElement("div");
    row.className = "key-row";

    const label = document.createElement("span");
    label.className = isSet ? "is-set" : "is-unset";
    label.textContent = `${name}: ${isSet ? "configured" : "not configured"}`;

    const field = document.createElement("input");
    field.type = "password";
    field.placeholder = "new value";
    field.autocomplete = "off";

    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "Save";
    // The typed value goes straight into the request body and nowhere else.
    save.onclick = () => writeKey(name, field.value, () => { field.value = ""; });

    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "Clear";
    clear.onclick = () => writeKey(name, null);

    row.append(label, field, save, clear);
    box.appendChild(row);
  }
}

async function writeKey(name, value, after) {
  try {
    await api("/api/keys", {
      method: "PUT",
      body: JSON.stringify({ [name]: value }),
    });
    clearBanner();
    if (after) after();
    await loadKeys();
  } catch (err) {
    // A refused write leaves the .env byte-identical, so nothing was lost and
    // nothing needs attempting twice. The message is the server's own.
    showBanner([err.message], "error");
  }
}

function currentMode() {
  return document.querySelector("input[name=mode]:checked").value;
}

function toggleMode() {
  const live = currentMode() === "live";
  $("live-only").classList.toggle("hidden", !live);
  // The recording is only read in recording mode; a path left over from an
  // earlier run must not sit in a live payload.
  $("source").disabled = live;
}

function numberField(id) {
  // A cleared box is sent as null, never as Number("") — and Number("") is 0, a
  // real number rather than "not provided". The cost cap is the worst of it: 0
  // means 0 >= 0, so the run halts its API calls on the very first segment and
  // reports itself halted with no stated cause. session._numeric documents blank
  // as absent, so null hands the field back to the backend's own default instead
  // of this page inventing one.
  const raw = $(id).value.trim();
  return raw === "" ? null : Number(raw);
}

function collect() {
  const live = currentMode() === "live";
  const body = {
    mode: live ? "live" : "recording",
    source: $("source").value,
    out: $("out").value,
    stt: $("stt").value,
    stt_model: $("stt_model").value || null,
    mt: $("mt").value,
    mt_model: $("mt_model").value || null,
    diarize: $("diarize").checked,
    num_speakers: $("diarize").checked ? numberField("num_speakers") : null,
    langs: $("langs").value.split(",").map((s) => s.trim()).filter(Boolean),
    english_only: $("english_only").checked,
    work_dir: $("work_dir").value,
    max_chunk_s: numberField("max_chunk_s"),
    min_chunk_s: numberField("min_chunk_s"),
  };
  if (live) {
    Object.assign(body, {
      device: $("device").value,
      mic_only: $("mic_only").checked,
      target_s: numberField("target_s"),
      noise_db: numberField("noise_db"),
      stt_workers: numberField("stt_workers"),
      max_cost: numberField("max_cost"),
    });
  }
  return body;
}

function capabilityPayload() {
  return {
    extras: $("setup-extras").value.trim(),
    live_setup: $("setup-live-setup").checked,
    route_output: $("setup-route-output").checked,
    prefetch: $("setup-prefetch").checked,
    run_tests: $("setup-run-tests").checked,
  };
}

async function previewSetup() {
  const plan = $("setup-plan");
  plan.hidden = false;
  plan.textContent = "working…";
  try {
    const response = await api("/api/setup/preview", {
      method: "POST",
      body: JSON.stringify(capabilityPayload()),
    });
    plan.textContent = response.output;
  } catch (err) {
    plan.textContent = err.message;
  }
}

async function applySetup() {
  // detached() streams the script's own lines into the log strip above and
  // then re-polls readiness, so the restart notice can wait for it: the
  // running process still has the modules it imported before the install.
  await detached("/api/setup/apply", capabilityPayload());
  // setup-apply: the button id wired through the detached map
  showBanner("Setup finished. Restart to use newly installed components.", "warn");
  $("relaunch").hidden = false;
}

async function relaunch() {
  await api("/api/relaunch", { method: "POST" });
}
}

function loadToken() {
  // /token.js is a script, so it is loaded as one. Fetching it would return its
  // text without running it, and the token would stay undefined — every request
  // would then go out without the header and the window would answer itself 403,
  // which looks like a dead server rather than a dead page. Nothing is inlined
  // here and no token is hard-coded; the token is a property of this launch.
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/token.js";
    script.onload = () => {
      if (!window.OMNILINGUAL_TOKEN) {
        reject(new Error("/token.js loaded but set no token"));
        return;
      }
      token = window.OMNILINGUAL_TOKEN;
      resolve();
    };
    // A 403 or a missing file arrives as an error event on the element, not as
    // a thrown exception, so it is caught here or the window sits half-loaded.
    script.onerror = () => reject(new Error("could not load /token.js"));
    document.head.appendChild(script);
  });
}

// --- audio readiness -------------------------------------------------------

function applyAudio(ready) {
  // mic_authorized is true when the probe was skipped as well as when it
  // passed, so mic_checked is what keeps "never asked" from reading as "ready".
  const mic = !ready.mic_checked ? "not checked"
    : ready.mic_authorized ? "authorized" : "not authorized";
  // Read-only, and the only thing the page says about the output device: a
  // wrong one is a top cause of a live run recording nothing, and switching it
  // breaks the user's volume keys, so the server reports it and nobody acts.
  $("audio").textContent = [
    `device ${ready.device}`,
    `blackhole ${ready.blackhole ? "yes" : "no"}`,
    `ffmpeg ${ready.ffmpeg ? "yes" : "no"}`,
    `ffprobe ${ready.ffprobe ? "yes" : "no"}`,
    `microphone ${mic}`,
    `system output ${ready.output || "could not be read"}`,
  ].join(" · ");

  // detail is the server's own wording, one line per problem. It is also where
  // the output-device warning arrives — appended there on purpose and kept out
  // of ok — so it is shown verbatim and coloured as a warning, not a failure.
  const problems = ready.detail.length ? ready.detail
    : ready.ok ? [] : ["capture is not ready"];
  if (problems.length) showBanner(problems, ready.ok ? "warn" : "error");
  else clearBanner();
}

async function checkAudio() {
  // One real probe: GET /api/audio runs the one-second capture, so this is a
  // hardware call and happens at startup and after a fix, never on a render.
  const ready = await api("/api/audio");
  applyAudio(ready);
}

async function detached(path, body) {
  try {
    // These routes answer 200 {"started": true} and finish on a worker thread
    // that cannot fail a request which has already returned. There is nothing
    // to read but the acknowledgement.
    const answer = await api(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!answer.started) return;
    logLine(`working: ${path}`, "warn");
    repolls = REPOLL_ATTEMPTS;
    repollAudio();
  } catch (err) {
    showBanner([err.message], "error");
  }
}

function repollAudio() {
  if (repolls <= 0) {
    // The budget is spent and readiness still is not ok. Whatever the work had
    // to say is in the log strip above, so the page stops asking.
    logLine("still not ready; the log above has the outcome", "warn");
    return;
  }
  repolls -= 1;
  setTimeout(() => {
    api("/api/audio").then((ready) => {
      applyAudio(ready);
      if (ready.ok) return;
      repollAudio();
    }).catch((err) => {
      logLine(err.message, "error");
      repollAudio();
    });
  }, REPOLL_DELAY_MS);
}

// --- running ---------------------------------------------------------------

function fmtTime(seconds) {
  // The saved file's own timestamp shape, so a row and the Markdown beside it
  // read the same way.
  const total = Math.floor(seconds || 0);
  const h = String(Math.floor(total / 3600)).padStart(2, "0");
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function canStop(snapshot) {
  // Only run_live was given a stop channel. A recording or a recovery runs to
  // the end whatever is asked, so the button is not offered rather than
  // promising a stop that cannot arrive.
  return snapshot.mode === "live" && BUSY.has(snapshot.status);
}

function cell(row, text, className = "") {
  const td = document.createElement("td");
  td.textContent = text;
  td.className = className;
  row.appendChild(td);
  return td;
}

function addRow(message) {
  const row = document.createElement("tr");
  // A kept=false segment is out of the saved file. The dimming says exactly
  // that, which is what the pipeline decided when it dropped the silence.
  if (!message.kept) row.className = "dropped";
  cell(row, fmtTime(message.start_s));
  cell(row, message.lang || "");
  cell(row, message.speaker || "", "speaker");
  // status rides along because "ok" is the only value that means the text is a
  // transcription; the rest are the notes the saved file carries.
  cell(row, message.status === "ok" ? "" : message.status || "", "note");
  cell(row, message.text || "");
  // The English cell is empty for an English source, a failure and silence,
  // because that is when the pipeline sends english: null.
  cell(row, message.english || "", "english");
  $("rows-body").appendChild(row);
  row.scrollIntoView({ block: "nearest" });
}

function renderState(message) {
  $("status").textContent = RUN_STATE[message.status] || message.status;
  $("status").dataset.status = message.status;
  $("start").disabled = BUSY.has(message.status);
  $("stop").disabled = !canStop(message);

  const summary = [
    fmtTime(message.elapsed_s),
    `${message.segments} segment${message.segments === 1 ? "" : "s"}`,
  ];
  if (message.dropped) summary.push(`${message.dropped} dropped`);
  function renderCost(message) {
  const meter = $("cost");
  const { cost_inr } = message;
  const spent = typeof cost_inr === "number" ? `₹${cost_inr.toFixed(2)}` : "";
  const cap = typeof message.cost_cap === "number" ? `of ₹${message.cost_cap.toFixed(0)}` : "";
  meter.textContent = spent || cap ? `${spent} ${cap}`.trim() : "";
  meter.hidden = meter.textContent === "";
}

function renderState(message) {
  $("status").textContent = RUN_STATE[message.status] || message.status;
  $("status").dataset.status = message.status;
  $("start").disabled = BUSY.has(message.status);
  $("stop").disabled = !canStop(message);

  const summary = [
    fmtTime(message.elapsed_s),
    `${message.segments} segment${message.segments === 1 ? "" : "s"}`,
  ];
  if (message.dropped) summary.push(`${message.dropped} dropped`);
  renderCost(message);
  $("summary").textContent = summary.join(" · ");

  if (message.error) logLine(message.error, "error");
}

function renderEnd(message) {
  $("stop").disabled = true;
  $("start").disabled = false;
  const panel = $("final");
  panel.textContent = "";
  panel.classList.remove("hidden");

  const saved = document.createElement("div");
  saved.textContent = message.out ? `Saved ${message.out}`
                                   : "no output file was written";
  panel.appendChild(saved);

  // exit_code is the only signal that 'done' was not clean: 2 means the file is
  // written and some segments failed. Without it, done would read as success.
  let verdict;
  if (message.exit_code === 0) {
    verdict = "every segment was transcribed";
  } else if (message.exit_code === 2) {
    verdict = "the transcript is written, but some segments failed";
  } else {
    verdict = `the run ended with exit code ${message.exit_code}`;
  }
  const detail = document.createElement("div");
  detail.className = "note";
  detail.textContent = verdict;
  panel.appendChild(detail);

  if (message.recoverable && message.session_dir) {
    // Offered only when there is something to replay: a Recover that always
    // fails is worse than none.
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Recover this session";
    button.onclick = () => recover(message.session_dir);
    panel.appendChild(button);
  }
}

function renderSegment(message) {
  addRow(message);
}

function render(message) {
  switch (message.type) {
    case "state": renderState(message); break;
    case "segment": addRow(message); break;
    case "status": logLine(message.message); break;
    case "log": logLine(message.message, message.level); break;
    case "end": renderEnd(message); break;
    default: break;
  }
}

async function start() {
  const body = collect();
  try {
    clearBanner();
    $("rows-body").textContent = "";
    $("log").textContent = "";
    $("final").classList.add("hidden");
    await api("/api/session/start", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch (err) {
    // A 409 means this window already holds a run the panel had not seen — a
    // reload raced a start, or a second tab is running. The catalogue is the
    // truth about that, so it is re-read instead of the panel guessing.
    showBanner([err.message], "error");
    if (err.status === 409) await loadRuns();
    return;
  }
  await saveSettings(body);
}

async function saveSettings(body) {
  try {
    // The same payload: the store rejects a key it does not name, and the
    // panel has no field the store does not know.
    await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
  } catch (err) {
    // The run has already started, so a preference that did not save is a note
    // in the log rather than a failure of the run.
    logLine(`settings not saved: ${err.message}`, "warn");
  }
}

async function stop() {
  try {
    await api("/api/session/stop", { method: "POST" });
  } catch (err) {
    showBanner([err.message], "error");
  }
}

async function recover(sessionDir) {
  try {
    $("rows-body").textContent = "";
    $("final").classList.add("hidden");
    await api("/api/session/recover", {
      method: "POST",
      body: JSON.stringify({ session_dir: sessionDir, out: $("out").value }),
    });
  } catch (err) {
    showBanner([err.message], "error");
  }
}

async function loadRuns() {
  const catalogue = await api("/api/runs");
  const box = $("runs");
  box.textContent = "";
  for (const entry of catalogue.runs || []) {
    const row = document.createElement("div");
    row.className = "run-row";
    // out is null for a session discovered on disk: the manifest does not
    // record where its transcript went, so it is shown as unknown rather than
    // guessed at.
    const label = document.createElement("span");
    label.textContent = `${entry.run_id} — ${entry.status} — `
      + `${entry.out || "output path unknown"}`;
    row.appendChild(label);
    if (entry.recoverable && entry.session_dir) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Recover";
      button.onclick = () => recover(entry.session_dir);
      row.appendChild(button);
    }
    box.appendChild(row);
  }
}

// --- the event stream ------------------------------------------------------

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(
    `${scheme}://${location.host}/ws?token=${encodeURIComponent(token)}`);
  socket.onopen = () => {
    reconnects = 0;
    // The handshake sends the current state, so a reconnected window re-renders
    // the run without asking for it. The rows that scrolled past while it was
    // closed are in no response at all — they only ever travelled on this
    // socket — so the catalogue is re-read to show what the run is and where it
    // writes, and the file remains the transcript of record.
    loadRuns().catch((err) => logLine(err.message, "error"));
  };
  socket.onmessage = (event) => render(JSON.parse(event.data));
  socket.onclose = () => {
    // A closed window never touches the run: it lives in the server's process
    // and keeps writing its file. Reconnecting is all that is needed.
    logLine("event stream closed; the run continues", "warn");
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  reconnects += 1;
  if (reconnects > RECONNECT_ATTEMPTS) {
    logLine("the server is not answering; reloading the page will retry", "error");
    return;
  }
  setTimeout(connect, Math.min(1000 * reconnects, 5000));
}

// --- start-up --------------------------------------------------------------

async function main() {
  await loadToken();

  await loadDefaults();
  await loadKeys();
  await checkAudio();
  await loadRuns();
  connect();

  $("start").onclick = start;
  $("stop").onclick = stop;
  for (const [path, button] of Object.entries(DETACHED)) {
    if ($(button)) $(button).onclick = () => detached(path);
  }
  $("setup-preview").onclick = previewSetup;
  $("setup-apply").onclick = applySetup;
  $("relaunch").onclick = relaunch;
  for (const radio of document.querySelectorAll("input[name=mode]")) {
    radio.onchange = toggleMode;
  }
  $("stt").onchange = () =>
    modelPlaceholder("stt", "stt_model", "default_stt_models");
  $("mt").onchange = () =>
    modelPlaceholder("mt", "mt_model", "default_mt_models");
}

main().catch((err) => showBanner([err.message], "error"));
