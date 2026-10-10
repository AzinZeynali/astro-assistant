// Astronomy Assistant: runs entirely in the browser (no server).
//  - Search: sentence embeddings (Transformers.js) over passages in knowledge.json
//  - Classifier: Random Forest (forest.js) trained from kepler.csv

import {
  FEATURES, makeRng, parseCsv, splitData, trainForest, predictProba, accuracy,
} from "./forest.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const TOP_K = 3;

const FIELDS = [
  ["koi_period", "Orbital period (days)", 10],
  ["koi_duration", "Transit duration (hours)", 3],
  ["koi_depth", "Transit depth (ppm)", 500],
  ["koi_prad", "Planet radius (Earth radii)", 2],
  ["koi_teq", "Equilibrium temperature (K)", 700],
  ["koi_insol", "Insolation (Earth = 1)", 100],
  ["koi_model_snr", "Signal-to-noise ratio", 20],
  ["koi_steff", "Star temperature (K)", 5700],
  ["koi_slogg", "Star surface gravity (log g)", 4.4],
  ["koi_srad", "Star radius (Sun = 1)", 1],
];

const EXAMPLES = [
  "How do astronomers detect exoplanets?",
  "What makes a planet habitable?",
  "What is the transit method?",
  "What did the Kepler telescope discover?",
];

const $ = (id) => document.getElementById(id);

// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    $("panel-ask").classList.toggle("hidden", btn.dataset.tab !== "ask");
    $("panel-classify").classList.toggle("hidden", btn.dataset.tab !== "classify");
  });
});

// ---------- Part 1: search ----------
let passages = [];
let extractor = null;
let embeddings = null; // Float32Array, passages.length * dim
let dim = 384;

function setSearchStatus(text, progress) {
  $("searchStatus").textContent = text;
  const bar = $("searchProgress");
  if (progress === undefined) bar.classList.add("hidden");
  else { bar.classList.remove("hidden"); bar.value = progress; }
}

// Tiny IndexedDB cache so the index is only built once per visitor.
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("astro-assistant", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("kv");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function cacheGet(key) {
  try {
    const db = await openDb();
    return await new Promise((resolve) => {
      const r = db.transaction("kv").objectStore("kv").get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => resolve(null);
    });
  } catch { return null; }
}
async function cacheSet(key, value) {
  try {
    const db = await openDb();
    db.transaction("kv", "readwrite").objectStore("kv").put(value, key);
  } catch { /* caching is optional */ }
}

async function embed(texts) {
  const out = await extractor(texts, { pooling: "mean", normalize: true });
  return out.data; // flat Float32Array
}

async function buildEmbeddings() {
  const total = passages.length;
  const key = `emb:${MODEL_ID}:${total}:${passages.reduce((s, p) => s + p.x.length, 0)}`;
  const cached = await cacheGet(key);
  if (cached && cached.length === total * dim) return cached;

  const all = new Float32Array(total * dim);
  const BATCH = 16;
  for (let i = 0; i < total; i += BATCH) {
    const batch = passages.slice(i, i + BATCH).map((p) => p.x);
    all.set(await embed(batch), i * dim);
    setSearchStatus(`Preparing search index (first visit only): ${Math.min(i + BATCH, total)} / ${total}`, Math.min(i + BATCH, total) / total);
  }
  cacheSet(key, all);
  return all;
}

async function initSearch() {
  try {
    setSearchStatus("Loading articles…", 0);
    passages = await (await fetch("knowledge.json")).json();

    setSearchStatus("Loading the language model (about 25 MB, first visit only)…", 0);
    const { pipeline } = await import(TRANSFORMERS_URL);
    extractor = await pipeline("feature-extraction", MODEL_ID, {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && typeof p.progress === "number") {
          setSearchStatus(`Downloading the language model… ${Math.round(p.progress)}%`, p.progress / 100);
        }
      },
    });

    dim = (await embed(["test"])).length;
    embeddings = await buildEmbeddings();

    setSearchStatus(`Ready. ${passages.length} passages from ${new Set(passages.map((p) => p.t)).size} articles.`);
    $("searchBtn").disabled = false;
  } catch (err) {
    console.error(err);
    setSearchStatus("Could not load the search model. Check your connection and reload the page.");
  }
}

async function runSearch() {
  const q = $("q").value.trim();
  const box = $("results");
  if (!q || !embeddings) return;
  $("searchBtn").disabled = true;
  const qv = await embed([q]);
  $("searchBtn").disabled = false;

  const scores = new Float32Array(passages.length);
  for (let i = 0; i < passages.length; i++) {
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += embeddings[off + d] * qv[d];
    scores[i] = s;
  }
  const top = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, TOP_K);

  box.replaceChildren();
  top.forEach((i, rank) => {
    const p = passages[i];
    const div = document.createElement("div");
    div.className = "result";

    const h = document.createElement("h3");
    const a = document.createElement("a");
    a.href = "https://en.wikipedia.org/wiki/" + encodeURIComponent(p.t.replace(/ /g, "_"));
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `${rank + 1}. ${p.t}`;
    h.appendChild(a);

    const sc = document.createElement("div");
    sc.className = "score";
    sc.textContent = `similarity ${scores[i].toFixed(2)}`;

    const body = document.createElement("p");
    body.textContent = p.x;

    div.append(h, sc, body);
    box.appendChild(div);
  });
}

$("searchBtn").addEventListener("click", runSearch);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
EXAMPLES.forEach((text) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = text;
  b.addEventListener("click", () => { $("q").value = text; runSearch(); });
  $("examples").appendChild(b);
});

// ---------- Part 2: classifier ----------
let trees = null;
let testSet = null;
let currentTruth = null; // NASA label for the last random example (null if inputs were edited)

FIELDS.forEach(([key, label, value]) => {
  const wrap = document.createElement("div");
  const l = document.createElement("label");
  l.htmlFor = key;
  l.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.id = key;
  input.value = value;
  input.addEventListener("input", () => { currentTruth = null; });
  wrap.append(l, input);
  $("fields").appendChild(wrap);
});

function setClfStatus(text, progress) {
  $("clfStatus").textContent = text;
  const bar = $("clfProgress");
  if (progress === undefined) bar.classList.add("hidden");
  else { bar.classList.remove("hidden"); bar.value = progress; }
}

async function initClassifier() {
  try {
    setClfStatus("Loading NASA Kepler data…", 0);
    const text = await (await fetch("kepler.csv")).text();
    const { X, y } = parseCsv(text);
    const { train, test } = splitData(X, y, 0.2, makeRng(7));
    testSet = test;

    trees = await trainForest(train.X, train.y, {}, (done, n) =>
      setClfStatus(`Training the Random Forest in your browser… ${done} / ${n} trees`, done / n));

    const acc = accuracy(trees, test.X, test.y);
    setClfStatus(`Ready. Accuracy on ${test.X.length} signals the model never saw: ${(acc * 100).toFixed(1)}%`);
    $("classifyBtn").disabled = false;
    $("randomBtn").disabled = false;
  } catch (err) {
    console.error(err);
    setClfStatus("Could not load the Kepler data. Reload the page to try again.");
  }
}

function readInputs() {
  return FEATURES.map((f) => parseFloat($(f).value));
}

function showPrediction() {
  const row = readInputs();
  if (row.some((v) => !Number.isFinite(v))) {
    $("clfResult").classList.remove("hidden");
    $("verdict").textContent = "Please fill in every field with a number.";
    $("barText").textContent = "";
    $("truth").textContent = "";
    $("bar").classList.add("hidden");
    return;
  }
  const p = predictProba(trees, row);
  const isPlanet = p >= 0.5;
  $("clfResult").classList.remove("hidden");
  $("bar").classList.remove("hidden");
  $("verdict").textContent = isPlanet ? "🪐 Looks like a real planet" : "⚠️ Looks like a false alarm";
  $("barFill").style.width = `${(p * 100).toFixed(0)}%`;
  $("barText").textContent = `${(p * 100).toFixed(0)}% real planet · ${((1 - p) * 100).toFixed(0)}% false alarm`;
  $("truth").textContent = currentTruth === null
    ? ""
    : `NASA's verdict for this real signal: ${currentTruth === 1 ? "confirmed planet" : "false positive"}.`;
}

$("classifyBtn").addEventListener("click", showPrediction);
$("randomBtn").addEventListener("click", () => {
  const i = Math.floor(Math.random() * testSet.X.length);
  FEATURES.forEach((f, k) => { $(f).value = String(+testSet.X[i][k].toPrecision(6)); });
  currentTruth = testSet.y[i];
  showPrediction();
});

// ---------- Start ----------
initClassifier();
initSearch();
