// Small Random Forest classifier in plain JavaScript (no libraries).
// Trained in the browser from the NASA Kepler table (kepler.csv).

export const FEATURES = [
  "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
  "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad",
];

// Small seeded random generator so results are the same on every visit.
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Read the CSV text. Label 1 = CONFIRMED planet, 0 = FALSE POSITIVE.
export function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  const header = lines[0].split(",").map((h) => h.replace(/"/g, "").trim());
  const labelIdx = header.indexOf("koi_disposition");
  const featIdx = FEATURES.map((f) => header.indexOf(f));
  if (labelIdx < 0 || featIdx.some((i) => i < 0)) {
    throw new Error("kepler.csv is missing expected columns");
  }
  const X = [];
  const y = [];
  for (let r = 1; r < lines.length; r++) {
    const cols = lines[r].split(",");
    const label = (cols[labelIdx] || "").replace(/"/g, "").trim();
    if (label !== "CONFIRMED" && label !== "FALSE POSITIVE") continue;
    const row = featIdx.map((i) => parseFloat(cols[i]));
    if (row.some((v) => !Number.isFinite(v))) continue;
    X.push(row);
    y.push(label === "CONFIRMED" ? 1 : 0);
  }
  return { X, y };
}

// Shuffle rows and split into train / test.
export function splitData(X, y, testFraction, rng) {
  const idx = X.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.round(idx.length * testFraction);
  const pick = (ids) => ({ X: ids.map((i) => X[i]), y: ids.map((i) => y[i]) });
  return { test: pick(idx.slice(0, nTest)), train: pick(idx.slice(nTest)) };
}

function buildTree(X, y, idx, depth, opts, rng) {
  const n = idx.length;
  let pos = 0;
  for (const i of idx) pos += y[i];
  const p = pos / n;
  if (depth >= opts.maxDepth || n < opts.minSplit || pos === 0 || pos === n) {
    return { p };
  }

  const nFeat = X[0].length;
  const feats = [];
  while (feats.length < opts.mtry) {
    const f = Math.floor(rng() * nFeat);
    if (!feats.includes(f)) feats.push(f);
  }

  const parentGini = 2 * p * (1 - p);
  let best = { gain: 1e-9, f: -1, thr: 0 };
  for (const f of feats) {
    const order = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
    let leftN = 0;
    let leftPos = 0;
    for (let k = 0; k < n - 1; k++) {
      const i = order[k];
      leftN++;
      leftPos += y[i];
      const v = X[i][f];
      const next = X[order[k + 1]][f];
      if (v === next) continue;
      const rightN = n - leftN;
      if (leftN < opts.minLeaf || rightN < opts.minLeaf) continue;
      const rightPos = pos - leftPos;
      const pl = leftPos / leftN;
      const pr = rightPos / rightN;
      const gain = parentGini - (leftN * 2 * pl * (1 - pl) + rightN * 2 * pr * (1 - pr)) / n;
      if (gain > best.gain) best = { gain, f, thr: (v + next) / 2 };
    }
  }
  if (best.f < 0) return { p };

  const left = [];
  const right = [];
  for (const i of idx) (X[i][best.f] <= best.thr ? left : right).push(i);
  return {
    f: best.f,
    thr: best.thr,
    l: buildTree(X, y, left, depth + 1, opts, rng),
    r: buildTree(X, y, right, depth + 1, opts, rng),
  };
}

// Train the forest. onProgress(done, total) lets the page show a progress bar.
export async function trainForest(X, y, options = {}, onProgress = () => {}) {
  const opts = {
    nTrees: 40,
    maxDepth: 12,
    minSplit: 4,
    minLeaf: 2,
    mtry: 3,
    maxSamples: 3000,
    seed: 42,
    ...options,
  };
  const rng = makeRng(opts.seed);
  const trees = [];
  for (let t = 0; t < opts.nTrees; t++) {
    const size = Math.min(opts.maxSamples, X.length);
    const idx = [];
    for (let i = 0; i < size; i++) idx.push(Math.floor(rng() * X.length)); // bootstrap
    trees.push(buildTree(X, y, idx, 0, opts, rng));
    onProgress(t + 1, opts.nTrees);
    if (t % 2 === 1) await new Promise((r) => setTimeout(r, 0)); // keep page responsive
  }
  return trees;
}

function walk(node, row) {
  while (node.l) node = row[node.f] <= node.thr ? node.l : node.r;
  return node.p;
}

// Probability that the signal is a real planet (average over all trees).
export function predictProba(trees, row) {
  let s = 0;
  for (const t of trees) s += walk(t, row);
  return s / trees.length;
}

export function accuracy(trees, X, y) {
  let ok = 0;
  for (let i = 0; i < X.length; i++) {
    if ((predictProba(trees, X[i]) >= 0.5 ? 1 : 0) === y[i]) ok++;
  }
  return ok / X.length;
}
