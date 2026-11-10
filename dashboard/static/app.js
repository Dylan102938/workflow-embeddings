/* wfemb dashboard: runs table, compare charts, group-by, queue, run detail, demo. Plain JS, no build step.
   All view state lives in S and is mirrored in the URL hash (bookmarkable). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v) => Charts.fmtVal(v);
const isNum = (v) => typeof v === "number" && isFinite(v);

const STATUSES = ["running", "stale", "done"];
const Q_STATUSES = ["not started", "running", "stale", "done"];
const VAR_NAMES = {
  arch: "architecture", dm: "d_model", dh: "d_head", nl: "n_layers", nh: "n_heads", drop: "dropout", smooth: "label smoothing",
  patch: "screen patch tokens per frame", lr: "peak learning rate", bs: "batch size (sessions)", ep: "epochs",
  beta: "hard-negative beta", seed: "seed",
};
const SMOOTH = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400];   // EMA window in steps (slider index)
const DASHES = [null, [6, 3], [2, 3]];
const MAX_POINTS = 2000;
const REFRESH_MS = 30000;
const KNN = /_avg_log_dist_\d+NN$/;

const TRAIN_DEFS = [
  { k: "loss", sub: "training objective", smooth: true },
  { k: "rigid_loss", sub: "one-hot target, no hard-negative weights", smooth: true },
  { k: "avg_dist", sub: "to the other sessions' keys in the batch", smooth: true },
  { k: "lr", sub: "", smooth: false },
  { k: "grad_norm", sub: "", smooth: true },
];
const VAL_DEFS = [
  { k: "val_loss", sub: "lower is better" },
  { k: "val_r@1", sub: "" },
  { k: "val_r@5", sub: "" },
  { k: "val_avg_log_dist_100NN", match: /^val_avg_log_dist_\d+NN$/, sub: "mean log distance to the nearest candidates" },
  { k: "val_judged_r@1", sub: "" },
  { k: "val_judged_r@5", sub: "" },
  { k: "val_judged_ndcg@10", sub: "" },
];
const STEP_BOOKKEEPING = new Set(["step", "epoch", "time", "data_s", "step_s"]);
const EVAL_BOOKKEEPING = new Set(["step", "epoch", "time", "max_mem_gb"]);

// ------------------------------------------------------------------------------------------------------- state
const S = {
  data: null, runs: [], byId: new Map(), varKeys: [], ref: {},
  tab: "runs", sel: [], q: "", st: new Set(STATUSES), f: {}, sort: { key: "vars", desc: false }, agg: false,
  smooth: 5, x: "step", raw: true, hidden: new Set(),
  gb: "", gb2: "", gm: "",
  demo: "", detail: "", live: false,
  queue: null, qq: "", qst: new Set(Q_STATUSES), qclosed: new Set(),
  series: new Map(), inflight: new Map(), colorOf: new Map(), cmpTok: 0,
  openVf: "", lastClick: null, rowOrder: [], updatedAt: 0, error: "", lastHash: "",
};

// ------------------------------------------------------------------------------------------------------- utils
async function api(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    let m = `${r.status}`;
    try { m = (await r.json()).error || m; } catch (e) { /* not json */ }
    throw new Error(m);
  }
  return r.json();
}
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, 1400);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) { /* ignore */ }
    ta.remove();
  }
  toast("Copied " + (text.length > 60 ? text.slice(0, 57) + "…" : text));
}
const COPY_SVG = `<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="5" y="5" width="8.5" height="8.5" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3 10.5V3.8C3 3.1 3.6 2.5 4.3 2.5H10.5" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>`;
const copyBtn = (text, title = "Copy id") => `<button class="copy" data-copy="${esc(text)}" title="${esc(title)}" aria-label="${esc(title)}">${COPY_SVG}</button>`;

// id segments as unbreakable spans (line breaks only between segments); values that differ from the best model in bold
function segSpans(pairs) {
  return pairs.map(([k, v]) => `<span class="sg${S.ref[k] != null && S.ref[k] !== v ? " x" : ""}"><span class="k">${esc(k)}-</span>${esc(v)}</span>`).join("_<wbr>");
}
function parseId(id) {
  const out = [];
  for (const seg of String(id).split("_")) {
    const i = seg.indexOf("-");
    if (i < 0 && out.length) out[out.length - 1][1] += "_" + seg;
    else if (i < 0) out.push([seg, ""]);
    else out.push([seg.slice(0, i), seg.slice(i + 1)]);
  }
  return out;
}
const numLike = (v) => /^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(String(v));
function cmpVal(a, b) {
  if (a === b) return 0;
  if (a == null || a === "") return 1;
  if (b == null || b === "") return -1;
  if (numLike(a) && numLike(b)) return parseFloat(a) - parseFloat(b);
  if (numLike(a) !== numLike(b)) return numLike(a) ? -1 : 1;
  return String(a).localeCompare(String(b));
}
function stat(vals) {
  const v = vals.filter(isNum);
  const n = v.length;
  if (!n) return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN };
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const std = n > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : NaN;
  return { n, mean, std, min: Math.min(...v), max: Math.max(...v) };
}
function direction(m) {
  if (/(^|_)loss$/.test(m)) return "min";
  if (/r@\d+$|ndcg@\d+$/.test(m)) return "max";
  return null;
}
const BASE_ORDER = [/^loss$/, /^r@1$/, /^r@5$/, /^r@\d+$/, /^avg_log_dist_\d+NN$/, /^judged_r@1$/, /^judged_r@5$/, /^judged_/];
function metricRank(base) { const i = BASE_ORDER.findIndex((r) => r.test(base)); return i < 0 ? 99 : i; }
function sortMetrics(keys) {
  return [...keys].sort((a, b) => {
    const pa = a.split("_")[0], pb = b.split("_")[0];
    if (pa !== pb) return pa === "val" ? -1 : pb === "val" ? 1 : pa.localeCompare(pb);
    const ba = a.slice(pa.length + 1), bb = b.slice(pb.length + 1);
    return metricRank(ba) - metricRank(bb) || ba.localeCompare(bb);
  });
}
function fmtDur(sec) {
  if (!isNum(sec)) return "";
  if (sec < 90) return `${Math.round(sec)}s`;
  if (sec < 5400) return `${Math.round(sec / 60)} min`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)} h`;
  return `${(sec / 86400).toFixed(1)} d`;
}
function fmtAgo(sec) {
  if (!isNum(sec)) return "";
  if (sec < 60) return `${Math.max(0, Math.round(sec))}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}
const fmtInt = (v) => (isNum(v) ? Math.round(v).toLocaleString() : "–");
const pill = (st, text) => `<span class="pill ${st.replace(" ", "")}">${esc(text ?? st)}</span>`;

let PAL = null;
function palette() {
  if (!PAL) {
    const cs = getComputedStyle(document.documentElement);
    PAL = Array.from({ length: 8 }, (_, i) => cs.getPropertyValue(`--s${i + 1}`).trim());
  }
  return PAL;
}
// colors follow the entity: a series keeps its slot while it stays on screen; freed slots are reused
function assignColors(keys) {
  const keep = new Set(keys);
  for (const k of [...S.colorOf.keys()]) if (!keep.has(k)) S.colorOf.delete(k);
  const used = new Set(S.colorOf.values());
  let next = 0;
  for (const k of keys) {
    if (S.colorOf.has(k)) continue;
    while (used.has(next)) next++;
    S.colorOf.set(k, next); used.add(next);
  }
}
function colorFor(key) {
  const i = S.colorOf.get(key) ?? 0;
  return { color: palette()[i % 8], dash: DASHES[Math.floor(i / 8) % DASHES.length] };
}
function lineKey(key, w = 16) {
  const { color, dash } = colorFor(key);
  return `<svg width="${w}" height="8" aria-hidden="true"><line x1="0" x2="${w}" y1="4" y2="4" stroke="${color}" stroke-width="3"${dash ? ` stroke-dasharray="${dash.join(" ")}"` : ""}/></svg>`;
}

// ------------------------------------------------------------------------------------------------------- hash
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const list = (k) => (p.get(k) || "").split(",").filter(Boolean);
  S.tab = p.get("tab") || "runs";
  S.sel = list("sel");
  S.q = p.get("q") || "";
  S.st = p.has("st") ? new Set(list("st")) : new Set(STATUSES);
  S.f = {};
  for (const [k, v] of p) if (k.startsWith("f.")) S.f[k.slice(2)] = new Set(v.split(","));
  const so = p.get("sort") || "vars";
  S.sort = so.startsWith("-") ? { key: so.slice(1), desc: true } : { key: so, desc: false };
  S.agg = p.get("agg") === "1";
  S.smooth = p.has("sm") ? Math.max(0, Math.min(SMOOTH.length - 1, +p.get("sm") || 0)) : 5;
  S.x = ["step", "epoch", "hours"].includes(p.get("x")) ? p.get("x") : "step";
  S.raw = p.get("raw") !== "0";
  S.hidden = new Set(list("hide"));
  S.gb = p.get("gb") || ""; S.gb2 = p.get("gb2") || ""; S.gm = p.get("gm") || "";
  S.demo = p.get("demo") || "";
  S.detail = p.get("run") || "";
  S.live = p.get("live") === "1";
  S.qq = p.get("qq") || "";
  S.qst = p.has("qst") ? new Set(list("qst")) : new Set(Q_STATUSES);
}
function writeHash() {
  const parts = [];
  const put = (k, v) => { if (v !== "" && v != null) parts.push(`${k}=${encodeURIComponent(v).replace(/%2C/g, ",").replace(/%40/g, "@")}`); };
  put("tab", S.tab === "runs" ? "" : S.tab);
  put("sel", S.sel.join(","));
  put("q", S.q);
  if (S.st.size !== STATUSES.length) put("st", [...S.st].join(",") || "none");
  for (const [k, set] of Object.entries(S.f)) if (set.size) put("f." + k, [...set].join(","));
  if (S.sort.key !== "vars" || S.sort.desc) put("sort", (S.sort.desc ? "-" : "") + S.sort.key);
  if (S.agg) put("agg", "1");
  if (S.smooth !== 5) put("sm", String(S.smooth));
  if (S.x !== "step") put("x", S.x);
  if (!S.raw) put("raw", "0");
  put("hide", [...S.hidden].join(","));
  put("gb", S.gb); put("gb2", S.gb2); put("gm", S.gm);
  put("demo", S.demo);
  put("run", S.detail);
  if (S.live) put("live", "1");
  put("qq", S.qq);
  if (S.qst.size !== Q_STATUSES.length) put("qst", [...S.qst].join(",") || "none");
  const h = "#" + parts.join("&");
  if (h !== location.hash) history.replaceState(null, "", h === "#" ? location.pathname + location.search : h);
  S.lastHash = location.hash;
}
function update() { writeHash(); render(); }

// ------------------------------------------------------------------------------------------------------- data
function setData(d) {
  S.data = d;
  S.runs = d.runs;
  S.byId = new Map(d.runs.map((r) => [r.id, r]));
  S.ref = Object.fromEntries(parseId(d.reference_id || ""));
  const keys = [];
  for (const k of Object.keys(S.ref)) if (d.runs.some((r) => k in r.vars)) keys.push(k);
  for (const r of d.runs) for (const k of Object.keys(r.vars)) if (!keys.includes(k)) keys.push(k);
  S.varKeys = keys;
}
async function refresh() {
  $("#refresh").classList.add("spin");
  try {
    setData(await api("/api/runs"));
    S.error = "";
    if (S.tab === "queue") S.queue = await api("/api/queue");
    else S.queue = null;
  } catch (e) {
    S.error = "Server unreachable: " + e.message;
  }
  S.updatedAt = Date.now();
  $("#refresh").classList.remove("spin");
  render();
  if (S.detail && !$("#drawer").hidden) loadDetail(S.detail, true);
}

const groupKey = (r) => Object.entries(r.vars).filter(([k]) => k !== "seed").map(([k, v]) => `${k}-${v}`).join("_");

function passes(r, skipKey) {
  if (S.st.size && !S.st.has(r.status)) return false;
  for (const [k, set] of Object.entries(S.f)) if (k !== skipKey && set.size && !set.has(r.vars[k] ?? "")) return false;
  if (S.q) {
    const id = r.id.toLowerCase();
    for (const t of S.q.toLowerCase().split(/\s+/).filter(Boolean)) {
      if (t.startsWith("!") ? t.length > 1 && id.includes(t.slice(1)) : !id.includes(t)) return false;
    }
  }
  return true;
}
const filteredRuns = () => S.runs.filter((r) => passes(r));

function rowsOf(runs) {
  if (!S.agg) return runs.map((r) => ({ key: r.id, id: r.id, runs: [r], vars: r.vars }));
  const m = new Map();
  for (const r of runs) {
    const k = groupKey(r);
    if (!m.has(k)) m.set(k, { key: k, id: k, runs: [], vars: Object.fromEntries(Object.entries(r.vars).filter(([kk]) => kk !== "seed")) });
    m.get(k).runs.push(r);
  }
  for (const g of m.values()) g.runs.sort((a, b) => cmpVal(a.vars.seed, b.vars.seed));
  return [...m.values()];
}
function varying(rows, keys = S.varKeys) {
  const out = [], fixed = [];
  for (const k of keys) {
    if (S.agg && k === "seed") continue;
    const vals = new Set(rows.map((r) => r.vars[k] ?? ""));
    (vals.size > 1 ? out : fixed).push(k);
  }
  return { varying: out, fixed };
}
function knnKey(prefix) {
  for (const r of S.runs) for (const k of Object.keys(r.final || {})) if (k.startsWith(prefix + "_avg_log_dist_") && KNN.test(k)) return k;
  return `${prefix}_avg_log_dist_100NN`;
}
function runMetric(r, key) {
  if (key === "best_val_loss") return r.best_val_loss;
  if (key === "train_minutes" || key === "best_step") return r.final ? r.final[key] : null;
  return r.final ? r.final[key] : null;
}
function progressOf(r) {
  if (r.status === "done") return 1;
  return isNum(r.last_step) && isNum(r.total_steps) && r.total_steps > 0 ? Math.min(1, r.last_step / r.total_steps) : 0;
}

// ------------------------------------------------------------------------------------------------------- render
function render() {
  if (!S.data) return;
  const n = S.runs.length, cnt = (s) => S.runs.filter((r) => r.status === s).length;
  $("#n-runs").textContent = n;
  const nsel = S.sel.filter((id) => S.byId.has(id)).length;
  $("#n-sel").textContent = nsel ? nsel : "";
  $("#health").innerHTML = S.error ? `<span class="err">${esc(S.error)}</span>`
    : `${n} runs` + (cnt("running") ? ` · ${pill("running", cnt("running") + " running")}` : "") + (cnt("stale") ? ` · ${pill("stale", cnt("stale") + " stale")}` : "");
  $("#live").checked = S.live;
  tickUpdated();
  for (const b of $$(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === S.tab));
  for (const v of $$(".view")) v.hidden = v.id !== "view-" + S.tab;
  if (S.tab === "runs") renderRuns();
  else if (S.tab === "compare") renderCompare();
  else if (S.tab === "group") renderGroup();
  else if (S.tab === "queue") renderQueue();
  else if (S.tab === "demo") renderDemo();
}
function tickUpdated() {
  $("#updated").textContent = S.updatedAt ? "updated " + fmtAgo((Date.now() - S.updatedAt) / 1000) : "";
}

// ======================================================================================================= runs view
const RUN_METRICS = () => [
  { key: "best_val_loss", label: "loss", sub: "best so far", group: "val" },
  { key: "test_loss", label: "loss", sub: "lower is better", group: "test" },
  { key: "test_r@1", label: "r@1", group: "test" },
  { key: "test_r@5", label: "r@5", group: "test" },
  { key: knnKey("test"), label: knnKey("test").replace("test_avg_log_dist_", "avg_log_<wbr>dist_"), group: "test" },
  { key: "test_judged_r@5", label: "judged_<wbr>r@5", group: "test" },
];
// the segments of an id that differ from the best model (the reference id), in id format
function diffPairs(vars) {
  return Object.entries(vars).filter(([k, v]) => S.ref[k] == null || S.ref[k] !== v);
}

function renderRuns() {
  if (document.activeElement !== $("#q")) $("#q").value = S.q;
  $("#agg").checked = S.agg;
  renderStatusSeg();
  renderVarFilters();
  const runs = filteredRuns();
  const rows = rowsOf(runs);
  const { varying: vk, fixed } = varying(rows);
  assignColors(compareSeries().map((s) => s.key));

  // fixed line
  const fx = fixed.filter((k) => rows.length).map((k) => `<span title="${esc(VAR_NAMES[k] || k)}">${esc(k)}-${esc(rows[0].vars[k] ?? "")}</span>`);
  $("#fixed").innerHTML = rows.length ? `<b>${runs.length} of ${S.runs.length} runs${S.agg ? ` in ${rows.length} configs` : ""}${fx.length ? " · fixed:" : ""}</b>${fx.join("")}` : "";

  // sort
  const mets = RUN_METRICS();
  const sortVal = (row, key) => {
    if (key === "id") return row.id;
    if (key === "status") return Math.min(...row.runs.map((r) => STATUSES.indexOf(r.status)));
    if (key === "progress") return stat(row.runs.map(progressOf)).mean;
    if (key === "n") return row.runs.length;
    if (key.startsWith("v:")) return row.vars[key.slice(2)];
    if (key.startsWith("m:")) return stat(row.runs.map((r) => runMetric(r, key.slice(2)))).mean;
    return null;
  };
  const { key: sk, desc } = S.sort;
  if (sk === "vars") rows.sort((a, b) => { for (const k of S.varKeys) { const c = cmpVal(a.vars[k], b.vars[k]); if (c) return c; } return 0; });
  else rows.sort((a, b) => {
    const va = sortVal(a, sk), vb = sortVal(b, sk);
    const na = va == null || (typeof va === "number" && !isFinite(va)), nb = vb == null || (typeof vb === "number" && !isFinite(vb));
    if (na || nb) return na === nb ? 0 : na ? 1 : -1;           // missing values last, either direction
    const c = typeof va === "number" && typeof vb === "number" ? va - vb : cmpVal(va, vb);
    return desc ? -c : c;
  });
  S.rowOrder = rows.map((r) => r.key);

  // best per metric column (among shown rows)
  const best = {};
  for (const m of mets) {
    const dir = direction(m.key === "best_val_loss" ? "val_loss" : m.key);
    if (!dir) continue;
    let bv = null;
    for (const row of rows) {
      const v = stat(row.runs.map((r) => runMetric(r, m.key))).mean;
      if (isNum(v) && (bv == null || (dir === "min" ? v < bv : v > bv))) bv = v;
    }
    best[m.key] = bv;
  }

  const selSet = new Set(S.sel);
  const allSel = rows.length && rows.every((row) => row.runs.every((r) => selSet.has(r.id)));
  const someSel = rows.some((row) => row.runs.some((r) => selSet.has(r.id)));
  const th = (key, label, cls = "", sub = "", title = "") => {
    const on = S.sort.key === key;
    return `<th class="sortable ${cls}" data-sort="${esc(key)}" title="${esc(title)}">${label}${on ? `<span class="arrow">${S.sort.desc ? "▼" : "▲"}</span>` : ""}${sub ? `<span class="sub">${sub}</span>` : ""}</th>`;
  };
  const nLead = 3 + vk.length + (S.agg ? 1 : 0) + 2;
  let h = `<thead><tr class="groups"><th class="cb stick"></th><th class="sw stick"></th><th class="idh stick"></th><th colspan="${nLead - 3}"></th><th class="num">val</th><th colspan="${mets.length - 1}" class="num">test (final.json)</th></tr>`;
  h += `<tr><th class="cb stick"><input type="checkbox" id="sel-all" aria-label="Select all shown"${allSel ? " checked" : ""}></th><th class="sw stick"></th>`;
  h += th("id", S.agg ? "config" : "run", "idh stick", "vs best model", "Segments of the id that differ from the best model; sort by id");
  for (const k of vk) h += th("v:" + k, esc(k), "", "", VAR_NAMES[k] || k);
  if (S.agg) h += th("n", "n", "num", "seeds", "Runs (seeds) per config");
  h += th("status", "status") + th("progress", "progress");
  for (const m of mets) h += th("m:" + m.key, m.label, "num wrap", esc(m.sub || ""), m.key);
  h += `</tr></thead><tbody>`;
  for (const row of rows) {
    const nsel = row.runs.filter((r) => selSet.has(r.id)).length;
    const isSel = nsel === row.runs.length;
    const skey = row.key;
    const r0 = row.runs[0];
    const dp = diffPairs(row.vars);
    const idHtml = dp.length ? segSpans(dp) : `<span class="ref">best model${S.agg ? "" : ""}</span>`;
    const warn = row.runs.flatMap((r) => r.warnings || []);
    const legacy = row.runs.some((r) => r.legacy) ? `<span class="legacy" title="imported pre-refactor run${r0.legacy_name ? ": " + esc(r0.legacy_name) : ""}; steps logged every 50">legacy</span>` : "";
    h += `<tr data-key="${esc(skey)}" class="${isSel ? "sel" : ""}">`;
    h += `<td class="cb stick"><input type="checkbox" data-row="${esc(skey)}" aria-label="Select"${isSel ? " checked" : ""}${nsel && !isSel ? ' data-ind="1"' : ""}></td>`;
    h += `<td class="sw stick">${nsel ? lineKey(S.agg ? skey : r0.id, 14) : ""}</td>`;
    h += `<td class="idh stick"><div class="idcell"><button class="idlink" data-open="${esc(r0.id)}" title="${esc(S.agg ? row.runs.map((r) => r.id).join("\n") : row.id)}">${idHtml}</button>`
      + copyBtn(S.agg ? row.runs.map((r) => r.id).join("\n") : row.id, S.agg ? "Copy the runs' ids" : "Copy id")
      + legacy + (warn.length ? `<span class="warnmark" title="${esc(warn.join("\n"))}">⚠</span>` : "") + `</div></td>`;
    for (const k of vk) h += `<td class="v">${esc(row.vars[k] ?? "")}</td>`;
    if (S.agg) h += `<td class="num" title="seeds ${esc(row.runs.map((r) => r.vars.seed).join(", "))}">${row.runs.length}</td>`;
    h += `<td>${statusCell(row.runs)}</td><td>${progressCell(row.runs)}</td>`;
    for (const m of mets) h += metricCell(row.runs.map((r) => runMetric(r, m.key)), best[m.key], m.key === "best_val_loss" ? row.runs.some((r) => r.status !== "done") : false);
    h += `</tr>`;
  }
  h += `</tbody>`;
  $("#runs-table").innerHTML = h;
  for (const cb of $$("#runs-table input[data-ind]")) cb.indeterminate = true;
  const sa = $("#sel-all");
  if (sa) sa.indeterminate = !allSel && someSel;
  const empty = $("#runs-empty");
  empty.hidden = rows.length > 0;
  empty.textContent = S.runs.length ? "No runs match the filters." : `No runs in ${S.data.results_root} (folders holding config.json).`;
  const ns = S.sel.filter((id) => S.byId.has(id)).length;
  $("#selinfo").textContent = ns ? `${ns} selected` : "";
  $("#sel-clear").hidden = !ns;
  $("#go-compare").disabled = !ns;
  $("#go-compare").textContent = ns ? `Compare ${ns}` : "Compare";
}
function statusCell(runs) {
  if (runs.length === 1) return pill(runs[0].status);
  return STATUSES.map((s) => [s, runs.filter((r) => r.status === s).length]).filter(([, c]) => c).map(([s, c]) => pill(s, `${c} ${s}`)).join(" ");
}
function progressCell(runs) {
  const now = S.data.now;
  if (runs.length > 1) {
    const p = stat(runs.map(progressOf)).mean;
    const st = runs.some((r) => r.status === "running") ? "running" : runs.some((r) => r.status === "stale") ? "stale" : "done";
    return `<div class="prog"><span class="bar ${st}"><span style="width:${(p * 100).toFixed(1)}%"></span></span><span class="t">${runs.filter((r) => r.status === "done").length}/${runs.length} done</span></div>`;
  }
  const r = runs[0], p = progressOf(r);
  let t = "";
  if (r.status === "done") t = `${fmtInt(r.last_step ?? r.total_steps)}<small>${r.final && isNum(r.final.train_minutes) ? " · " + fmtDur(r.final.train_minutes * 60) : ""}</small>`;
  else {
    t = `${fmtInt(r.last_step)}<small> / ${fmtInt(r.total_steps)}</small>`;
    if (r.status === "running" && isNum(r.last_time) && r.last_step > 0 && isNum(r.total_steps)) t += ` <small>· ETA ${fmtDur((r.total_steps - r.last_step) * r.last_time / r.last_step)}</small>`;
    if (r.status === "stale" && isNum(r.last_update)) t += ` <small>· ${fmtAgo(now - r.last_update)}</small>`;
  }
  return `<div class="prog"><span class="bar ${r.status}"><span style="width:${(p * 100).toFixed(1)}%"></span></span><span class="t">${t}</span></div>`;
}
function metricCell(vals, best, soFar) {
  const s = stat(vals);
  if (!s.n) return `<td class="num na">–</td>`;
  const isBest = isNum(best) && s.mean === best;
  const inner = s.n > 1 || vals.length > 1
    ? `${fmt(s.mean)}${isNum(s.std) ? ` <span class="pm">± ${fmt(s.std)}</span>` : ""}<span class="nsub">${s.n}</span>`
    : fmt(s.mean);
  return `<td class="num${isBest ? " best" : ""}"${soFar ? ' title="includes unfinished runs (best so far)"' : ""}>${soFar ? `<i>${inner}</i>` : inner}</td>`;
}

function renderStatusSeg() {
  const counts = Object.fromEntries(STATUSES.map((s) => [s, S.runs.filter((r) => r.status === s).length]));
  $("#stfilter").innerHTML = STATUSES.map((s) => `<button data-st="${s}" aria-pressed="${S.st.has(s)}">${pill(s)}<span class="n">${counts[s]}</span></button>`).join("");
}
function renderVarFilters() {
  const keys = S.varKeys.filter((k) => new Set(S.runs.map((r) => r.vars[k] ?? "")).size > 1);
  let h = "";
  for (const k of keys) {
    const vals = [...new Set(S.runs.map((r) => r.vars[k] ?? ""))].sort(cmpVal);
    const facet = S.runs.filter((r) => passes(r, k));
    const set = S.f[k] || new Set();
    const on = set.size > 0;
    const label = on ? [...set].sort(cmpVal).join(", ") : "all";
    h += `<details class="vf${on ? " on" : ""}" data-k="${esc(k)}"${S.openVf === k ? " open" : ""}><summary><span class="k">${esc(k)}</span><span class="v">${esc(label)}</span></summary>
      <div class="pop"><div class="name">${esc(VAR_NAMES[k] || k)}</div>`;
    for (const v of vals) {
      const c = facet.filter((r) => (r.vars[k] ?? "") === v).length;
      h += `<label><input type="checkbox" data-fk="${esc(k)}" data-fv="${esc(v)}"${set.has(v) ? " checked" : ""}>${esc(v === "" ? "(none)" : v)}<button class="only" data-only="${esc(v)}" data-fk="${esc(k)}">only</button><span class="c">${c}</span></label>`;
    }
    h += `<div class="acts"><button class="btn small" data-fclear="${esc(k)}">All values</button></div></div></details>`;
  }
  if (Object.values(S.f).some((s) => s.size) || S.q || S.st.size !== STATUSES.length) h += `<button class="btn small" id="clear-filters">Clear filters</button>`;
  $("#varfilters").innerHTML = h;
}

function toggleRows(keys, on) {
  const rows = rowsOf(filteredRuns());
  const ids = rows.filter((r) => keys.includes(r.key)).flatMap((r) => r.runs.map((x) => x.id));
  if (on) for (const id of ids) { if (!S.sel.includes(id)) S.sel.push(id); }
  else S.sel = S.sel.filter((id) => !ids.includes(id));
}

// ======================================================================================================= compare
function compareSeries() {
  const runs = S.sel.map((id) => S.byId.get(id)).filter(Boolean);
  let ser;
  if (!S.agg) ser = runs.map((r) => ({ key: r.id, runs: [r], vars: r.vars }));
  else {
    const m = new Map();
    for (const r of runs) {
      const k = groupKey(r);
      if (!m.has(k)) m.set(k, { key: k, runs: [], vars: Object.fromEntries(Object.entries(r.vars).filter(([kk]) => kk !== "seed")) });
      m.get(k).runs.push(r);
    }
    ser = [...m.values()];
  }
  // labels: only the variables that differ between the series on screen
  const keys = S.varKeys.filter((k) => !(S.agg && k === "seed"));
  const vk = keys.filter((k) => new Set(ser.map((s) => s.vars[k] ?? "")).size > 1);
  for (const s of ser) {
    // legend: the varying variables (values that differ from the best model in bold); tooltips: just the differences
    s.pairs = vk.length ? vk.map((k) => [k, s.vars[k] ?? ""]) : diffPairs(s.vars);
    const diff = diffPairs(s.vars).filter(([k]) => vk.includes(k) || !vk.length);
    s.label = diff.length ? diff.map(([k, v]) => `${k}-${v}`).join(" ") : "best model";
    s.html = s.pairs.length ? segSpans(s.pairs) : `<span class="ref">best model</span>`;
  }
  return ser;
}

let GROUP = null;
const CHARTS = new Map();   // name -> {chart, kind, def}
function initCharts() {
  GROUP = new Charts.Group();
  GROUP.onRange = (r) => { $("#zoom-reset").hidden = !r; };
  for (const def of TRAIN_DEFS) ensureChart(def, "steps");
  for (const def of VAL_DEFS) ensureChart(def, "evals");
}
function ensureChart(def, kind) {
  if (CHARTS.has(def.k)) return CHARTS.get(def.k);
  const host = document.createElement("div");
  $(kind === "steps" ? "#charts-train" : "#charts-val").appendChild(host);
  const chart = new Charts.Chart(host, { title: def.k, sub: def.sub || "", xFormat: xFormatter(), xName: S.x });
  GROUP.add(chart);
  const c = { chart, kind, def, host };
  CHARTS.set(def.k, c);
  return c;
}
function xFormatter() {
  if (S.x === "step") return (v, st, full) => full ? Math.round(v).toLocaleString()
    : Math.abs(v) >= 1000 ? (v / 1000).toFixed(st >= 1000 || !st ? 0 : st >= 100 ? 1 : 2).replace(/\.0+$/, "") + "k" : String(Math.round(v));
  if (S.x === "epoch") return (v, st, full) => full ? v.toFixed(2) : v.toFixed(st && st < 1 ? 1 : 0);
  return (v, st, full) => full ? v.toFixed(2) + " h" : v.toFixed(st && st < 1 ? 1 : 0) + "h";
}
function colOf(cols, def) {
  if (cols[def.k]) return def.k;
  if (def.match) for (const k of Object.keys(cols)) if (def.match.test(k)) return k;
  return null;
}
function xOf(r, cols) {
  if (S.x === "step") return cols.step || null;
  if (S.x === "hours") return cols.time ? cols.time.map((v) => v / 3600) : null;
  if (r.steps_per_epoch && cols.step) return cols.step.map((v) => v / r.steps_per_epoch);
  return cols.epoch || null;
}
function ema(steps, y, w) {
  const n = y.length, out = new Float64Array(n);
  let num = 0, den = 0, prev = NaN;
  for (let i = 0; i < n; i++) {
    const v = y[i];
    if (!isFinite(v)) { out[i] = NaN; continue; }
    let gap = isFinite(prev) ? steps[i] - prev : (i + 1 < n ? steps[i + 1] - steps[i] : 1);
    if (!(gap > 0)) gap = 1;
    const d = Math.exp(-gap / w);                     // decay per step, so runs logged every 50 steps smooth alike
    num = d * num + (1 - d) * v; den = d * den + (1 - d);
    out[i] = num / den; prev = steps[i];
  }
  return out;
}
function interp(xs, ys, g) {
  const n = xs.length;
  if (!n || g < xs[0] || g > xs[n - 1]) return NaN;
  const i = Charts.lowerBound(xs, g);
  if (xs[i] === g) return ys[i];
  const a = ys[i - 1], b = ys[i];
  if (!isFinite(a)) return b;
  if (!isFinite(b)) return a;
  return a + (b - a) * (g - xs[i - 1]) / (xs[i] - xs[i - 1]);
}
// mean and min-max band over runs, on the step grid of the longest run
function aggCurve(parts) {
  let ref = parts[0];
  for (const p of parts) {
    const a = p.steps[p.steps.length - 1] ?? -Infinity, b = ref.steps[ref.steps.length - 1] ?? -Infinity;
    if (a > b || (a === b && p.steps.length > ref.steps.length)) ref = p;
  }
  const n = ref.steps.length, mean = new Float64Array(n), lo = new Float64Array(n), hi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = ref.steps[i];
    let s = 0, c = 0, mn = Infinity, mx = -Infinity;
    for (const p of parts) {
      const v = p === ref ? p.y[i] : interp(p.steps, p.y, g);
      if (!isFinite(v)) continue;
      s += v; c++; if (v < mn) mn = v; if (v > mx) mx = v;
    }
    mean[i] = c ? s / c : NaN; lo[i] = c ? mn : NaN; hi[i] = c ? mx : NaN;
  }
  return { x: ref.x, y: mean, band: { lo, hi } };
}
function toCols(t) {
  if (!t || !t.columns) return {};
  const out = {};
  for (const [k, a] of Object.entries(t.columns)) out[k] = Float64Array.from(a, (v) => (v == null ? NaN : v));
  return out;
}
async function ensureSeries(runs) {
  await Promise.all(runs.map((r) => {
    const c = S.series.get(r.id);
    if (c && c.version === r.version) return null;
    const key = r.id + "@" + r.version;
    if (S.inflight.has(key)) return S.inflight.get(key);
    const p = api(`/api/series?id=${encodeURIComponent(r.id)}&max_points=${MAX_POINTS}`).then((res) => {
      S.series.set(r.id, { version: r.version, steps: toCols(res.steps), evals: toCols(res.evals),
        bucket: res.steps ? res.steps.bucket : 1, rows: res.steps ? res.steps.rows : 0 });
    }).catch((e) => { S.series.set(r.id, { version: r.version, steps: {}, evals: {}, error: e.message }); })
      .finally(() => S.inflight.delete(key));
    S.inflight.set(key, p);
    return p;
  }));
}

function renderLegend(ser) {
  $("#legend").innerHTML = ser.map((s) => {
    const off = S.hidden.has(s.key);
    const st = s.runs.length === 1 ? pill(s.runs[0].status) : statusCell(s.runs);
    const n = s.runs.length > 1 ? `<span class="meta">n=${s.runs.length} (seeds ${esc(s.runs.map((r) => r.vars.seed).join(","))})</span>` : "";
    return `<span class="lg${off ? " off" : ""}" data-key="${esc(s.key)}"><span class="key" data-hide="${esc(s.key)}" title="${off ? "Show" : "Hide"}">${lineKey(s.key, 18)}</span>`
      + `<span class="lab" title="${esc(s.runs.map((r) => r.id).join("\n"))}">${s.html}</span><span class="meta">${st}</span>${n}`
      + (s.runs.length === 1 ? `<button data-open="${esc(s.runs[0].id)}" title="Details">ⓘ</button>` : "")
      + `<button data-unsel="${esc(s.key)}" title="Remove from comparison" aria-label="Remove">✕</button></span>`;
  }).join("");
}

async function renderCompare() {
  $("#smooth").value = S.smooth;
  $("#smooth-val").textContent = SMOOTH[S.smooth] ? `EMA ${SMOOTH[S.smooth].toLocaleString()} steps` : "off";
  $("#xaxis").value = S.x;
  $("#raw").checked = S.raw;
  $("#agg2").checked = S.agg;
  const ser = compareSeries();
  assignColors(ser.map((s) => s.key));
  renderLegend(ser);
  $("#compare-empty").hidden = ser.length > 0;
  $("#compare-body").hidden = !ser.length;
  if (!ser.length) return;
  const tok = ++S.cmpTok;
  const runs = ser.flatMap((s) => s.runs);
  await ensureSeries(runs);
  if (tok !== S.cmpTok || S.tab !== "compare") return;

  // charts for columns beyond the standard set (e.g. a new eval metric), and beta when it is ever nonzero
  const extra = { steps: new Set(), evals: new Set() };
  for (const r of runs) {
    const d = S.series.get(r.id);
    if (!d) continue;
    for (const k of Object.keys(d.steps)) if (!STEP_BOOKKEEPING.has(k) && !TRAIN_DEFS.some((x) => x.k === k) && (k !== "beta" || d.steps.beta.some((v) => v > 0))) extra.steps.add(k);
    for (const k of Object.keys(d.evals)) if (!EVAL_BOOKKEEPING.has(k) && !VAL_DEFS.some((x) => colOf({ [k]: 1 }, x))) extra.evals.add(k);
  }
  for (const kind of ["steps", "evals"]) for (const k of [...extra[kind]].sort()) ensureChart({ k, sub: "", smooth: kind === "steps" }, kind);
  for (const [k, c] of CHARTS) {
    const std = (c.kind === "steps" ? TRAIN_DEFS : VAL_DEFS).some((x) => x.k === k);
    if (!std && !extra[c.kind].has(k)) { c.chart.destroy(); c.host.remove(); CHARTS.delete(k); }
  }

  const w = SMOOTH[S.smooth];
  const xf = xFormatter();
  for (const [, c] of CHARTS) {
    c.chart.opt.xFormat = xf; c.chart.opt.xName = S.x;
    const out = [];
    let colName = null;
    for (const s of ser) {
      if (S.hidden.has(s.key)) continue;
      const { color, dash } = colorFor(s.key);
      const parts = [];
      for (const r of s.runs) {
        const d = S.series.get(r.id);
        if (!d) continue;
        const cols = d[c.kind];
        const col = colOf(cols, c.def);
        if (!col) continue;
        colName = colName || col;
        const x = xOf(r, cols);
        if (!x) continue;
        const steps = cols.step || Float64Array.from(x, (_, i) => i);
        const raw = cols[col];
        const sm = c.def.smooth && w ? ema(steps, raw, w) : raw;
        parts.push({ r, steps, x, y: sm, raw });
      }
      if (!parts.length) continue;
      if (parts.length === 1) {
        const p = parts[0];
        const item = { id: s.key, label: s.label, color, dash, x: p.x, y: p.y, points: c.kind === "evals",
          raw: c.def.smooth && w && S.raw ? { x: p.x, y: p.raw } : null };
        const bs = p.r.final && p.r.final.best_step;
        if (c.kind === "evals" && isNum(bs)) {
          const i = Charts.nearest(p.steps, bs);
          if (i >= 0 && p.steps[i] === bs) item.markers = [{ x: p.x[i], y: p.y[i] }];
        }
        out.push(item);
      } else {
        const a = aggCurve(parts);
        out.push({ id: s.key, label: s.label, color, dash, x: a.x, y: a.y, band: a.band, points: c.kind === "evals" });
      }
    }
    if (colName && colName !== c.def.k) c.chart.setTitle(colName);
    c.chart.setSeries(out);
  }
  const buckets = runs.map((r) => (S.series.get(r.id) || {}).bucket || 1);
  const maxB = Math.max(...buckets);
  $("#train-note").textContent = [w ? `EMA over ~${w.toLocaleString()} steps, faint = unsmoothed` : "no smoothing",
    maxB > 1 ? `long runs shown as means of ${maxB} consecutive steps` : "", S.agg ? "merged seeds: mean line, min–max band" : ""].filter(Boolean).join(" · ");
  renderFinalTable(ser);
}

function renderFinalTable(ser) {
  const keys = new Set();
  for (const s of ser) for (const r of s.runs) for (const [k, v] of Object.entries(r.final || {})) if ((k.startsWith("val_") || k.startsWith("test_")) && typeof v === "number") keys.add(k);
  const cols = sortMetrics(keys);
  const vcols = cols.filter((k) => k.startsWith("val_")), tcols = cols.filter((k) => k.startsWith("test_"));
  const all = ["best_step", ...vcols, ...tcols, "train_minutes"];
  const val = (r, k) => {
    if (r.final) return r.final[k];
    if (k === "val_loss") return r.best_val_loss;                // unfinished: best so far
    return null;
  };
  const best = {};
  for (const k of all) {
    const dir = direction(k);
    if (!dir) continue;
    // rank finished runs only (an unfinished run's best-so-far val_loss is shown, in italics, but not ranked)
    const means = ser.map((s) => stat(s.runs.map((r) => (r.final ? r.final[k] : null))).mean).filter(isNum);
    if (means.length > 1) best[k] = dir === "min" ? Math.min(...means) : Math.max(...means);
  }
  const head = (k) => {
    const base = k.replace(/^(val|test)_/, "");
    return `<th class="num" title="${esc(k)}">${esc(base)}${direction(k) === "min" ? '<span class="sub">lower is better</span>' : ""}</th>`;
  };
  let h = `<thead><tr class="groups"><th></th><th></th>${vcols.length ? `<th colspan="${vcols.length}" class="num">validation (chosen checkpoint)</th>` : ""}${tcols.length ? `<th colspan="${tcols.length}" class="num">test</th>` : ""}<th></th></tr>`;
  h += `<tr><th>${S.agg ? "config" : "run"}</th><th class="num">best_step</th>${vcols.map(head).join("")}${tcols.map(head).join("")}<th class="num">train time</th></tr></thead><tbody>`;
  for (const s of ser) {
    h += `<tr><td class="lab"><div class="idcell">${lineKey(s.key, 14)}<span title="${esc(s.runs.map((r) => r.id).join("\n"))}">${s.html}${s.runs.length > 1 ? `<span class="nsub">n=${s.runs.length}</span>` : ""}</span></div></td>`;
    const bsv = stat(s.runs.map((r) => (r.final ? r.final.best_step : null)));
    h += `<td class="num">${bsv.n ? (bsv.n > 1 ? `${fmtInt(bsv.mean)}<span class="nsub">${bsv.n}</span>` : fmtInt(bsv.mean)) : "–"}</td>`;
    for (const k of [...vcols, ...tcols]) {
      const unfinished = k === "val_loss" && s.runs.some((r) => !r.final);
      h += metricCell(s.runs.map((r) => val(r, k)), best[k], unfinished);
    }
    const tm = stat(s.runs.map((r) => (r.final ? r.final.train_minutes : null)));
    h += `<td class="num">${tm.n ? fmtDur(tm.mean * 60) : "–"}</td></tr>`;
  }
  h += `</tbody>`;
  $("#final-table").innerHTML = h;
}

// ======================================================================================================= group by
function renderGroup() {
  const runs = filteredRuns();
  const keys = S.varKeys.filter((k) => new Set(runs.map((r) => r.vars[k] ?? "")).size > 1);
  // the choices in S (and the URL) survive a filter that temporarily hides them; fall back locally
  const gb = keys.includes(S.gb) ? S.gb : keys.find((k) => k !== "seed") || keys[0] || "";
  const gb2 = S.gb2 && keys.includes(S.gb2) && S.gb2 !== gb ? S.gb2 : "";
  const mkeys = new Set(["best_val_loss"]);
  for (const r of runs) for (const [k, v] of Object.entries(r.final || {})) if ((k.startsWith("val_") || k.startsWith("test_")) && typeof v === "number") mkeys.add(k);
  const metrics = ["best_val_loss", ...sortMetrics([...mkeys].filter((k) => k !== "best_val_loss")).sort((a, b) => (a.startsWith("test_") ? 0 : 1) - (b.startsWith("test_") ? 0 : 1)), "train_minutes"];
  const gm = metrics.includes(S.gm) ? S.gm : metrics.includes("test_loss") ? "test_loss" : "best_val_loss";
  $("#gb").innerHTML = keys.length ? keys.map((k) => `<option value="${esc(k)}"${k === gb ? " selected" : ""}>${esc(k)} · ${esc(VAR_NAMES[k] || k)}</option>`).join("") : `<option value="">(nothing varies)</option>`;
  $("#gb2").innerHTML = `<option value="">—</option>` + keys.filter((k) => k !== gb).map((k) => `<option value="${esc(k)}"${k === gb2 ? " selected" : ""}>${esc(k)}</option>`).join("");
  $("#gm").innerHTML = metrics.map((k) => `<option value="${esc(k)}"${k === gm ? " selected" : ""}>${esc(k === "best_val_loss" ? "val_loss (best so far)" : k)}</option>`).join("");
  $("#group-src").textContent = `${runs.length} runs after the Runs filters`;
  const body = $("#group-body");
  if (!gb) { body.innerHTML = `<p class="empty">No variable varies among the ${runs.length} filtered runs. Loosen the filters in Runs.</p>`; return; }
  const dir = direction(gm === "best_val_loss" ? "val_loss" : gm);
  const mval = (r) => (gm === "best_val_loss" ? r.best_val_loss : r.final ? r.final[gm] : null);
  const withM = runs.filter((r) => isNum(mval(r)));
  const others = keys.filter((k) => k !== gb && k !== gb2 && k !== "seed");
  let h = `<p class="gnote">${esc(gm)} by <code>${esc(gb)}</code>${gb2 ? ` × <code>${esc(gb2)}</code>` : ""}: mean ± std over runs (seeds count as runs)${dir ? `, ${dir === "min" ? "lower" : "higher"} is better` : ""}. `
    + `${runs.length - withM.length ? `${runs.length - withM.length} filtered runs have no ${esc(gm)} yet. ` : ""}`
    + (others.length ? `<span class="err">Also varying: ${others.map((k) => `<code>${esc(k)}</code> (${new Set(withM.map((r) => r.vars[k])).size})`).join(", ")}</span>, so groups mix those settings; narrow them with the Runs filters to isolate one ablation.` : "All other variables are fixed, so this reads as a clean ablation.")
    + `</p>`;
  const v1 = [...new Set(runs.map((r) => r.vars[gb] ?? ""))].sort(cmpVal);
  if (!gb2) {
    const groups = v1.map((v) => { const rs = withM.filter((r) => (r.vars[gb] ?? "") === v); return { v, rs, s: stat(rs.map(mval)) }; });
    const allv = withM.map(mval);
    let lo = Math.min(...allv), hi = Math.max(...allv);
    if (!(hi > lo)) { lo -= 1e-3; hi += 1e-3; }
    const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
    const means = groups.map((g) => g.s.mean).filter(isNum);
    const bestV = dir && means.length > 1 ? (dir === "min" ? Math.min(...means) : Math.max(...means)) : null;
    h += `<div class="tablewrap"><table class="grid"><thead><tr><th>${esc(gb)}</th><th class="num">n</th><th class="num">mean ± std</th><th class="num">min</th><th class="num">max</th><th>runs <span class="sub">${fmt(lo)} … ${fmt(hi)} · bar = mean ± std · click a dot for the run</span></th></tr></thead><tbody>`;
    for (const g of groups) {
      const isBest = isNum(bestV) && g.s.mean === bestV;
      h += `<tr><td class="v">${esc(gb)}-${esc(g.v)}</td><td class="num">${g.s.n}</td>`
        + `<td class="num${isBest ? " best" : ""}">${g.s.n ? fmt(g.s.mean) + (isNum(g.s.std) ? ` <span class="pm">± ${fmt(g.s.std)}</span>` : "") : "–"}</td>`
        + `<td class="num">${g.s.n ? fmt(g.s.min) : "–"}</td><td class="num">${g.s.n ? fmt(g.s.max) : "–"}</td>`
        + `<td class="dots">${g.s.n ? dotSvg(g.rs.map((r) => ({ v: mval(r), id: r.id })), g.s, lo, hi) : ""}</td></tr>`;
    }
    h += `</tbody></table></div>`;
  } else {
    const v2 = [...new Set(runs.map((r) => r.vars[gb2] ?? ""))].sort(cmpVal);
    const cell = (a, b) => stat(withM.filter((r) => (r.vars[gb] ?? "") === a && (r.vars[gb2] ?? "") === b).map(mval));
    const cells = v1.map((a) => v2.map((b) => cell(a, b)));
    const means = cells.flat().map((c) => c.mean).filter(isNum);
    const bestV = dir && means.length > 1 ? (dir === "min" ? Math.min(...means) : Math.max(...means)) : null;
    h += `<div class="tablewrap"><table class="grid matrix"><thead><tr><th>${esc(gb)} \\ ${esc(gb2)}</th>${v2.map((b) => `<th class="num mono">${esc(gb2)}-${esc(b)}</th>`).join("")}</tr></thead><tbody>`;
    v1.forEach((a, i) => {
      h += `<tr><td class="v">${esc(gb)}-${esc(a)}</td>`;
      for (const c of cells[i]) h += `<td class="num cell${isNum(bestV) && c.mean === bestV ? " best" : ""}">${c.n ? `${fmt(c.mean)}${isNum(c.std) ? ` <span class="pm">± ${fmt(c.std)}</span>` : ""}<span class="nsub">${c.n}</span>` : '<span class="muted">–</span>'}</td>`;
      h += `</tr>`;
    });
    h += `</tbody></table></div>`;
  }
  body.innerHTML = h;
}
function dotSvg(vals, s, lo, hi) {
  const W = 300, H = 20, y = H / 2;
  const sx = (v) => (6 + (v - lo) / (hi - lo) * (W - 12)).toFixed(1);
  let o = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><line class="ax" x1="6" x2="${W - 6}" y1="${y}" y2="${y}" stroke-dasharray="1 3"/>`;
  if (isNum(s.std)) o += `<line class="sd" x1="${sx(s.mean - s.std)}" x2="${sx(s.mean + s.std)}" y1="${y}" y2="${y}"/>`;
  for (const d of vals) o += `<circle class="d" cx="${sx(d.v)}" cy="${y}" r="4.5" data-open="${esc(d.id)}" style="cursor:pointer"><title>${esc(d.id)}\n${fmt(d.v)}</title></circle>`;
  o += `<line class="m" x1="${sx(s.mean)}" x2="${sx(s.mean)}" y1="3" y2="${H - 3}"/></svg>`;
  return o;
}

// ======================================================================================================= queue
function segHtml(id, clickable) {
  const segs = segSpans(parseId(id));
  return clickable ? `<button data-open="${esc(id)}">${segs}</button>` : segs;
}
async function renderQueue() {
  if (document.activeElement !== $("#qq")) $("#qq").value = S.qq;
  if (!S.queue) {
    $("#queue-body").innerHTML = `<p class="empty">Loading…</p>`;
    try { S.queue = await api("/api/queue"); } catch (e) { $("#queue-body").innerHTML = `<p class="empty err">${esc(e.message)}</p>`; return; }
    if (S.tab !== "queue") return;
  }
  const Q = S.queue;
  const uniq = new Map();
  for (const f of Q.files) for (const e of f.entries) uniq.set(e.id, e.status);
  const counts = Object.fromEntries(Q_STATUSES.map((s) => [s, [...uniq.values()].filter((x) => x === s).length]));
  $("#qstfilter").innerHTML = Q_STATUSES.map((s) => `<button data-qst="${s}" aria-pressed="${S.qst.has(s)}">${pill(s)}<span class="n">${counts[s]}</span></button>`).join("");
  $("#queue-summary").textContent = `${uniq.size} distinct ids in ${Q.files.length} file${Q.files.length === 1 ? "" : "s"} under ${Q.jobs_root}`;
  const terms = S.qq.toLowerCase().split(/\s+/).filter(Boolean);
  const keep = (e) => S.qst.has(e.status) && terms.every((t) => (t.startsWith("!") ? !e.id.toLowerCase().includes(t.slice(1)) : e.id.toLowerCase().includes(t)));
  let h = "";
  if (!Q.files.length) h += `<p class="empty">No queue files (*.txt) in ${esc(Q.jobs_root)}.</p>`;
  for (const f of Q.files) {
    const ents = f.entries.filter(keep);
    const fc = Q_STATUSES.map((s) => [s, f.entries.filter((e) => e.status === s).length]).filter(([, c]) => c);
    const withResults = f.entries.filter((e) => e.status !== "not started").map((e) => e.id);
    h += `<details class="qfile" data-file="${esc(f.name)}"${S.qclosed.has(f.name) ? "" : " open"}><summary><span class="fname">${esc(f.name)}</span>`
      + `<span class="counts">${fc.map(([s, c]) => pill(s, `${c} ${s}`)).join("")}</span><span class="spacer"></span>`
      + (withResults.length ? `<button class="btn small" data-qcompare="${esc(f.name)}">Compare its ${withResults.length} runs</button>` : "") + `</summary>`;
    if (f.header && f.header.length) h += `<p class="desc">${esc(f.header.join("\n"))}</p>`;
    if (f.error) h += `<p class="desc err">${esc(f.error)}</p>`;
    if (!ents.length) { h += `<p class="desc muted">${f.entries.length ? "No entries match the filters." : "No entries."}</p></details>`; continue; }
    h += `<table class="grid"><thead><tr><th class="num">line</th><th>status</th><th>id <span class="sub">bold = differs from the best model</span></th><th>overrides</th><th>progress</th><th class="num">val_loss<span class="sub">best</span></th><th class="num">test_loss</th><th class="num">test_r@1</th><th></th></tr></thead><tbody>`;
    for (const e of ents) {
      const prog = e.status === "not started" ? "" : e.status === "done" ? `${fmtInt(e.last_step)} steps` : `${fmtInt(e.last_step)} / ${fmtInt(e.total_steps)}`;
      h += `<tr><td class="num muted">${e.line}</td><td>${pill(e.status)}</td><td class="qidtd"><div class="qid">${segHtml(e.id, e.status !== "not started")}</div>`
        + (e.problems.length ? `<div class="problem">${esc(e.problems.join("; "))}</div>` : "") + `</td>`
        + `<td>${e.overrides.map((o) => `<span class="ov">${esc(o)}</span>`).join("")}${e.comment ? `<span class="cmt"># ${esc(e.comment)}</span>` : ""}</td>`
        + `<td class="muted">${prog}</td><td class="num">${fmt(e.best_val_loss)}</td><td class="num">${fmt(e.final && e.final.test_loss)}</td><td class="num">${fmt(e.final && e.final["test_r@1"])}</td>`
        + `<td>${copyBtn(e.id)}</td></tr>`;
    }
    h += `</tbody></table></details>`;
  }
  if (Q.unqueued && Q.unqueued.length) {
    h += `<details class="qfile" data-file="(unqueued)"${S.qclosed.has("(unqueued)") ? "" : " open"}><summary><span class="fname">in results, in no queue file</span><span class="counts muted">${Q.unqueued.length}</span></summary><table class="grid"><tbody>`;
    for (const id of Q.unqueued) {
      const r = S.byId.get(id);
      h += `<tr><td>${r ? pill(r.status) : ""}</td><td class="qidtd"><div class="qid">${segHtml(id, true)}</div></td><td>${copyBtn(id)}</td></tr>`;
    }
    h += `</tbody></table></details>`;
  }
  $("#queue-body").innerHTML = h;
}

// ======================================================================================================= demo
function renderDemo() {
  const bundles = new Set(S.data.bundles || []);
  const selected = S.sel.filter((id) => S.byId.has(id));
  const target = S.demo || selected.find((id) => bundles.has(id)) || selected[0] || (S.data.bundles || [])[0] || "";
  const opt = (id) => `<option value="${esc(id)}"${id === target ? " selected" : ""}>${bundles.has(id) ? "▶ " : "○ "}${esc(id)}</option>`;
  const withB = S.runs.filter((r) => r.has_bundle && !selected.includes(r.id)).map((r) => r.id);
  const orphan = [...bundles].filter((id) => !S.byId.has(id));
  let o = "";
  if (selected.length) o += `<optgroup label="Selected runs">${selected.map(opt).join("")}</optgroup>`;
  if (withB.length) o += `<optgroup label="Runs with a bundle">${withB.map(opt).join("")}</optgroup>`;
  if (orphan.length) o += `<optgroup label="Bundles without a run in results">${orphan.map(opt).join("")}</optgroup>`;
  if (target && !selected.includes(target) && !bundles.has(target)) o = opt(target) + o;
  $("#demo-run").innerHTML = o || `<option value="">(no runs selected, no bundles)</option>`;
  const frame = $("#demo-frame"), msg = $("#demo-msg"), open = $("#demo-open");
  if (target && bundles.has(target)) {
    const src = `demo.html?bundle=${encodeURIComponent(`/bundles/${encodeURIComponent(target)}/`)}`;
    if (frame.getAttribute("src") !== src) frame.setAttribute("src", src);
    frame.hidden = false; open.hidden = false; open.href = src;
    msg.innerHTML = "";
    return;
  }
  frame.hidden = true; open.hidden = true;
  if (frame.getAttribute("src")) frame.removeAttribute("src");
  const root = S.data.bundles_root;
  if (!target) {
    msg.innerHTML = `<div class="howto"><h3>No run to replay</h3><p>Select runs in the Runs tab (the Demo tab opens the first selected run's bundle), or make a bundle in <code>${esc(root)}</code>.</p></div>`;
    return;
  }
  const out = `${S.data.bundles_out || "dashboard/bundles"}/${target}`;
  const cmds = [
    `python -m wfemb.demo embed --id ${target}`,
    `python -m wfemb.demo background --id ${target} --out ${out}`,
    `python -m wfemb.demo export --id ${target} --out ${out} --auto 8`,
  ];
  msg.innerHTML = `<div class="howto"><h3>No demo bundle for this run yet</h3>
    <p>The bundle is read from <code>${esc(root)}/${esc(target)}/</code>. Make it from the repo root (needs the run's <code>best.pt</code> and the test frame features; screenshots of AgentNet sessions come from <code>python -m wfemb.demo frames --id …</code> on rnn):</p>
    ${cmds.map((c) => `<div class="cmd"><code>${esc(c)}</code>${copyBtn(c, "Copy command")}</div>`).join("")}
    <div class="cmd"><code>${esc(cmds.join(" && \\\n  "))}</code>${copyBtn(cmds.join(" && "), "Copy all three")}</div>
    <p class="muted">Bundles are found on the next refresh.</p></div>`;
}

// ======================================================================================================= detail drawer
async function openDetail(id) {
  S.detail = id;
  writeHash();
  $("#drawer").hidden = false;
  $("#drawer-title").innerHTML = `<div class="rid">${esc(id)}</div>`;
  $("#drawer-body").innerHTML = `<p class="muted">Loading…</p>`;
  await loadDetail(id);
}
async function loadDetail(id, quiet) {
  let d;
  try { d = await api(`/api/run?id=${encodeURIComponent(id)}`); }
  catch (e) { if (!quiet) $("#drawer-body").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  if (S.detail !== id) return;
  const r = d.run, cfg = d.config || {}, fin = d.final;
  const inSel = S.sel.includes(id);
  $("#drawer-title").innerHTML = `<div class="rid">${segHtml(id, false)} ${copyBtn(id)}</div>
    <div class="row">${pill(r.status)}${progressCell([r])}${r.legacy ? `<span class="legacy">legacy${r.legacy_name ? ": " + esc(r.legacy_name) : ""}</span>` : ""}</div>
    <div class="row"><span class="mono muted" title="results folder">${esc(r.path)}</span></div>
    <div class="row"><button class="btn small" data-toggle-sel="${esc(id)}">${inSel ? "Remove from comparison" : "Add to comparison"}</button>
      <button class="btn small" data-demo="${esc(id)}">${r.has_bundle ? "Open demo" : "Demo: how to make a bundle"}</button></div>`;
  let h = "";
  if (r.warnings && r.warnings.length) h += `<div class="warnbox">${r.warnings.map(esc).join("<br>")}</div>`;
  // final metrics: one row per metric, val and test side by side
  if (fin) {
    const bases = [];
    for (const k of sortMetrics(Object.keys(fin).filter((k) => /^(val|test)_/.test(k)))) { const b = k.replace(/^(val|test)_/, ""); if (!bases.includes(b)) bases.push(b); }
    h += `<section><h3>final.json</h3><table class="kv"><tbody>`
      + `<tr><td>best_step</td><td>${fmtInt(fin.best_step)}${r.total_steps ? ` of ${fmtInt(r.total_steps)}` : ""} (selected on ${esc(fin.selected_on ?? "?")})</td></tr>`
      + (isNum(fin.train_minutes) ? `<tr><td>train time</td><td>${fmtDur(fin.train_minutes * 60)}</td></tr>` : "")
      + `</tbody></table><table class="grid" style="margin-top:6px"><thead><tr><th>metric</th><th class="num">val</th><th class="num">test</th></tr></thead><tbody>`
      + bases.map((b) => `<tr><td class="v">${esc(b)}${direction(b) === "min" ? ' <span class="lower">lower is better</span>' : ""}</td><td class="num">${fmt(fin["val_" + b])}</td><td class="num">${fmt(fin["test_" + b])}</td></tr>`).join("")
      + `</tbody></table></section>`;
    const other = Object.entries(fin).filter(([k]) => !/^(val|test)_/.test(k) && !["best_step", "selected_on", "train_minutes"].includes(k));
    if (other.length) h += kvTable("final.json (other)", Object.fromEntries(other));
  } else {
    h += `<section><h3>final.json</h3><p class="muted">Not written yet (${esc(r.status)}). Best val_loss so far: ${fmt(r.best_val_loss)}${isNum(r.best_val_step) ? ` at step ${fmtInt(r.best_val_step)}` : ""}.</p></section>`;
  }
  for (const sec of ["experiment", "train", "eval", "data", "model", "meta"]) if (cfg[sec] && typeof cfg[sec] === "object") h += kvTable(sec, cfg[sec], sec === "experiment" ? S.ref : null);
  for (const [k, v] of Object.entries(cfg)) if (!["id", "experiment", "train", "eval", "data", "model", "meta"].includes(k)) h += kvTable(k, typeof v === "object" && v ? v : { [k]: v });
  h += `<section><h3>files</h3><table class="kv"><tbody>${Object.entries(d.files || {}).map(([n, f]) => `<tr><td>${esc(n)}</td><td>${fmtBytes(f.size)} · ${new Date(f.mtime * 1000).toLocaleString()}</td></tr>`).join("")}</tbody></table></section>`;
  h += `<details class="raw"><summary>config.json</summary><pre>${esc(JSON.stringify(cfg, null, 2))}</pre></details>`;
  if (fin) h += `<details class="raw"><summary>final.json</summary><pre>${esc(JSON.stringify(fin, null, 2))}</pre></details>`;
  $("#drawer-body").innerHTML = h;
}
function fmtBytes(n) { return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`; }
function kvTable(title, obj, ref) {
  const rows = [];
  const walk = (o, pre) => {
    for (const [k, v] of Object.entries(o)) {
      if (v && typeof v === "object" && !Array.isArray(v)) walk(v, pre + k + ".");
      else rows.push([pre + k, v]);
    }
  };
  walk(obj, "");
  const show = (v) => (v === null ? "null" : Array.isArray(v) ? JSON.stringify(v) : String(v));
  return `<section><h3>${esc(title)}</h3><table class="kv"><tbody>${rows.map(([k, v]) => {
    const diff = ref && ref[k] != null && ref[k] !== (v === null ? "off" : String(v)) && !(numLike(ref[k]) && numLike(v) && parseFloat(ref[k]) === +v);
    return `<tr class="${diff ? "diff" : ""}"><td>${esc(k)}</td><td>${esc(show(v))}${diff ? ` <span class="muted">(best model: ${esc(ref[k])})</span>` : ""}</td></tr>`;
  }).join("")}</tbody></table></section>`;
}
function closeDetail() { $("#drawer").hidden = true; S.detail = ""; writeHash(); }

// ======================================================================================================= events
function bind() {
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-copy],[data-open],[data-tab],[data-sort],[data-st],[data-qst],[data-only],[data-fclear],#clear-filters,[data-unsel],[data-hide],[data-toggle-sel],[data-demo],[data-qcompare]");
    if (!e.target.closest(".vf")) for (const d of $$(".vf[open]")) { d.open = false; S.openVf = ""; }
    if (!t) return;
    if (t.dataset.copy != null) { e.preventDefault(); e.stopPropagation(); copyText(t.dataset.copy); return; }
    if (t.dataset.open != null) { e.preventDefault(); openDetail(t.dataset.open); return; }
    if (t.dataset.tab) { e.preventDefault(); S.tab = t.dataset.tab; update(); return; }
    if (t.dataset.sort) {
      const k = t.dataset.sort;
      S.sort = S.sort.key === k ? (S.sort.desc ? { key: "vars", desc: false } : { key: k, desc: true }) : { key: k, desc: false };
      update(); return;
    }
    if (t.dataset.st) { S.st.has(t.dataset.st) ? S.st.delete(t.dataset.st) : S.st.add(t.dataset.st); if (!S.st.size) S.st = new Set(STATUSES); update(); return; }
    if (t.dataset.qst) { S.qst.has(t.dataset.qst) ? S.qst.delete(t.dataset.qst) : S.qst.add(t.dataset.qst); if (!S.qst.size) S.qst = new Set(Q_STATUSES); update(); return; }
    if (t.dataset.only != null) { e.preventDefault(); S.f[t.dataset.fk] = new Set([t.dataset.only]); update(); return; }
    if (t.dataset.fclear != null) { delete S.f[t.dataset.fclear]; update(); return; }
    if (t.id === "clear-filters") { S.f = {}; S.q = ""; $("#q").value = ""; S.st = new Set(STATUSES); update(); return; }
    if (t.dataset.unsel != null) {
      const k = t.dataset.unsel;
      S.sel = S.sel.filter((id) => id !== k && !(S.agg && S.byId.get(id) && groupKey(S.byId.get(id)) === k));
      S.hidden.delete(k); update(); return;
    }
    if (t.dataset.hide != null) { const k = t.dataset.hide; S.hidden.has(k) ? S.hidden.delete(k) : S.hidden.add(k); update(); return; }
    if (t.dataset.toggleSel != null) {
      const id = t.dataset.toggleSel;
      S.sel = S.sel.includes(id) ? S.sel.filter((x) => x !== id) : [...S.sel, id];
      update(); loadDetail(id, true); return;
    }
    if (t.dataset.demo != null) { S.demo = t.dataset.demo; S.tab = "demo"; closeDetail(); update(); return; }
    if (t.dataset.qcompare != null) {
      e.preventDefault();
      const f = S.queue.files.find((x) => x.name === t.dataset.qcompare);
      S.sel = [...new Set(f.entries.filter((x) => x.status !== "not started").map((x) => x.id))];
      S.tab = "compare"; update(); return;
    }
  });
  // run table selection (with shift-click ranges)
  $("#runs-table").addEventListener("click", (e) => {
    const cb = e.target.closest("input[type=checkbox]");
    if (!cb) return;
    if (cb.id === "sel-all") {
      const rows = rowsOf(filteredRuns());
      toggleRows(rows.map((r) => r.key), cb.checked);
    } else {
      const key = cb.dataset.row;
      if (e.shiftKey && S.lastClick && S.rowOrder.includes(S.lastClick)) {
        const a = S.rowOrder.indexOf(S.lastClick), b = S.rowOrder.indexOf(key);
        toggleRows(S.rowOrder.slice(Math.min(a, b), Math.max(a, b) + 1), cb.checked);
      } else toggleRows([key], cb.checked);
      S.lastClick = key;
    }
    update();
  });
  $("#varfilters").addEventListener("change", (e) => {
    const cb = e.target.closest("input[data-fk]");
    if (!cb) return;
    const k = cb.dataset.fk, v = cb.dataset.fv;
    const set = S.f[k] || (S.f[k] = new Set());
    cb.checked ? set.add(v) : set.delete(v);
    if (!set.size) delete S.f[k];
    S.openVf = k;
    update();
  });
  $("#varfilters").addEventListener("toggle", (e) => {
    const d = e.target;
    if (!d.matches || !d.matches(".vf")) return;
    if (d.open) { S.openVf = d.dataset.k; for (const o of $$(".vf[open]")) if (o !== d) o.open = false; }
    else if (S.openVf === d.dataset.k) S.openVf = "";
  }, true);
  let qt = 0;
  $("#q").addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => { S.q = e.target.value.trim(); update(); }, 120); });
  $("#qq").addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => { S.qq = e.target.value.trim(); update(); }, 120); });
  $("#agg").addEventListener("change", (e) => { S.agg = e.target.checked; update(); });
  $("#agg2").addEventListener("change", (e) => { S.agg = e.target.checked; update(); });
  $("#sel-clear").addEventListener("click", () => { S.sel = []; update(); });
  $("#go-compare").addEventListener("click", () => { S.tab = "compare"; update(); });
  $("#smooth").addEventListener("input", (e) => { S.smooth = +e.target.value; update(); });
  $("#xaxis").addEventListener("change", (e) => { S.x = e.target.value; GROUP.xRange = null; $("#zoom-reset").hidden = true; update(); });
  $("#raw").addEventListener("change", (e) => { S.raw = e.target.checked; update(); });
  $("#zoom-reset").addEventListener("click", () => GROUP.setRange(null));
  $("#gb").addEventListener("change", (e) => { S.gb = e.target.value; update(); });
  $("#gb2").addEventListener("change", (e) => { S.gb2 = e.target.value; update(); });
  $("#gm").addEventListener("change", (e) => { S.gm = e.target.value; update(); });
  $("#demo-run").addEventListener("change", (e) => { S.demo = e.target.value; update(); });
  $("#live").addEventListener("change", (e) => { S.live = e.target.checked; writeHash(); if (S.live) refresh(); });
  $("#refresh").addEventListener("click", () => refresh());
  $("#drawer-close").addEventListener("click", closeDetail);
  $("#queue-body").addEventListener("toggle", (e) => {
    const d = e.target;
    if (!d.matches || !d.matches(".qfile")) return;
    d.open ? S.qclosed.delete(d.dataset.file) : S.qclosed.add(d.dataset.file);
  }, true);
  $("#legend").addEventListener("pointerover", (e) => { const l = e.target.closest(".lg"); GROUP.setHighlight(l ? l.dataset.key : null); });
  $("#legend").addEventListener("pointerleave", () => GROUP.setHighlight(null));
  document.addEventListener("keydown", (e) => {
    const typing = e.target.matches("input, select, textarea");
    if (e.key === "Escape") { if (!$("#drawer").hidden) closeDetail(); else for (const d of $$(".vf[open]")) d.open = false; return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); S.tab = S.tab === "queue" ? "queue" : "runs"; update(); (S.tab === "queue" ? $("#qq") : $("#q")).focus(); }
    else if (e.key === "r") refresh();
  });
  window.addEventListener("hashchange", () => {
    if (location.hash === S.lastHash) return;
    readHash(); render();
    if (S.detail) openDetail(S.detail); else $("#drawer").hidden = true;
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { PAL = null; Charts.readTheme(); render(); GROUP.redrawAll(); });
  setInterval(() => { if (S.live && !document.hidden) refresh(); }, REFRESH_MS);
  setInterval(tickUpdated, 5000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && S.live && Date.now() - S.updatedAt > REFRESH_MS) refresh(); });
}

// ------------------------------------------------------------------------------------------------------- boot
(async function boot() {
  readHash();
  initCharts();
  bind();
  $("#q").value = S.q;
  $("#qq").value = S.qq;
  await refresh();
  writeHash();
  if (S.detail) openDetail(S.detail);
})();
