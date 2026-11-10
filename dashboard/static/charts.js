/* Canvas line charts for the dashboard: many series, NaN gaps, min-max bands, point markers, a crosshair synced
   across a group of charts, and drag-to-zoom on x (double-click resets). No dependencies.

   series: [{ id, label, color, dash: [..] | null, x: Float64Array, y: Float64Array,
              raw: {x, y} | null      (drawn faint behind the line, e.g. the unsmoothed values),
              band: {lo, hi} | null   (same x as the line, e.g. min-max over seeds),
              points: bool            (dots on every point, for sparse eval series),
              markers: [{x, y, label}] (rings, e.g. the selected checkpoint) }] */
"use strict";
(function () {
  const theme = {};
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["grid", "axis", "muted", "text", "text-2", "surface", "mono", "sans", "accent"]) theme[k] = cs.getPropertyValue("--" + k).trim();
  }
  readTheme();

  // ------------------------------------------------------------------------------------------------ helpers
  function niceStep(range, target) {
    const raw = range / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  }
  function ticks(lo, hi, target) {
    if (!(hi > lo)) return { ticks: [lo], step: 1 };
    const st = niceStep(hi - lo, target), out = [];
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    return { ticks: out, step: st };
  }
  function fmtTick(v, step) {
    if (v === 0) return "0";
    if (step < 1e-3 || Math.abs(v) >= 1e6) {
      const e = Math.floor(Math.log10(Math.abs(v)));
      const d = Math.max(0, Math.min(2, e - Math.floor(Math.log10(step) + 1e-9)));
      return (v / Math.pow(10, e)).toFixed(d) + "e" + e;
    }
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
    return v.toFixed(dec);
  }
  function fmtVal(v) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && a < 1e-3) return v.toExponential(2);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return v.toFixed(4);
  }
  // first index with x[i] >= v (x ascending, NaN-free)
  function lowerBound(x, v) {
    let lo = 0, hi = x.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (x[m] < v) lo = m + 1; else hi = m; }
    return lo;
  }
  function nearest(x, v) {
    const n = x.length;
    if (!n) return -1;
    const i = lowerBound(x, v);
    if (i <= 0) return 0;
    if (i >= n) return n - 1;
    return v - x[i - 1] <= x[i] - v ? i - 1 : i;
  }

  // ------------------------------------------------------------------------------------------------ group
  class Group {
    constructor() { this.charts = new Set(); this.xRange = null; this.hoverX = null; this.hoverSrc = null; this.highlight = null; this.onRange = null; }
    add(c) { this.charts.add(c); c.group = this; }
    remove(c) { this.charts.delete(c); }
    setHover(x, src, py) {
      this.hoverX = x; this.hoverSrc = src; this.hoverPy = py;
      for (const c of this.charts) c.drawOverlay();
    }
    setRange(r) {
      this.xRange = r;
      for (const c of this.charts) c.draw();
      if (this.onRange) this.onRange(r);
    }
    setHighlight(id) {
      if (this.highlight === id) return;
      this.highlight = id;
      for (const c of this.charts) c.draw();
    }
    redrawAll() { readTheme(); for (const c of this.charts) c.draw(); }
  }

  // ------------------------------------------------------------------------------------------------ chart
  class Chart {
    constructor(host, opt) {
      this.opt = Object.assign({ xFormat: (v) => String(v), height: 190 }, opt);
      this.host = host;
      host.classList.add("chart");
      host.innerHTML = `<div class="chart-head"><span class="chart-title"></span><span class="chart-sub"></span></div>
        <div class="chart-body" style="height:${this.opt.height}px"><canvas class="c-base"></canvas><canvas class="c-over"></canvas>
        <div class="chart-empty" hidden>No data</div><div class="chart-tip" hidden></div></div>`;
      this.body = host.querySelector(".chart-body");
      this.base = host.querySelector(".c-base");
      this.over = host.querySelector(".c-over");
      this.tip = host.querySelector(".chart-tip");
      this.empty = host.querySelector(".chart-empty");
      host.querySelector(".chart-title").textContent = this.opt.title || "";
      this.setSub(this.opt.sub || "");
      this.series = [];
      this.W = 0; this.H = 0;
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.body);
      this.drag = null;
      const ov = this.over;
      ov.addEventListener("pointermove", (e) => this.onMove(e));
      ov.addEventListener("pointerleave", () => { if (!this.drag && this.group) this.group.setHover(null, null); });
      ov.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || !this.plot) return;
        const p = this.local(e);
        this.drag = { x0: p.x, x1: p.x };
        ov.setPointerCapture(e.pointerId);
      });
      ov.addEventListener("pointerup", () => {
        const d = this.drag;
        this.drag = null;
        if (d && Math.abs(d.x1 - d.x0) > 4 && this.group) {
          const a = this.xInv(Math.min(d.x0, d.x1)), b = this.xInv(Math.max(d.x0, d.x1));
          this.group.setRange([a, b]);
        } else this.drawOverlay();
      });
      ov.addEventListener("dblclick", () => { if (this.group) this.group.setRange(null); });
    }
    destroy() { this.ro.disconnect(); if (this.group) this.group.remove(this); }
    setSub(s) { this.host.querySelector(".chart-sub").textContent = s; }
    setTitle(s) { this.host.querySelector(".chart-title").textContent = s; }
    setSeries(series) { this.series = series || []; this.draw(); }
    local(e) { const r = this.over.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    resize() {
      const W = this.body.clientWidth, H = this.body.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!W || !H) return;
      this.W = W; this.H = H;
      for (const cv of [this.base, this.over]) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
        cv.style.width = W + "px"; cv.style.height = H + "px";
      }
      this.draw();
    }

    // data extent: x over all series; y over the lines (and bands) inside the visible x range
    extent() {
      let x0 = Infinity, x1 = -Infinity;
      for (const s of this.series) {
        const n = s.x.length;
        for (let i = 0; i < n; i++) { const v = s.x[i]; if (isFinite(s.y[i])) { if (v < x0) x0 = v; break; } }
        for (let i = n - 1; i >= 0; i--) { const v = s.x[i]; if (isFinite(s.y[i])) { if (v > x1) x1 = v; break; } }
      }
      if (!(x1 >= x0)) return null;
      const r = this.group && this.group.xRange;
      if (r) { x0 = r[0]; x1 = r[1]; }
      if (x1 === x0) { x0 -= 1; x1 += 1; }
      let y0 = Infinity, y1 = -Infinity;
      for (const s of this.series) {
        const i0 = Math.max(0, lowerBound(s.x, x0) - 1), i1 = Math.min(s.x.length - 1, lowerBound(s.x, x1));
        const cols = s.band ? [s.y, s.band.lo, s.band.hi] : [s.y];
        for (const a of cols) for (let i = i0; i <= i1; i++) { const v = a[i]; if (isFinite(v)) { if (v < y0) y0 = v; if (v > y1) y1 = v; } }
        if (s.markers) for (const m of s.markers) if (m.x >= x0 && m.x <= x1 && isFinite(m.y)) { y0 = Math.min(y0, m.y); y1 = Math.max(y1, m.y); }
      }
      if (!(y1 >= y0)) return null;
      if (y1 === y0) { const d = Math.abs(y0) * 0.05 || 1; y0 -= d; y1 += d; }
      const pad = (y1 - y0) * 0.06;
      if (this.opt.yMin != null) y0 = Math.max(this.opt.yMin, y0 - pad); else y0 -= pad;
      y1 += pad;
      return { x0, x1, y0, y1 };
    }

    draw() {
      if (!this.W) return;
      const g = this.base.getContext("2d"), dpr = window.devicePixelRatio || 1;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, this.W, this.H);
      const ex = this.extent();
      this.empty.hidden = !!ex;
      this.plot = null;
      if (!ex) { this.drawOverlay(); return; }
      const W = this.W, H = this.H;
      const yt = ticks(ex.y0, ex.y1, Math.max(3, Math.round(H / 42)));
      g.font = `10.5px ${theme.mono}`;
      const ylabels = yt.ticks.filter((v) => v >= ex.y0 && v <= ex.y1).map((v) => [v, fmtTick(v, yt.step)]);
      const padL = Math.max(30, Math.ceil(Math.max(...ylabels.map(([, s]) => g.measureText(s).width), 0)) + 10);
      const P = { l: padL, r: 10, t: 6, b: 20 };
      const pw = W - P.l - P.r, ph = H - P.t - P.b;
      const sx = (v) => P.l + (v - ex.x0) / (ex.x1 - ex.x0) * pw;
      const sy = (v) => P.t + (1 - (v - ex.y0) / (ex.y1 - ex.y0)) * ph;
      this.plot = { P, pw, ph, sx, sy, ex };
      this.xInv = (px) => ex.x0 + (px - P.l) / pw * (ex.x1 - ex.x0);

      // grid + axes
      g.lineWidth = 1;
      g.strokeStyle = theme.grid;
      g.fillStyle = theme.muted;
      g.textAlign = "right"; g.textBaseline = "middle";
      for (const [v, s] of ylabels) {
        const y = Math.round(sy(v)) + 0.5;
        g.beginPath(); g.moveTo(P.l, y); g.lineTo(W - P.r, y); g.stroke();
        g.fillText(s, P.l - 6, y);
      }
      const xt = ticks(ex.x0, ex.x1, Math.max(2, Math.round(pw / 90)));
      g.textAlign = "center"; g.textBaseline = "top";
      for (const v of xt.ticks) {
        if (v < ex.x0 || v > ex.x1) continue;
        const x = Math.round(sx(v)) + 0.5;
        g.beginPath(); g.moveTo(x, P.t); g.lineTo(x, P.t + ph); g.stroke();
        g.fillText(this.opt.xFormat(v, xt.step), Math.min(W - 18, Math.max(P.l + 8, x)), P.t + ph + 5);
      }
      g.strokeStyle = theme.axis;
      g.beginPath(); g.moveTo(P.l, P.t + ph + 0.5); g.lineTo(W - P.r, P.t + ph + 0.5); g.stroke();

      // series
      g.save();
      g.beginPath(); g.rect(P.l, P.t - 2, pw, ph + 4); g.clip();
      const hl = this.group && this.group.highlight;
      const order = hl ? [...this.series.filter((s) => s.id !== hl), ...this.series.filter((s) => s.id === hl)] : this.series;
      for (const s of order) {
        const dim = hl && s.id !== hl;
        const i0 = Math.max(0, lowerBound(s.x, ex.x0) - 1), i1 = Math.min(s.x.length - 1, lowerBound(s.x, ex.x1));
        if (s.band) {
          g.globalAlpha = dim ? 0.04 : 0.16; g.fillStyle = s.color;
          let open = false, start = -1;
          const flush = (end) => {
            if (start < 0) return;
            for (let j = end; j >= start; j--) g.lineTo(sx(s.x[j]), sy(s.band.lo[j]));
            g.closePath(); g.fill(); start = -1; open = false;
          };
          for (let i = i0; i <= i1; i++) {
            const lo = s.band.lo[i], hi = s.band.hi[i];
            if (!isFinite(lo) || !isFinite(hi)) { flush(i - 1); continue; }
            if (!open) { g.beginPath(); g.moveTo(sx(s.x[i]), sy(hi)); open = true; start = i; } else g.lineTo(sx(s.x[i]), sy(hi));
          }
          flush(i1);
        }
        if (s.raw) {
          g.globalAlpha = dim ? 0.05 : 0.2; g.strokeStyle = s.color; g.lineWidth = 1; g.setLineDash([]);
          const r0 = Math.max(0, lowerBound(s.raw.x, ex.x0) - 1), r1 = Math.min(s.raw.x.length - 1, lowerBound(s.raw.x, ex.x1));
          this.line(g, s.raw.x, s.raw.y, r0, r1, sx, sy);
        }
        g.globalAlpha = dim ? 0.22 : 1;
        g.strokeStyle = s.color; g.lineWidth = s.id === hl ? 2.25 : 1.6; g.setLineDash(s.dash || []);
        g.lineJoin = "round";
        this.line(g, s.x, s.y, i0, i1, sx, sy);
        g.setLineDash([]);
        if (s.points && (i1 - i0) < pw / 5) {
          g.fillStyle = s.color;
          for (let i = i0; i <= i1; i++) if (isFinite(s.y[i])) { g.beginPath(); g.arc(sx(s.x[i]), sy(s.y[i]), 2, 0, 6.2832); g.fill(); }
        }
        g.globalAlpha = 1;
      }
      for (const s of this.series) {
        if (!s.markers || (hl && s.id !== hl)) continue;
        for (const m of s.markers) {
          if (!isFinite(m.y) || m.x < ex.x0 || m.x > ex.x1) continue;
          g.beginPath(); g.arc(sx(m.x), sy(m.y), 4.5, 0, 6.2832);
          g.fillStyle = theme.surface; g.fill(); g.lineWidth = 2; g.strokeStyle = s.color; g.stroke();
        }
      }
      g.restore();
      this.drawOverlay();
    }

    line(g, xs, ys, i0, i1, sx, sy) {
      g.beginPath();
      let pen = false;
      for (let i = i0; i <= i1; i++) {
        const v = ys[i];
        if (!isFinite(v)) { pen = false; continue; }
        const X = sx(xs[i]), Y = sy(v);
        if (pen) g.lineTo(X, Y); else { g.moveTo(X, Y); pen = true; }
      }
      g.stroke();
    }

    onMove(e) {
      if (!this.plot || !this.group) return;
      const p = this.local(e);
      if (this.drag) { this.drag.x1 = Math.max(this.plot.P.l, Math.min(this.plot.P.l + this.plot.pw, p.x)); }
      // snap to the nearest data x of any series in this chart
      const xv = this.xInv(p.x);
      let best = null, bd = Infinity;
      for (const s of this.series) {
        const i = nearest(s.x, xv);
        if (i < 0) continue;
        const d = Math.abs(s.x[i] - xv);
        if (d < bd) { bd = d; best = s.x[i]; }
      }
      this.group.setHover(best, this, p.y);
    }

    drawOverlay() {
      if (!this.W) return;
      const g = this.over.getContext("2d"), dpr = window.devicePixelRatio || 1;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, this.W, this.H);
      this.tip.hidden = true;
      const pl = this.plot;
      if (!pl) return;
      const { P, ph, sx, sy, ex } = pl;
      if (this.drag) {
        const a = Math.min(this.drag.x0, this.drag.x1), b = Math.max(this.drag.x0, this.drag.x1);
        g.fillStyle = theme.accent; g.globalAlpha = 0.12; g.fillRect(a, P.t, b - a, ph); g.globalAlpha = 1;
      }
      const G = this.group, hx = G && G.hoverX;
      if (hx == null || hx < ex.x0 || hx > ex.x1) return;
      const X = Math.round(sx(hx)) + 0.5;
      g.strokeStyle = theme.axis; g.lineWidth = 1;
      g.beginPath(); g.moveTo(X, P.t); g.lineTo(X, P.t + ph); g.stroke();
      const rows = [];
      for (const s of this.series) {
        const n = s.x.length;
        if (!n) continue;
        const i = nearest(s.x, hx);
        // only within the series' own span (a run that stopped early shows nothing past its end)
        const gap = n > 1 ? (s.x[n - 1] - s.x[0]) / (n - 1) : 0;
        if (hx < s.x[0] - gap || hx > s.x[n - 1] + gap) continue;
        const v = s.y[i];
        if (!isFinite(v)) continue;
        const Y = sy(v);
        if (Y >= P.t - 2 && Y <= P.t + ph + 2) {
          g.beginPath(); g.arc(sx(s.x[i]), Y, 3.5, 0, 6.2832);
          g.fillStyle = s.color; g.fill(); g.lineWidth = 1.5; g.strokeStyle = theme.surface; g.stroke();
        }
        rows.push({ s, v, Y, lo: s.band ? s.band.lo[i] : null, hi: s.band ? s.band.hi[i] : null });
      }
      if (G.hoverSrc !== this || !rows.length) return;
      // tooltip: x, then every series' value (closest line to the pointer in bold)
      let near = null, nd = Infinity;
      for (const r of rows) { const d = Math.abs(r.Y - (G.hoverPy ?? 0)); if (d < nd) { nd = d; near = r; } }
      const tip = this.tip;
      tip.textContent = "";
      const head = document.createElement("div");
      head.className = "tip-x";
      head.textContent = (this.opt.xName ? this.opt.xName + " " : "") + this.opt.xFormat(hx, 0, true);
      tip.appendChild(head);
      for (const r of rows) {
        const row = document.createElement("div");
        row.className = "tip-row" + (r === near ? " near" : "");
        const key = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        key.setAttribute("width", "14"); key.setAttribute("height", "8");
        const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
        ln.setAttribute("x1", "0"); ln.setAttribute("x2", "14"); ln.setAttribute("y1", "4"); ln.setAttribute("y2", "4");
        ln.setAttribute("stroke", r.s.color); ln.setAttribute("stroke-width", "2.5");
        if (r.s.dash) ln.setAttribute("stroke-dasharray", r.s.dash.join(" "));
        key.appendChild(ln);
        const val = document.createElement("b");
        val.textContent = fmtVal(r.v);
        const lab = document.createElement("span");
        lab.textContent = r.s.label + (r.lo != null && isFinite(r.lo) && r.hi !== r.lo ? `  (${fmtVal(r.lo)} – ${fmtVal(r.hi)})` : "");
        row.append(key, val, lab);
        tip.appendChild(row);
      }
      tip.hidden = false;
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let left = X + 14;
      if (left + tw > this.W) left = X - 14 - tw;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, Math.min(this.H - th, (G.hoverPy ?? 0) - th / 2)) + "px";
    }
  }

  window.Charts = { Chart, Group, fmtVal, fmtTick, readTheme, nearest, lowerBound };
})();
