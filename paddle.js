// PaddleOCR pipeline running in the browser with ONNX Runtime Web.
//
//   1. Detection (DB model) finds every text line as a rotated rectangle.
//   2. Each line is cut out, turned upright (vertical lines are rotated and a
//      small classifier fixes upside down lines), and scaled to 48 px tall.
//   3. Recognition (SVTR/CTC model) reads the characters. CTC time steps also
//      tell us where each character sits, which gives per word positions.
//
// All geometry is in pixels of the source canvas passed to recognizePage().

import * as ort from './vendor/ort/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

const MODELS = new URL('./models/', import.meta.url).href;

export const MODEL_SETS = {
  fast: { det: 'PP-OCRv5_mobile_det.onnx', rec: 'en_PP-OCRv5_mobile_rec.onnx', dict: 'ppocrv5_en_dict.txt' },
};

const DET = { thresh: 0.3, boxThresh: 0.6, unclip: 1.5, maxSide: 2560, overlap: 256, minSize: 3 };
const REC = { height: 48, batch: 8, maxWidth: 3200 };

// ------------------------------------------------------------------ loading

export async function createEngine(setName = 'fast', { onStatus = () => {}, providers } = {}) {
  const set = typeof setName === 'string' ? MODEL_SETS[setName] : setName;
  const ep = providers || ['wasm'];
  const opts = { executionProviders: ep, graphOptimizationLevel: 'all' };
  onStatus('Loading OCR models');
  const [det, rec, cls, dictText] = await Promise.all([
    ort.InferenceSession.create(MODELS + set.det, opts),
    ort.InferenceSession.create(MODELS + set.rec, opts),
    ort.InferenceSession.create(MODELS + (set.cls || 'PP-LCNet_x0_25_textline_ori.onnx'), opts),
    fetch(MODELS + set.dict).then(r => { if (!r.ok) throw new Error('dictionary ' + r.status); return r.text(); }),
  ]);
  // Index 0 is the CTC blank. Some models add a space after the dictionary,
  // which recognize() appends once it sees the model's class count.
  const dict = [''].concat(dictText.replace(/\r/g, '').split('\n').filter(c => c !== ''));
  return { det, rec, cls, dict, set: setName };
}

// ------------------------------------------------------------------ helpers

function canvas(w, h) {
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  return c;
}

// RGBA pixels to a normalized float CHW tensor in BGR order (Paddle convention).
function toTensor(rgba, w, h, mean, std, out, offset = 0, rowStride = w) {
  const plane = h * rowStride;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const o = offset + y * rowStride + x;
      out[o] = (rgba[i + 2] / 255 - mean[0]) / std[0];
      out[o + plane] = (rgba[i + 1] / 255 - mean[1]) / std[1];
      out[o + 2 * plane] = (rgba[i] / 255 - mean[2]) / std[2];
    }
  }
}

// Convex hull (monotone chain) of points [[x,y],...].
function hull(pts) {
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  up.pop(); lo.pop();
  return lo.concat(up);
}

// Minimum area rectangle around points. Returns {cx, cy, w, h, ex, ey} where
// ex is the unit vector along w (kept within 45 degrees of +x) and ey = ex
// rotated 90 degrees clockwise on screen (pointing "down" for upright text).
function minAreaRect(points) {
  const hp = points.length > 2 ? hull(points) : points;
  let best = null;
  for (let i = 0; i < hp.length; i++) {
    const p = hp[i], q = hp[(i + 1) % hp.length];
    let dx = q[0] - p[0], dy = q[1] - p[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) continue;
    dx /= len; dy /= len;
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const r of hp) {
      const a = r[0] * dx + r[1] * dy, b = -r[0] * dy + r[1] * dx;
      if (a < a0) a0 = a; if (a > a1) a1 = a; if (b < b0) b0 = b; if (b > b1) b1 = b;
    }
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best.area) best = { area, dx, dy, a0, a1, b0, b1 };
  }
  if (!best) { const [x, y] = points[0]; return { cx: x, cy: y, w: 1, h: 1, ex: [1, 0], ey: [0, 1] }; }
  const { dx, dy, a0, a1, b0, b1 } = best;
  const am = (a0 + a1) / 2, bm = (b0 + b1) / 2;
  let rect = { cx: am * dx - bm * dy, cy: am * dy + bm * dx, w: a1 - a0, h: b1 - b0, ex: [dx, dy], ey: [-dy, dx] };
  return normalizeRect(rect);
}

// Pick the axis closest to +x as the width axis, with ey = ex turned 90 deg clockwise.
function normalizeRect(r) {
  let { ex, w, h } = r;
  const cands = [
    [ex, w, h], [[-ex[0], -ex[1]], w, h],
    [[-ex[1], ex[0]], h, w], [[ex[1], -ex[0]], h, w],
  ];
  let bestC = cands[0];
  for (const c of cands) if (c[0][0] > bestC[0][0]) bestC = c;
  const [e, ww, hh] = bestC;
  return { cx: r.cx, cy: r.cy, w: ww, h: hh, ex: e, ey: [-e[1], e[0]] };
}

function rectCorners(r) {
  const hw = r.w / 2, hh = r.h / 2, [ex, ey] = [r.ex, r.ey];
  return [
    [r.cx - ex[0] * hw - ey[0] * hh, r.cy - ex[1] * hw - ey[1] * hh],
    [r.cx + ex[0] * hw - ey[0] * hh, r.cy + ex[1] * hw - ey[1] * hh],
    [r.cx + ex[0] * hw + ey[0] * hh, r.cy + ex[1] * hw + ey[1] * hh],
    [r.cx - ex[0] * hw + ey[0] * hh, r.cy - ex[1] * hw + ey[1] * hh],
  ];
}
function aabb(r) {
  const c = rectCorners(r);
  const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
const boxArea = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const boxInter = (a, b) => boxArea([Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]);

// ------------------------------------------------------------------ detection

// DB post processing on one probability map. Returns rects in map pixels.
function dbBoxes(prob, W, H) {
  const bitmap = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) bitmap[i] = prob[i] > DET.thresh ? 1 : 0;
  const seen = new Uint8Array(W * H);
  const stack = new Int32Array(W * H);
  const rows = new Map();
  const out = [];
  for (let start = 0; start < W * H; start++) {
    if (!bitmap[start] || seen[start]) continue;
    let sp = 0, n = 0, sum = 0;
    stack[sp++] = start; seen[start] = 1;
    rows.clear();
    while (sp) {
      const i = stack[--sp];
      const x = i % W, y = (i / W) | 0;
      n++; sum += prob[i];
      const r = rows.get(y);
      if (r) { if (x < r[0]) r[0] = x; if (x > r[1]) r[1] = x; } else rows.set(y, [x, x]);
      // 8 connectivity
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= W) continue;
          const j = yy * W + xx;
          if (bitmap[j] && !seen[j]) { seen[j] = 1; stack[sp++] = j; }
        }
      }
    }
    if (n < 4) continue;
    if (sum / n < DET.boxThresh) continue;
    const pts = [];
    for (const [y, [x0, x1]] of rows) { pts.push([x0, y], [x1 + 1, y], [x0, y + 1], [x1 + 1, y + 1]); }
    const r = minAreaRect(pts);
    if (Math.min(r.w, r.h) < DET.minSize) continue;
    // Unclip: grow each side by area * ratio / perimeter.
    const d = (r.w * r.h * DET.unclip) / (2 * (r.w + r.h));
    r.w += 2 * d; r.h += 2 * d;
    if (Math.min(r.w, r.h) < DET.minSize + 2) continue;
    out.push(r);
  }
  return out;
}

async function detectTile(engine, src, sx, sy, sw, sh, scale) {
  const W = Math.max(32, Math.round(sw * scale / 32) * 32);
  const H = Math.max(32, Math.round(sh * scale / 32) * 32);
  const c = canvas(W, H);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, W, H);
  const rgba = ctx.getImageData(0, 0, W, H).data;
  const data = new Float32Array(3 * W * H);
  toTensor(rgba, W, H, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], data);
  const res = await engine.det.run({ [engine.det.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, H, W]) });
  const prob = res[engine.det.outputNames[0]].data;
  const kx = sw / W, ky = sh / H;
  return dbBoxes(prob, W, H).map(r => {
    // Map from tile map pixels to source pixels (kx and ky are nearly equal).
    const k = (kx + ky) / 2;
    return { ...r, cx: sx + r.cx * kx, cy: sy + r.cy * ky, w: r.w * k, h: r.h * k };
  });
}

// Detect lines on the whole source, tiling when it is larger than the model limit.
async function detect(engine, src, SW, SH, detScale) {
  const maxSrc = DET.maxSide / detScale;
  const ov = DET.overlap / detScale;
  const nx = SW <= maxSrc ? 1 : Math.ceil((SW - ov) / (maxSrc - ov));
  const ny = SH <= maxSrc ? 1 : Math.ceil((SH - ov) / (maxSrc - ov));
  if (nx === 1 && ny === 1) return detectTile(engine, src, 0, 0, SW, SH, detScale);

  const tw = Math.ceil((SW + (nx - 1) * ov) / nx), th = Math.ceil((SH + (ny - 1) * ov) / ny);
  const whole = [], cut = [];
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const x0 = Math.min(SW - tw, ix * (tw - ov)), y0 = Math.min(SH - th, iy * (th - ov));
      const rects = await detectTile(engine, src, x0, y0, tw, th, detScale);
      const edge = 3 / detScale;
      for (const r of rects) {
        const b = aabb(r);
        const touches = (ix > 0 && b[0] <= x0 + edge) || (ix < nx - 1 && b[2] >= x0 + tw - edge)
          || (iy > 0 && b[1] <= y0 + edge) || (iy < ny - 1 && b[3] >= y0 + th - edge);
        (touches ? cut : whole).push(r);
      }
    }
  }
  // Join pieces of lines that were cut by tile seams.
  const parent = cut.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const cb = cut.map(aabb);
  for (let i = 0; i < cut.length; i++) {
    for (let j = i + 1; j < cut.length; j++) {
      const hor = (r) => r.w >= r.h;
      if (hor(cut[i]) !== hor(cut[j])) continue;
      const inter = boxInter(cb[i], cb[j]);
      if (inter <= 0) continue;
      const a = cb[i], b = cb[j];
      const crossOverlap = hor(cut[i])
        ? Math.min(a[3], b[3]) - Math.max(a[1], b[1])
        : Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
      const crossSize = hor(cut[i]) ? Math.min(a[3] - a[1], b[3] - b[1]) : Math.min(a[2] - a[0], b[2] - b[0]);
      if (crossOverlap > 0.5 * crossSize) parent[find(i)] = find(j);
    }
  }
  const groups = new Map();
  cut.forEach((r, i) => { const g = find(i); (groups.get(g) || groups.set(g, []).get(g)).push(r); });
  const merged = [];
  for (const g of groups.values()) {
    if (g.length === 1) { merged.push(g[0]); continue; }
    // Rectangle along the first piece's direction, covering all corners.
    const { ex, ey } = g[0];
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const r of g) for (const p of rectCorners(r)) {
      const a = p[0] * ex[0] + p[1] * ex[1], b = p[0] * ey[0] + p[1] * ey[1];
      a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b);
    }
    const am = (a0 + a1) / 2, bm = (b0 + b1) / 2;
    merged.push({ cx: am * ex[0] + bm * ey[0], cy: am * ex[1] + bm * ey[1], w: a1 - a0, h: b1 - b0, ex, ey });
  }
  // Whole lines found twice in an overlap, or inside a merged line, are duplicates.
  const all = merged.concat(whole).map(r => ({ r, b: aabb(r) }));
  all.sort((p, q) => boxArea(q.b) - boxArea(p.b));
  const kept = [];
  for (const p of all) {
    if (kept.some(k => boxInter(k.b, p.b) > 0.6 * boxArea(p.b))) continue;
    kept.push(p);
  }
  return kept.map(k => k.r);
}

// ------------------------------------------------------------------ recognition

// Cut a rect out of src into a canvas of size W x H, with the rect's ex axis
// mapped to +x and ey axis to +y.
function cropRect(src, r, W, H) {
  const c = canvas(W, H);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const sx = W / r.w, sy = H / r.h;
  const [ex, ey] = [r.ex, r.ey];
  const e = -(r.cx * ex[0] + r.cy * ex[1]) * sx + W / 2;
  const f = -(r.cx * ey[0] + r.cy * ey[1]) * sy + H / 2;
  ctx.setTransform(ex[0] * sx, ey[0] * sy, ex[1] * sx, ey[1] * sy, e, f);
  // Only hand the area around the rect to drawImage; large pages are slow otherwise.
  const b = aabb(r);
  const x0 = Math.max(0, Math.floor(b[0]) - 2), y0 = Math.max(0, Math.floor(b[1]) - 2);
  const x1 = Math.min(src.width, Math.ceil(b[2]) + 2), y1 = Math.min(src.height, Math.ceil(b[3]) + 2);
  if (x1 > x0 && y1 > y0) ctx.drawImage(src, x0, y0, x1 - x0, y1 - y0, x0, y0, x1 - x0, y1 - y0);
  return ctx.getImageData(0, 0, W, H).data;
}

// Turn a rect so its reading direction is ex. Tall rects are read bottom to top.
function upright(r) {
  if (r.h >= 1.5 * r.w) {
    return { ...r, w: r.h, h: r.w, ex: [-r.ey[0], -r.ey[1]], ey: [r.ex[0], r.ex[1]], turned: true };
  }
  return r;
}
const flip = (r) => ({ ...r, ex: [-r.ex[0], -r.ex[1]], ey: [-r.ey[0], -r.ey[1]] });

async function classify(engine, src, rects) {
  const W = 160, H = 80, B = 16;
  for (let s = 0; s < rects.length; s += B) {
    const part = rects.slice(s, s + B);
    const data = new Float32Array(part.length * 3 * W * H);
    part.forEach((r, i) => toTensor(cropRect(src, r, W, H), W, H, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], data, i * 3 * W * H));
    const res = await engine.cls.run({ [engine.cls.inputNames[0]]: new ort.Tensor('float32', data, [part.length, 3, H, W]) });
    const p = res[engine.cls.outputNames[0]].data;
    part.forEach((r, i) => { if (p[i * 2 + 1] > 0.9) rects[s + i] = flip(r); });
  }
  return rects;
}

function decode(probs, T, C, dict) {
  const chars = [];
  let prev = 0, confSum = 0;
  for (let t = 0; t < T; t++) {
    let best = 0, bp = -1;
    const o = t * C;
    for (let k = 0; k < C; k++) if (probs[o + k] > bp) { bp = probs[o + k]; best = k; }
    if (best !== 0 && best !== prev) {
      chars.push({ ch: dict[best] ?? '', t0: t, t1: t, p: bp });
      confSum += bp;
    } else if (best !== 0 && best === prev && chars.length) {
      chars[chars.length - 1].t1 = t;
    }
    prev = best;
  }
  return { chars, conf: chars.length ? confSum / chars.length : 0 };
}

async function recognize(engine, src, rects) {
  const H = REC.height;
  const items = rects.map((r, idx) => ({ r, idx, W: Math.min(REC.maxWidth, Math.max(16, Math.round(H * r.w / r.h))) }));
  items.sort((a, b) => a.W - b.W);
  const results = new Array(rects.length);
  for (let s = 0; s < items.length; s += REC.batch) {
    const part = items.slice(s, s + REC.batch);
    const Wp = Math.ceil(Math.max(...part.map(p => p.W)) / 8) * 8;
    const data = new Float32Array(part.length * 3 * H * Wp);
    part.forEach((p, i) => toTensor(cropRect(src, p.r, p.W, H), p.W, H, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], data, i * 3 * H * Wp, Wp));
    const res = await engine.rec.run({ [engine.rec.inputNames[0]]: new ort.Tensor('float32', data, [part.length, 3, H, Wp]) });
    const out = res[engine.rec.outputNames[0]];
    const [, T, C] = out.dims;
    if (engine.dict.length === C - 1) engine.dict.push(' ');
    part.forEach((p, i) => {
      const d = decode(out.data.subarray(i * T * C, (i + 1) * T * C), T, C, engine.dict);
      results[p.idx] = { ...d, stepPx: Wp / T, W: p.W };
    });
  }
  return results;
}

// Split a recognized line into words with positions as fractions of the line width.
function toWords(line) {
  const { chars, stepPx, W } = line;
  const pos = chars.map(c => ((c.t0 + c.t1 + 1) / 2) * stepPx / W);
  const gaps = [];
  for (let i = 1; i < chars.length; i++) if (chars[i].ch !== ' ' && chars[i - 1].ch !== ' ') gaps.push(pos[i] - pos[i - 1]);
  gaps.sort((a, b) => a - b);
  const pitch = gaps.length ? gaps[gaps.length >> 1] : (chars.length ? 1 / chars.length : 1);
  const words = [];
  let cur = null;
  chars.forEach((c, i) => {
    if (c.ch === ' ') { cur = null; return; }
    if (!cur) { cur = { text: '', f0: pos[i], f1: pos[i] }; words.push(cur); }
    cur.text += c.ch; cur.f1 = pos[i];
  });
  for (const w of words) { w.f0 = Math.max(0, w.f0 - pitch / 2); w.f1 = Math.min(1, w.f1 + pitch / 2); }
  return words;
}

// ------------------------------------------------------------------ public

// OCR a canvas. detScale is the factor from source pixels to detection pixels.
// Returns lines: { text, conf, rect, words: [{ text, a, b, size }] } where a and b
// are the start and end of the word baseline in source pixels and size is the
// text height in source pixels.
export async function recognizePage(engine, src, { detScale = 0.5, minConf = 0.5 } = {}) {
  const SW = src.width, SH = src.height;
  const t0 = performance.now();
  let rects = (await detect(engine, src, SW, SH, detScale)).map(upright);
  const t1 = performance.now();
  rects = await classify(engine, src, rects);
  const t2 = performance.now();
  const recs = await recognize(engine, src, rects);
  const t3 = performance.now();
  engine.lastTiming = { detect: t1 - t0, classify: t2 - t1, recognize: t3 - t2, lines: rects.length, size: [SW, SH] };
  const lines = [];
  rects.forEach((r, i) => {
    const rec = recs[i];
    const text = rec.chars.map(c => c.ch).join('').trim();
    // Drop empty reads and bits of linework read as punctuation.
    if (!/[\p{L}\p{N}]/u.test(text) || rec.conf < minConf) return;
    const words = toWords(rec).map(w => {
      const along = (f) => [r.cx + (f - 0.5) * r.w * r.ex[0] + 0.25 * r.h * r.ey[0], r.cy + (f - 0.5) * r.w * r.ex[1] + 0.25 * r.h * r.ey[1]];
      return { text: w.text, a: along(w.f0), b: along(w.f1), size: 0.6 * r.h };
    });
    lines.push({ text, conf: rec.conf, rect: r, words });
  });
  return lines;
}
