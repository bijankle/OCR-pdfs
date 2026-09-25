// PDF OCR: runs entirely in the browser. Pages are rendered with pdf.js, read
// with PaddleOCR (PP-OCRv5 models on ONNX Runtime Web, see paddle.js), and an
// invisible text layer is written onto the ORIGINAL pages with pdf-lib, so the
// visible content, vector linework and file quality are left untouched.

import * as pdfjsLib from './vendor/pdfjs/pdf.min.mjs';
import {
  PDFDocument, StandardFonts, TextRenderingMode,
  pushGraphicsState, popGraphicsState, beginText, endText,
  setFontAndSize, setTextRenderingMode, setTextMatrix, setCharacterSqueeze, showText,
} from './vendor/pdf-lib/pdf-lib.esm.min.js';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdfjs/pdf.worker.min.mjs', import.meta.url).href;

const MAX_PIXELS = 64e6;     // cap on rendered page size (A0 sheets get less than the chosen dpi)
const MAX_SIDE = 16000;

// Text is found on a copy at 2/3 of the scan resolution (fast, and tested best
// on both typed pages and A1 drawings), then read from the full resolution scan.
const DET_SCALE = 0.67;

// Characters the standard PDF font cannot hold, mapped to near equivalents.
const SUBST = { '\u2300': '\u00d8', '\u2212': '-', '\u2010': '-', '\u2011': '-', '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"' };

// The OCR engine lives in a Web Worker so the page stays responsive.
let worker = null, nextId = 0;
const calls = new Map();
function callWorker(msg, transfer, onProgress) {
  if (!worker) {
    worker = new Worker(new URL('./ocr-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      const c = calls.get(data.id);
      if (!c) return;
      if (data.type === 'progress') c.onProgress && c.onProgress(data.fraction, data.stage);
      else { calls.delete(data.id); data.type === 'error' ? c.reject(new Error(data.message)) : c.resolve(data); }
    };
    worker.onerror = (e) => {
      for (const c of calls.values()) c.reject(new Error(e.message || 'The OCR engine stopped unexpectedly'));
      calls.clear(); worker = null;
    };
  }
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    calls.set(id, { resolve, reject, onProgress });
    worker.postMessage({ ...msg, id }, transfer || []);
  });
}

// ---------------------------------------------------------------- page work

// Text the page already has, as PDF space boxes. OCR words that land on
// existing text are dropped so nothing is doubled.
async function existingText(page) {
  const tc = await page.getTextContent();
  let chars = 0;
  const boxes = [];
  for (const it of tc.items) {
    if (!it.str || !it.str.trim()) continue;
    chars += it.str.trim().length;
    const [a, b, c, d, e, f] = it.transform;
    const h = Math.hypot(c, d) || it.height;
    const ux = a / (Math.hypot(a, b) || 1), uy = b / (Math.hypot(a, b) || 1);
    const pts = [[e, f], [e + ux * it.width, f + uy * it.width], [e - uy * h, f + ux * h], [e + ux * it.width - uy * h, f + uy * it.width + ux * h]];
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    boxes.push([Math.min(...xs) - 1, Math.min(...ys) - 0.3 * h, Math.max(...xs) + 1, Math.max(...ys) + 0.3 * h]);
  }
  return { chars, boxes };
}

async function renderPage(page, dpi) {
  const base = page.getViewport({ scale: 1 });
  let scale = dpi / 72;
  scale = Math.min(scale, Math.sqrt(MAX_PIXELS / (base.width * base.height)), MAX_SIDE / Math.max(base.width, base.height));
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(viewport.width);
  canvas.height = Math.ceil(viewport.height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport, annotationMode: pdfjsLib.AnnotationMode.ENABLE }).promise;
  return { canvas, viewport, scale };
}

// Write invisible text (render mode 3, same technique OCRmyPDF uses) onto a page.
function writeTextLayer(pdfPage, font, fontKey, charset, result) {
  const { words, viewport, scale, existing = [] } = result;
  const ops = [pushGraphicsState(), beginText(), setTextRenderingMode(TextRenderingMode.Invisible)];
  let count = 0;
  for (const w of words) {
    const text = [...w.text].map(ch => SUBST[ch] || ch).map(ch => (charset.has(ch.codePointAt(0)) ? ch : '?')).join('');
    const [ax, ay] = viewport.convertToPdfPoint(w.a[0], w.a[1]);
    const [bx, by] = viewport.convertToPdfPoint(w.b[0], w.b[1]);
    const len = Math.hypot(bx - ax, by - ay);
    const size = w.size / scale;
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    if (existing.some(r => mx >= r[0] && mx <= r[2] && my >= r[1] && my <= r[3])) continue;
    const natural = font.widthOfTextAtSize(text, size);
    if (len <= 0 || size <= 0 || natural <= 0) continue;
    const cos = (bx - ax) / len, sin = (by - ay) / len;
    ops.push(
      setFontAndSize(fontKey, size),
      setCharacterSqueeze(100 * len / natural),
      setTextMatrix(cos, sin, -sin, cos, ax, ay),
      showText(font.encodeText(text)),
    );
    count++;
  }
  ops.push(endText(), popGraphicsState());
  if (count) pdfPage.pushOperators(...ops);
  return count;
}

// ---------------------------------------------------------------- whole file

async function processFile(file, opts, ui) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  ui.status('Opening PDF');
  const pdf = await pdfjsLib.getDocument({ data: bytes.slice() }).promise;
  const out = await PDFDocument.load(bytes, { ignoreEncryption: true, updateMetadata: false });
  const font = await out.embedFont(StandardFonts.Helvetica);
  const charset = new Set(font.getCharacterSet());
  const pages = out.getPages();
  const n = pdf.numPages;

  // Bar: 0 to 10% starting the engine, 10 to 95% pages, then saving.
  const bar = (f) => ui.bar(f);
  const { device } = await callWorker({ type: 'init', device: opts.gpu ? 'auto' : 'cpu' }, [], (f, stage) => { bar(0.1 * f); ui.status(`${stage} ${Math.round(100 * f)}%`); });
  let on = device === 'gpu' ? 'graphics card' : 'processor';
  bar(0.1);
  let done = 0, skipped = 0, wordsTotal = 0;

  // Render the next page while the current one is being read.
  const load = async (i) => {
    const page = await pdf.getPage(i);
    const existing = await existingText(page);
    if (opts.skipText && existing.chars > 300) return { page, skip: true };
    return { page, existing, ...(await renderPage(page, opts.dpi)) };
  };
  ui.status(`Page 1 of ${n}: preparing`);
  let pending = load(1);
  for (let i = 1; i <= n; i++) {
    const cur = await pending;
    if (i < n) pending = load(i + 1);
    ui.status(`Page ${i} of ${n}: finding text`);
    if (cur.skip) {
      skipped++;
    } else {
      const bitmap = await createImageBitmap(cur.canvas);
      const res = await callWorker({ type: 'page', bitmap, detScale: DET_SCALE }, [bitmap], (f, stage) => {
        bar(0.1 + 0.85 * (done + f) / n);
        ui.status(`Page ${i} of ${n}: ${stage.toLowerCase()} ${Math.round(100 * f)}% on ${on}`);
      });
      const { lines } = res;
      on = res.device === 'gpu' ? 'graphics card' : 'processor';
      const pdfPage = pages[i - 1];
      pdfPage.node.normalize();
      const fontKey = pdfPage.node.newFontDictionary(font.name, font.ref);
      // Each word carries a real space that stretches to the next word, so
      // viewers never glue neighbouring words together.
      const words = lines.flatMap(l => l.words.map((w, k, all) => (k < all.length - 1 ? { ...w, text: w.text + ' ', b: all[k + 1].a } : w)));
      wordsTotal += writeTextLayer(pdfPage, font, fontKey, charset, { words, viewport: cur.viewport, scale: cur.scale, existing: cur.existing.boxes });
      cur.canvas.width = cur.canvas.height = 0;
    }
    cur.page.cleanup();
    done++;
    bar(0.1 + 0.85 * done / n);
  }

  ui.status('Saving');
  bar(0.97);
  const saved = await out.save({ useObjectStreams: true });
  await pdf.destroy();
  return { blob: new Blob([saved], { type: 'application/pdf' }), pages: n, skipped, words: wordsTotal, on };
}

// ---------------------------------------------------------------- interface

const $ = (s) => document.querySelector(s);
const drop = $('#drop');
const input = $('#file');
const list = $('#jobs');
const jobs = [];       // every file added this session, in order
let busy = false;
let paused = false;

function readOptions() {
  return { dpi: +$('#dpi').value, skipText: $('#skip').checked, gpu: $('#gpu').checked };
}

const counts = () => ({
  total: jobs.length,
  done: jobs.filter(j => j.status === 'done').length,
  fresh: jobs.filter(j => j.status === 'done' && !j.downloaded).length,
  waiting: jobs.filter(j => j.status === 'waiting').length,
  running: jobs.filter(j => j.status === 'running').length,
  failed: jobs.filter(j => j.status === 'failed').length,
});

// Batch panel: overall progress and bulk actions.
function refresh() {
  const c = counts();
  $('#batch').hidden = c.total === 0;
  const parts = [`${c.done} of ${c.total} done`];
  if (c.running) parts.push(paused ? 'finishing current file, then pausing' : 'working');
  if (c.waiting) parts.push(`${c.waiting} waiting${paused ? ' (paused)' : ''}`);
  if (c.failed) parts.push(`${c.failed} failed`);
  $('#batch-summary').textContent = parts.join(', ');
  const running = jobs.find(j => j.status === 'running');
  const overall = c.total ? (c.done + c.failed + (running ? running.fraction || 0 : 0)) / c.total : 0;
  $('#batch-bar').style.width = `${(100 * overall).toFixed(1)}%`;
  const zip = $('#zip');
  zip.disabled = c.fresh === 0;
  zip.textContent = c.fresh === 0
    ? (c.done ? 'All finished files downloaded' : 'Download finished files')
    : `Download ${c.fresh} ${c.done > c.fresh ? 'new ' : ''}finished file${c.fresh === 1 ? '' : 's'} (ZIP)`;
  const pause = $('#pause');
  pause.hidden = !(c.waiting || c.running);
  pause.textContent = paused ? 'Resume' : 'Pause';
  $('#clear').hidden = !(c.done || c.failed);
}

function addFiles(files) {
  for (const f of files) {
    if (f.type !== 'application/pdf' && !/\.pdf$/i.test(f.name)) continue;
    const li = document.createElement('li');
    li.className = 'job';
    li.innerHTML = `<div class="job-head"><span class="name"></span><span class="state">Waiting</span>
      <button class="remove" type="button" aria-label="Remove from queue" title="Remove from queue">&times;</button></div>
      <div class="bar"><div></div></div><div class="result"></div>`;
    li.querySelector('.name').textContent = f.name;
    const job = { file: f, li, opts: readOptions(), status: 'waiting', fraction: 0, downloaded: false };
    li.querySelector('.remove').addEventListener('click', () => {
      if (job.status !== 'waiting') return;
      jobs.splice(jobs.indexOf(job), 1);
      li.remove();
      refresh();
    });
    list.append(li);
    jobs.push(job);
  }
  refresh();
  pump();
}

function markDownloaded(job) {
  job.downloaded = true;
  job.li.classList.add('downloaded');
  refresh();
}

async function pump() {
  if (busy) return;
  busy = true;
  for (;;) {
    const job = paused ? null : jobs.find(j => j.status === 'waiting');
    if (!job) break;
    const { file, li, opts } = job;
    const state = li.querySelector('.state');
    const bar = li.querySelector('.bar > div');
    const result = li.querySelector('.result');
    job.status = 'running';
    li.classList.add('running');
    refresh();
    const t0 = performance.now();
    let label = 'Starting';
    const show = () => { state.textContent = `${label} (${Math.round((performance.now() - t0) / 1000)} s)`; };
    const timer = setInterval(show, 1000);
    try {
      const r = await processFile(file, opts, {
        status: (s) => { label = s; show(); },
        bar: (f) => { bar.style.width = `${(100 * f).toFixed(1)}%`; job.fraction = f; refresh(); },
      });
      clearInterval(timer);
      bar.style.width = '100%';
      const secs = ((performance.now() - t0) / 1000).toFixed(0);
      job.name = file.name.replace(/\.pdf$/i, '') + '_searchable.pdf';
      job.blob = r.blob;
      job.status = 'done';
      state.textContent = `Done in ${secs} s`;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(r.blob); a.download = job.name; a.className = 'download';
      a.textContent = 'Download';
      a.addEventListener('click', () => markDownloaded(job));
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = `${r.pages} page${r.pages === 1 ? '' : 's'}, ${r.words.toLocaleString()} words added`
        + (r.skipped ? `, ${r.skipped} page${r.skipped === 1 ? '' : 's'} already searchable and left as is` : '')
        + `, ${(r.blob.size / 1048576).toFixed(1)} MB, read on the ${r.on}`;
      const tick = document.createElement('span');
      tick.className = 'tick';
      tick.textContent = 'Downloaded';
      result.append(a, note, tick);
      li.classList.add('ok');
    } catch (err) {
      console.error(err);
      clearInterval(timer);
      job.status = 'failed';
      state.textContent = 'Failed';
      result.textContent = /password/i.test(err && err.message)
        ? 'This PDF is password protected. Remove the password and try again.'
        : `Could not process this file: ${err && err.message ? err.message : err}`;
      li.classList.add('err');
    }
    clearInterval(timer);
    li.classList.remove('running');
    refresh();
  }
  busy = false;
  refresh();
}

// One ZIP with every finished file not downloaded yet. PDFs are already
// compressed, so they are stored as is, which is fast.
async function downloadZip() {
  const fresh = jobs.filter(j => j.status === 'done' && !j.downloaded);
  if (!fresh.length) return;
  const btn = $('#zip');
  btn.disabled = true;
  btn.textContent = 'Preparing ZIP';
  try {
    const { zipSync } = await import('./vendor/fflate/fflate.mjs');
    const entries = {};
    const used = new Set();
    for (const j of fresh) {
      let name = j.name;
      for (let k = 2; used.has(name.toLowerCase()); k++) name = j.name.replace(/\.pdf$/i, ` (${k}).pdf`);
      used.add(name.toLowerCase());
      entries[name] = [new Uint8Array(await j.blob.arrayBuffer()), { level: 0 }];
    }
    const zipped = zipSync(entries);
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ').replace(':', '.');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([zipped], { type: 'application/zip' }));
    a.download = `searchable PDFs ${stamp}.zip`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    fresh.forEach(markDownloaded);
  } finally {
    refresh();
  }
}

$('#zip').addEventListener('click', downloadZip);
$('#pause').addEventListener('click', () => { paused = !paused; refresh(); if (!paused) pump(); });
$('#clear').addEventListener('click', () => {
  for (const j of jobs.filter(j => j.status === 'done' || j.status === 'failed')) {
    if (j.status === 'done' && !j.downloaded && !confirm(`${j.name} has not been downloaded yet. Clear it anyway?`)) continue;
    const a = j.li.querySelector('a.download');
    if (a) URL.revokeObjectURL(a.href);
    j.li.remove();
    jobs.splice(jobs.indexOf(j), 1);
  }
  refresh();
});

// Warn before closing the tab while work is queued or results are not saved.
window.addEventListener('beforeunload', (e) => {
  const c = counts();
  if (c.waiting || c.running || c.fresh) { e.preventDefault(); e.returnValue = ''; }
});

drop.addEventListener('click', () => input.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

// Exposed for automated testing.
window.__ocr = { processFile, readOptions, jobs };
