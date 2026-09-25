// Runs the OCR engine off the page's main thread so the page never freezes.
import { createEngine, recognizePage } from './paddle.js';

let engine = null;
let engineDevice = null;

self.onmessage = async ({ data: msg }) => {
  const post = (m) => self.postMessage({ id: msg.id, ...m });
  try {
    if (msg.type === 'init') {
      if (!engine || engineDevice !== msg.device) {
        engineDevice = msg.device;
        engine = createEngine('fast', { device: msg.device, onStatus: (stage, f) => post({ type: 'progress', stage, fraction: f }) });
        engine.catch(() => { engine = null; });
      }
      post({ type: 'done', device: (await engine).device });
    } else if (msg.type === 'page') {
      const bmp = msg.bitmap;
      const canvas = new OffscreenCanvas(bmp.width, bmp.height);
      canvas.getContext('2d').drawImage(bmp, 0, 0);
      bmp.close();
      const run = (e) => recognizePage(e, canvas, {
        detScale: msg.detScale,
        onProgress: (fraction, stage) => post({ type: 'progress', stage, fraction }),
      });
      let e = await engine, lines;
      try {
        lines = await run(e);
      } catch (err) {
        // A graphics card can fail mid job (driver reset, out of memory). Switch
        // to the processor for the rest of the session and redo this page.
        if (e.device !== 'gpu') throw err;
        console.warn('Graphics card failed, continuing on the processor.', err);
        engine = Promise.resolve(e.fallback);
        e = e.fallback;
        lines = await run(e);
      }
      post({ type: 'done', device: e.device, lines: lines.map(l => ({ text: l.text, words: l.words })) });
    }
  } catch (err) {
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
