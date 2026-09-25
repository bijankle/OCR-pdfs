// Runs the OCR engine off the page's main thread so the page never freezes.
import { createEngine, recognizePage } from './paddle.js';

let engine = null;

self.onmessage = async ({ data: msg }) => {
  const post = (m) => self.postMessage({ id: msg.id, ...m });
  try {
    if (msg.type === 'init') {
      if (!engine) engine = createEngine('fast', { onStatus: (stage, f) => post({ type: 'progress', stage, fraction: f }) });
      await engine;
      post({ type: 'done' });
    } else if (msg.type === 'page') {
      const bmp = msg.bitmap;
      const canvas = new OffscreenCanvas(bmp.width, bmp.height);
      canvas.getContext('2d').drawImage(bmp, 0, 0);
      bmp.close();
      const lines = await recognizePage(await engine, canvas, {
        detScale: msg.detScale,
        onProgress: (fraction, stage) => post({ type: 'progress', stage, fraction }),
      });
      post({ type: 'done', lines: lines.map(l => ({ text: l.text, words: l.words })) });
    }
  } catch (err) {
    engine = msg.type === 'init' ? null : engine;
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
