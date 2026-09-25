# PDF OCR

Turn scanned PDFs and CAD drawings into searchable PDFs, right in the browser.
Drop a PDF on the page and download the same PDF back with an invisible text
layer on every page, so Ctrl+F, copy and paste, and document indexing all work.

Nothing is uploaded. The OCR engine runs on your own computer inside the
browser tab, so the site is just static files on GitHub Pages.

## How it works

1. **pdf.js** renders each page at the chosen resolution (300 dpi by default).
2. **PaddleOCR PP-OCRv5** (detection model plus the English recognition model)
   runs through **ONNX Runtime Web**. It finds every text line at any angle,
   turns vertical lines upright, and reads them. Positions of individual words
   come from the recognizer's character timing.
3. **pdf-lib** writes the words onto the original page as invisible text
   (PDF render mode 3, the same technique OCRmyPDF uses). The original page
   content is not touched, so vector drawings stay sharp and the file size
   barely changes.

Text that is already real text in the PDF is kept, and OCR words that land on
it are dropped so nothing is doubled.

## Why PaddleOCR and not Tesseract

Measured with the test bench used during development (word level F1 score):

| Test page | Tesseract 5 | PaddleOCR PP-OCRv5 |
|---|---|---|
| Typed page, clean | 1.00 | 1.00 |
| Typed page, skewed 4 degrees | 0.46 | 1.00 |
| Typed page, 7 pt, 200 dpi, JPEG | 1.00 | 1.00 |
| Drawing, degraded scan | 0.71 | 0.98 |
| Drawing, 150 dpi fax | 0.69 | 0.97 |
| Vertical text on drawings | 0 of 3 | 3 of 3 |

On a real A1 P&ID sheet it read line numbers, instrument tags, vertical pipe
labels and the title block, in about 25 seconds on a 4 core machine. The larger
PP-OCRv6 medium model was also tried. It was 6 times slower and mixed up O and
0, I and 1 on engineering tags, so it is not used.

## Files

| Path | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | The page, PDF handling and text layer writing |
| `paddle.js` | The PaddleOCR pipeline (detection, orientation, recognition) |
| `ocr-worker.js` | Runs the pipeline in a background thread so the page never freezes |
| `models/` | PP-OCRv5 mobile detection, English PP-OCRv5 mobile recognition, text line orientation model |
| `vendor/` | ONNX Runtime Web, pdf.js (legacy build for older browsers), pdf-lib |
| `coi-serviceworker.min.js` | Lets GitHub Pages run the engine on all CPU cores |

## Publishing

Settings, Pages, Source: "Deploy from a branch", branch `main`, folder `/ (root)`.

## Licenses

PaddleOCR models: Apache 2.0. ONNX Runtime Web: MIT. pdf.js: Apache 2.0.
pdf-lib: MIT. coi-serviceworker: MIT.
