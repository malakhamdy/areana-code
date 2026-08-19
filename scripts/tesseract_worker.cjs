#!/usr/bin/env node
'use strict';

// Line-delimited JSON worker. Images arrive as in-memory base64 PNG and are never
// written to disk. This is an availability fallback; PaddleOCR remains primary.
const readline = require('readline');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { createWorker, OEM, PSM } = require('tesseract.js');
const arabic = require('@tesseract.js-data/ara');
const english = require('@tesseract.js-data/eng');

// Tesseract needs both language files under one path. They are copied from pinned npm
// data packages into temporary model storage; identity images never touch this path.
const combinedLangPath = path.join(os.tmpdir(), 'egyid-tesseract-languages-v1');
fs.mkdirSync(combinedLangPath, { recursive: true });
for (const language of [arabic, english]) {
  const source = path.join(language.langPath, `${language.code}.traineddata.gz`);
  const destination = path.join(combinedLangPath, `${language.code}.traineddata.gz`);
  if (!fs.existsSync(destination)) fs.copyFileSync(source, destination);
}

const workerPromise = (async () => {
  const worker = await createWorker([arabic.code, english.code], OEM.LSTM_ONLY, {
    langPath: combinedLangPath,
    gzip: true,
    cacheMethod: 'none',
  });
  await worker.setParameters({
    tessedit_pageseg_mode: PSM.SINGLE_BLOCK,
    preserve_interword_spaces: '1',
    user_defined_dpi: '300',
  });
  return worker;
})();

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let queue = Promise.resolve();

async function handle(line) {
  let message;
  try {
    message = JSON.parse(line);
    if (message.command === 'terminate') {
      const worker = await workerPromise;
      await worker.terminate();
      process.exit(0);
    }
    const worker = await workerPromise;
    const image = Buffer.from(message.image_base64, 'base64');
    const { data } = await worker.recognize(image);
    process.stdout.write(JSON.stringify({
      id: message.id,
      ok: true,
      text: data.text || '',
      confidence: Number(data.confidence || 0) / 100,
    }) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: message && message.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }) + '\n');
  }
}

input.on('line', (line) => {
  if (line.trim()) queue = queue.then(() => handle(line));
});
