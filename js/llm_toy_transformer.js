/*
Tiny toy “Transformer-like” LLM (JS, no dependencies).
Not a real training implementation—intended for learning structure.
*/

const { buildVocab } = require("./llm_toy_tokenizer");

function softmax(xs) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0) + 1e-12;
  return exps.map((e) => e / s);
}

function sampleFromProbs(probs) {
  let r = Math.random();
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (r <= c) return i;
  }
  return probs.length - 1;
}

// A toy scoring function: assigns a pseudo-logit to each character.
function logitsForNextToken(prompt, vocabSize) {
  const base = prompt.length % Math.max(1, vocabSize);
  const logits = [];
  for (let j = 0; j < vocabSize; j++) {
    const dist = Math.abs((j - base) % vocabSize);
    const noise = (Math.random() - 0.5) * 0.15;
    logits.push(-dist / 7.0 + noise);
  }
  return logits;
}

function generateText(prompt, maxNewTokens = 60) {
  const vocab = buildVocab();
  const vocabSize = vocab.vocab.length;

  let out = prompt;
  for (let t = 0; t < maxNewTokens; t++) {
    const logits = logitsForNextToken(out, vocabSize);
    const probs = softmax(logits);
    const idx = sampleFromProbs(probs);
    const nextCh = vocab.vocab[idx];
    out += nextCh;
  }
  return out;
}

if (require.main === module) {
  const prompt = process.argv[2] ?? "AI";
  const result = generateText(prompt, 50);
  console.log(result);
}

module.exports = { generateText };

