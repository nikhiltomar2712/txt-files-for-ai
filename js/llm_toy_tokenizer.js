/*
Toy tokenizer utilities (no external dependencies).
This is a tiny helper for the other JS LLM demos in this repo.
*/

function buildVocab() {
  // Simple character vocabulary
  const vocab = [];
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_\n";
  for (const c of chars) {
    if (!vocab.includes(c)) vocab.push(c);
  }
  // Reserve 0 for <unk>
  return { vocab, stoi: Object.fromEntries(vocab.map((c, i) => [c, i + 1])), itos: ["<unk>", ...vocab] };
}

function encode(text, vocab) {
  const { stoi } = vocab;
  const ids = [];
  for (const ch of text) {
    ids.push(stoi[ch] ?? 0);
  }
  return ids;
}

function decode(ids, vocab) {
  const { itos } = vocab;
  return ids.map((id) => itos[id] ?? "<unk>").join("");
}

module.exports = { buildVocab, encode, decode };

