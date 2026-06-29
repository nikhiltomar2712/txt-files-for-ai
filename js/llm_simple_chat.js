/*
Toy LLM-like chat demo (NOT a real LLM).

This demonstrates the *shape* of an LLM generation loop in JavaScript:
- tokenization (toy)
- context window
- greedy token selection

Run:
  node llm_simple_chat.js
*/

function tokenize(text) {
  return text.trim().split(/\s+/).filter(Boolean);
}

function detokenize(tokens) {
  return tokens.join(' ');
}

class ToyLLM {
  constructor(vocab) {
    this.vocab = vocab;
    this.v2i = new Map(vocab.map((w, i) => [w, i]));
  }

  nextTokenLogits(contextTokens) {
    // Create toy "scores": prefer tokens that appear in the recent context.
    const logits = Array(this.vocab.length).fill(-10);

    for (const tok of contextTokens.slice(-20)) {
      if (this.v2i.has(tok)) logits[this.v2i.get(tok)] = 3;
    }

    const specials = [
      'the', 'a', 'to', 'and', 'I', 'you', 'AI', 'build', 'model', 'data',
      'train', 'learn', 'generate', 'context', '.', '?', '!' 
    ];
    for (const s of specials) {
      if (this.v2i.has(s)) logits[this.v2i.get(s)] = Math.max(logits[this.v2i.get(s)], 1);
    }

    return logits;
  }

  softmax(logits, temperature = 0.9) {
    const t = Math.max(1e-6, temperature);
    const exps = logits.map(l => Math.exp(l / t));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  generate(prompt, { maxNewTokens = 60, temperature = 0.8, contextWindow = 120 } = {}) {
    let tokens = tokenize(prompt);
    let context = tokens.slice(-contextWindow);
    const generated = [];

    for (let i = 0; i < maxNewTokens; i++) {
      const logits = this.nextTokenLogits(context);
      const probs = this.softmax(logits, temperature);
      // Greedy pick the most probable token (simple)
      let bestIdx = 0;
      for (let j = 1; j < probs.length; j++) if (probs[j] > probs[bestIdx]) bestIdx = j;

      const nextTok = this.vocab[bestIdx];
      generated.push(nextTok);

      context.push(nextTok);
      if (context.length > contextWindow) context = context.slice(-contextWindow);

      if (['.', '?', '!'].includes(nextTok)) break;
    }

    return detokenize(tokens.concat(generated));
  }
}

function buildDemoVocab() {
  const base = [
    'I','you','we','AI','artificial','intelligence','learn','learning','model','data',
    'train','training','loss','optimize','build','pipeline','predict','token','context',
    'generate','response','the','a','to','and','for','with','in','on','of','it','is','are',
    'can','will','should','.','?','!','begin','steps','first','next','then','finally',
    'example','how','does','work','explain','create','system','dataset','evaluation','validate',
    'deploy','code','python','js','web','app'
  ];

  // ensure unique
  return Array.from(new Set(base));
}

function main() {
  const vocab = buildDemoVocab();
  const llm = new ToyLLM(vocab);

  const readline = require('node:readline');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log('Toy LLM Chat Demo (not a real LLM). Type exit to quit.\n');

  rl.setPrompt('You: ');
  rl.prompt();

  rl.on('line', (line) => {
    const user = line.trim();
    if (['exit', 'quit', 'q'].includes(user.toLowerCase())) {
      rl.close();
      return;
    }

    const prompt =
      'You are learning how AI works. Explain clearly and give a small JS-code idea. ' +
      'User request: ' + user;

    const out = llm.generate(prompt, { maxNewTokens: 70, temperature: 0.8 });
    console.log('Assistant:', out);
    console.log();
    rl.prompt();
  });
}

main();

