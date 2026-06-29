// Tiny toy "Transformer-like" generator in Rust (no external crates).
// This is not a real trained transformer.
// It demonstrates the structure: logits -> softmax -> sample next token.

mod llm_toy_tokenizer;

use llm_toy_tokenizer::build_vocab;

fn softmax(xs: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &x in xs {
        if x > m { m = x; }
    }

    let mut exps = Vec::with_capacity(xs.len());
    let mut sum = 0.0f32;
    for &x in xs {
        let e = (x - m).exp();
        exps.push(e);
        sum += e;
    }
    let sum = sum + 1e-12;
    exps.into_iter().map(|e| e / sum).collect()
}

fn sample_from_probs(probs: &[f32], rng_state: &mut u64) -> usize {
    // Simple deterministic PRNG (xorshift64*)
    *rng_state ^= *rng_state >> 12;
    *rng_state ^= *rng_state << 25;
    *rng_state ^= *rng_state >> 27;
    let r = (*rng_state).wrapping_mul(2685821657736338717u64);
    let f = (r as f64 / u64::MAX as f64) as f32; // [0,1)

    let mut c = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        c += p;
        if f <= c {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

fn logits_for_next_token(prompt: &str, vocab_size: usize, rng_state: &mut u64) -> Vec<f32> {
    let base = prompt.len() % std::cmp::max(1, vocab_size);

    let mut logits = Vec::with_capacity(vocab_size);
    for j in 0..vocab_size {
        // pseudo distance-based logits + tiny noise
        let dist = ((j as i32 - base as i32).abs() % vocab_size as i32) as f32;

        // noise from rng
        *rng_state ^= *rng_state >> 12;
        *rng_state ^= *rng_state << 25;
        *rng_state ^= *rng_state >> 27;
        let r = (*rng_state).wrapping_mul(2685821657736338717u64);
        let noise = ((r % 1000) as f32 / 1000.0 - 0.5) * 0.15;

        logits.push(-dist / 7.0 + noise);
    }
    logits
}

pub fn generate_text(prompt: &str, max_new_tokens: usize) -> String {
    let (vocab, _stoi) = build_vocab();
    let vocab_size = vocab.len();

    let mut out = prompt.to_string();
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;

    for _ in 0..max_new_tokens {
        let logits = logits_for_next_token(&out, vocab_size, &mut rng_state);
        let probs = softmax(&logits);
        let idx = sample_from_probs(&probs, &mut rng_state);
        let next_ch = vocab[idx];
        out.push(next_ch);
    }
    out
}

fn main() {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "AI".to_string());
    let result = generate_text(&prompt, 60);
    println!("{}", result);
}

