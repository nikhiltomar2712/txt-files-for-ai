// Bigram language model demo in Rust (no external crates).
// Trains on a small text corpus and generates by sampling bigram probabilities.

use std::collections::HashMap;

fn preprocess(text: &str) -> Vec<char> {
    text.chars().filter(|c| !c.is_control()).collect()
}

fn train_bigram(corpus: &[char]) -> (Vec<char>, HashMap<(char, char), usize>, HashMap<char, usize>) {
    let mut vocab: Vec<char> = Vec::new();
    for &c in corpus {
        if !vocab.contains(&c) {
            vocab.push(c);
        }
    }

    let mut pair_counts: HashMap<(char, char), usize> = HashMap::new();
    let mut prev_counts: HashMap<char, usize> = HashMap::new();

    for w in corpus.windows(2) {
        let a = w[0];
        let b = w[1];
        *prev_counts.entry(a).or_insert(0) += 1;
        *pair_counts.entry((a, b)).or_insert(0) += 1;
    }

    (vocab, pair_counts, prev_counts)
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(2685821657736338717u64)
}

fn sample_next(prev: char, vocab: &[char], pair_counts: &HashMap<(char, char), usize>, prev_counts: &HashMap<char, usize>, rng: &mut u64) -> char {
    let total = *prev_counts.get(&prev).unwrap_or(&0);
    if total == 0 {
        // fallback uniform
        let idx = (xorshift64(rng) as usize) % vocab.len().max(1);
        return vocab[idx];
    }

    // roulette wheel
    let r = (xorshift64(rng) % total as u64) as usize;
    let mut cumsum = 0usize;
    for &cand in vocab {
        let cnt = *pair_counts.get(&(prev, cand)).unwrap_or(&0);
        cumsum += cnt;
        if r < cumsum {
            return cand;
        }
    }

    // last resort
    vocab[0]
}

pub fn generate_bigram(prompt: &str, steps: usize) -> String {
    let corpus = preprocess(
        "AI is intelligence demonstrated by machines. "
        "Machine learning builds models from data. "
        "Neural networks learn patterns. "
        "Deep learning uses many layers."
    );

    let (vocab, pair_counts, prev_counts) = train_bigram(&corpus);

    let mut rng_state: u64 = 0xfeed_beef_cafe_babe;
    let mut out: Vec<char> = prompt.chars().collect();
    if out.is_empty() {
        out.push('A');
    }

    let mut prev = *out.last().unwrap();
    for _ in 0..steps {
        let next = sample_next(prev, &vocab, &pair_counts, &prev_counts, &mut rng_state);
        out.push(next);
        prev = next;
    }

    out.into_iter().collect()
}

fn main() {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "AI".to_string());
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let result = generate_bigram(&prompt, steps);
    println!("{}", result);
}

