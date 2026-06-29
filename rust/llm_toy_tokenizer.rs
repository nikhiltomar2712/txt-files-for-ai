// Toy tokenizer utilities for a tiny LLM-like demo.
// No external crates: character-level encoding.

use std::collections::HashMap;

pub fn build_vocab() -> (Vec<char>, HashMap<char, usize>) {
    let chars: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_\n";
    let mut vocab: Vec<char> = Vec::new();
    for c in chars.chars() {
        if !vocab.contains(&c) {
            vocab.push(c);
        }
    }

    let mut stoi: HashMap<char, usize> = HashMap::new();
    for (i, c) in vocab.iter().enumerate() {
        // reserve 0 for <unk>
        stoi.insert(*c, i + 1);
    }
    (vocab, stoi)
}

pub fn encode(text: &str, stoi: &HashMap<char, usize>) -> Vec<usize> {
    let mut ids = Vec::new();
    for ch in text.chars() {
        ids.push(*stoi.get(&ch).unwrap_or(&0));
    }
    ids
}

pub fn decode(ids: &[usize], vocab: &[char]) -> String {
    let mut out = String::new();
    for &id in ids {
        if id == 0 {
            out.push_str("<unk>");
        } else {
            let idx = id - 1;
            if idx < vocab.len() {
                out.push(vocab[idx]);
            } else {
                out.push_str("<unk>");
            }
        }
    }
    out
}

