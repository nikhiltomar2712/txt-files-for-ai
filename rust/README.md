# Rust (toy AI / LLM demos)

This folder contains small **no-dependency** Rust examples to illustrate AI/LLM ideas:

## Run (no Cargo)

### 1) Toy "Transformer-like" generator
```bash
rustc llm_toy_transformer.rs -O
./llm_toy_transformer "Hello AI"
```

### 2) Bigram language model
```bash
rustc llm_bigram_lm.rs -O
./llm_bigram_lm "AI" 80
```

### 3) Small MLP classifier (shows training pipeline)
```bash
rustc llm_mlp_classifier.rs -O
./llm_mlp_classifier
```

## Files
- `llm_toy_tokenizer.rs` - character-level tokenizer helpers
- `llm_toy_transformer.rs` - toy generator: logits -> softmax -> sample
- `llm_bigram_lm.rs` - bigram LM: learn counts -> sample next char
- `llm_mlp_classifier.rs` - tiny MLP with SGD training

