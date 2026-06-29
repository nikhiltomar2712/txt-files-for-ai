# Rust (toy LLM demos)

This folder contains small **no-dependency** Rust examples that mimic the structure of an LLM generator.

## Run

Because these files avoid Cargo dependencies, you can compile directly:

```bash
rustc llm_toy_transformer.rs -O
./llm_toy_transformer "Hello AI"
```

## Files
- `llm_toy_tokenizer.rs` - character-level tokenizer helpers
- `llm_toy_transformer.rs` - toy generator: logits -> softmax -> sample

