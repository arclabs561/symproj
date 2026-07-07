# symproj

[![crates.io](https://img.shields.io/crates/v/symproj.svg)](https://crates.io/crates/symproj)
[![Documentation](https://docs.rs/symproj/badge.svg)](https://docs.rs/symproj)

Codebook-based token-to-vector projection.

See [examples/README.md](examples/README.md) for runnable projection, search,
reranking, and normalization examples.

## Usage

```rust
use symproj::{Codebook, Projection};
use textprep::VocabTokenizer;

// 1. Load a Codebook (flattened [vocab_size * dim] matrix + dimension)
let matrix = vec![/* vocab_size * dim f32 values */];
let codebook = Codebook::new(matrix, 384).unwrap();

// 2. Build a tokenizer over your vocab, then a Projection (tokenizer + codebook)
let tokenizer = VocabTokenizer::from_vocab(vocab); // vocab: HashMap<String, u32>
let proj = Projection::new(tokenizer, codebook);

// 3. Encode text -> vector (mean pooling over token embeddings)
let vec = proj.encode("Hello world");
```

## Features

- **Codebook**: Dense embedding matrix lookup.
- **Pooling**: Mean, weighted mean (SIF), and sequence output.
- **Normalization**: L2 normalization and component removal (PCA-based denoising).

## License

MIT OR Apache-2.0
