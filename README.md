# symproj

[![crates.io](https://img.shields.io/crates/v/symproj.svg)](https://crates.io/crates/symproj)
[![Documentation](https://docs.rs/symproj/badge.svg)](https://docs.rs/symproj)

Codebook-based token-to-vector projection.

See [examples/README.md](examples/README.md) for runnable projection, search,
reranking, and normalization examples.

## Usage

```rust
use std::collections::HashMap;
use symproj::{Codebook, Projection};
use textprep::VocabTokenizer;

let mut vocab = HashMap::new();
vocab.insert("hello".to_string(), 0);
vocab.insert("world".to_string(), 1);

let tokenizer = VocabTokenizer::from_vocab(vocab);
let codebook = Codebook::new(vec![
    1.0, 0.0, // hello
    0.0, 1.0, // world
], 2).unwrap();
let proj = Projection::new(tokenizer, codebook);

let vector = proj.encode("hello world");
assert_eq!(vector, vec![0.5, 0.5]);
```

## License

MIT OR Apache-2.0
