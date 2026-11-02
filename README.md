# symproj

[![crates.io](https://img.shields.io/crates/v/symproj.svg)](https://crates.io/crates/symproj)
[![Documentation](https://docs.rs/symproj/badge.svg)](https://docs.rs/symproj)

Codebook-based token-to-vector projection.

Supply a vocabulary and embedding matrix to look up token vectors or pool them
into a text vector. Pooling supports a mean or SIF weights (Arora et al. 2017).

See [examples/README.md](examples/README.md) for runnable projection, search,
reranking, and normalization examples.

## Install

The example uses `textprep` to tokenize text:

```toml
[dependencies]
symproj = "0.1.5"
textprep = "0.1.6"
```

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

`cargo run --example embed_and_search` encodes 12 two-word phrases with a
hand-built 4-dimensional codebook, indexes them in HNSW, and queries new
phrases. One query from its output:

```text
Query: "truck bus"
  0: "car truck"  (distance=0.0028)
  1: "car bus"  (distance=0.0055)
  2: "bike bus"  (distance=0.0144)
  3: "fast bike"  (distance=0.0543)
```

## License

MIT OR Apache-2.0
