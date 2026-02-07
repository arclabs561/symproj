# symproj

Symbolic projection (embeddings).
Maps discrete symbols to continuous vectors using a Codebook and pooling.

## Intuition

Imagine a library where every book has a call number. The call number isn't just a label; it tells you where the book sits in a 3D space. `symproj` is the system that maps "book names" (tokens) to "library coordinates" (vectors).

## Usage

```rust
use symproj::{Codebook, Projection};
use textprep::BpeTokenizer;

// 1. Load a Codebook (matrix + dimension)
let matrix = vec![...]; // flattened [vocab_size * dim]
let codebook = Codebook::new(matrix, 384).unwrap();

// 2. Create a Projection (tokenizer + codebook)
let tokenizer = BpeTokenizer::from_file("tokenizer.json")?;
let proj = Projection::new(tokenizer, codebook);

// 3. Encode text -> vector (mean pooling)
let vec = proj.encode("Hello world").unwrap();
```

## Features

- **Codebook**: Dense embedding matrix lookup.
- **Pooling**: Mean, weighted mean (SIF), and sequence output.
- **Normalization**: L2 normalization and component removal (PCA-based denoising).

## License

MIT OR Apache-2.0
