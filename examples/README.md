# symproj examples

Examples for token-to-vector projection and small retrieval pipelines.

## Running

```sh
cargo run --example embed_and_search
cargo run --example embed_search_rerank
cargo run --example simd_normalize
```

Use `cargo test --examples --all-features` to compile every example.

## Task map

| Goal | Example | What to inspect |
|---|---|---|
| Encode text and search neighbors | `embed_and_search` | A hand-built codebook projects short phrases into vectors, then a `vicinity` HNSW index returns nearest phrases for new queries. |
| Add reranking stages | `embed_search_rerank` | ANN candidates are reranked with `rankops` MMR and then scored with flat ColBERT MaxSim buffers. |
| Check normalization compatibility | `simd_normalize` | `symproj` normalization is compared against `innr` normalization on the same encoded vectors. |

## Reading path

Start with `embed_and_search` to see the basic projection contract: tokens map
to rows in a codebook, phrase vectors are pooled, and the resulting vectors can
be indexed like any other embeddings. Read `embed_search_rerank` next if you
want the full retrieval loop. Use `simd_normalize` when you are checking that
normalization remains compatible with a separate vector-kernel layer.

## Expected Output

`embed_and_search` should retrieve the exact phrase match first for simple
queries:

```text
Query: "dog cat"
  0: "cat dog"  (distance=-0.0000)
  1: "cat fish"  (distance=0.0351)
```

`embed_search_rerank` prints the ANN candidates, the MMR order, and a flat
ColBERT MaxSim order:

```text
Query tokens: [0, 1, 2, 50, 51]
ANN results (distance, lower = closer):
  0: doc  0 (dist=0.0296)  tokens=[0, 1, 2, 3, 4]

Flat ColBERT reranked (MaxSim):
  0: doc  7 (maxsim=24.6244)  tokens=[50, 51, 52, 53, 54]

Ranking unchanged.
```

`simd_normalize` should keep `symproj` and `innr` normalization within floating
point tolerance:

```text
symproj simd feature: disabled

ids=[0, 1, 2]
  norm (symproj): 0.99999982
  norm (innr):    0.99999994
  max element diff: 2.98e-8
  PASSED
```
