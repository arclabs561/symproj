# symproj examples

Examples for token-to-vector projection and small retrieval pipelines.

## Running

```sh
cargo run --example embed_and_search
cargo run --example embed_search_rerank
cargo run --example simd_normalize
```

Use `cargo test --examples` to compile every example.

## Task map

| Goal | Example | What to inspect |
|---|---|---|
| Encode text and search neighbors | `embed_and_search` | A hand-built codebook projects short phrases into vectors, then a `vicinity` HNSW index returns nearest phrases for new queries. |
| Add a reranking stage | `embed_search_rerank` | ANN candidates are reranked with `rankops` MMR so similar documents do not occupy the whole result list. |
| Check normalization compatibility | `simd_normalize` | `symproj` scalar normalization is compared against `innr` SIMD normalization on the same encoded vectors. |

## Reading path

Start with `embed_and_search` to see the basic projection contract: tokens map
to rows in a codebook, phrase vectors are pooled, and the resulting vectors can
be indexed like any other embeddings. Read `embed_search_rerank` next if you
want the full retrieval loop. Use `simd_normalize` when you are checking that
normalization remains compatible with a separate vector-kernel layer.
