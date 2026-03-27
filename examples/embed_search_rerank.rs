//! Full embed -> search -> rerank pipeline using symproj, vicinity, and rankops.
//!
//! Steps:
//!   1. Create a Codebook with synthetic embeddings (100 vocab, dim=64)
//!   2. Encode "documents" (token-ID sequences) via mean pooling
//!   3. L2-normalize all document vectors
//!   4. Build a vicinity HNSW index
//!   5. Search for a query
//!   6. Rerank with rankops MMR for diversity
//!   7. Print before/after rankings

use rankops::{mmr_embeddings, MmrConfig};
use symproj::{l2_normalize_in_place, Codebook};
use vicinity::hnsw::HNSWIndex;

/// Deterministic pseudo-random f32 in [-0.5, 0.5) from an LCG.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 33) as f32) / (u32::MAX as f32) - 0.5
}

fn main() {
    // ── 1. Codebook ──────────────────────────────────────────────────────
    let dim = 64;
    let vocab_size = 100;

    let mut state: u64 = 123;
    let matrix: Vec<f32> = (0..vocab_size * dim)
        .map(|_| lcg_next(&mut state))
        .collect();
    let codebook = Codebook::new(matrix, dim).expect("valid codebook");
    println!("Codebook: vocab_size={}, dim={dim}", codebook.vocab_size());

    // ── 2. Documents (token-ID sequences) ────────────────────────────────
    // Each "document" is a short sequence of token IDs drawn from the vocab.
    let documents: Vec<Vec<u32>> = vec![
        vec![0, 1, 2, 3, 4],
        vec![5, 6, 7, 8, 9],
        vec![10, 11, 12, 13, 14],
        vec![0, 5, 10, 15, 20],
        vec![1, 6, 11, 16, 21],
        vec![2, 7, 12, 17, 22],
        vec![3, 8, 13, 18, 23],
        vec![50, 51, 52, 53, 54],
        vec![60, 61, 62, 63, 64],
        vec![70, 71, 72, 73, 74],
        vec![80, 81, 82, 83, 84],
        vec![90, 91, 92, 93, 94],
        vec![0, 10, 20, 30, 40],
        vec![1, 11, 21, 31, 41],
        vec![2, 12, 22, 32, 42],
        vec![25, 35, 45, 55, 65],
        vec![26, 36, 46, 56, 66],
        vec![27, 37, 47, 57, 67],
        vec![28, 38, 48, 58, 68],
        vec![99, 98, 97, 96, 95],
    ];

    // ── 3. Encode + normalize ────────────────────────────────────────────
    let doc_vecs: Vec<Vec<f32>> = documents
        .iter()
        .map(|ids| {
            let mut v = codebook.encode_ids(ids);
            l2_normalize_in_place(&mut v);
            v
        })
        .collect();

    println!("Encoded {} documents\n", doc_vecs.len());

    // ── 4. Build HNSW index ──────────────────────────────────────────────
    let mut index = HNSWIndex::new(dim, 8, 16).expect("valid HNSW params");
    for (i, v) in doc_vecs.iter().enumerate() {
        index.add_slice(i as u32, v).expect("add should succeed");
    }
    index.build().expect("build should succeed");
    println!("HNSW index built ({} vectors)\n", doc_vecs.len());

    // ── 5. Search ────────────────────────────────────────────────────────
    // Query: a document similar to doc 0 (shares tokens 0,1,2).
    let query_ids: Vec<u32> = vec![0, 1, 2, 50, 51];
    let mut query_vec = codebook.encode_ids(&query_ids);
    l2_normalize_in_place(&mut query_vec);

    let k = 10;
    let ef_search = 32;
    let ann_results = index.search(&query_vec, k, ef_search).expect("search ok");

    println!("Query tokens: {query_ids:?}");
    println!("ANN results (distance, lower = closer):");
    for (rank, &(doc_id, dist)) in ann_results.iter().enumerate() {
        let tokens = &documents[doc_id as usize];
        println!("  {rank}: doc {doc_id:>2} (dist={dist:.4})  tokens={tokens:?}");
    }

    // ── 6. Rerank with MMR for diversity ─────────────────────────────────
    // Convert ANN (doc_id, distance) -> (doc_id, similarity, embedding) for MMR.
    // Distance is L2 on unit vectors: sim = 1 - dist/2 (exact for cosine).
    let candidates: Vec<(u32, f32, Vec<f32>)> = ann_results
        .iter()
        .map(|&(doc_id, dist)| {
            let sim = 1.0 - dist / 2.0;
            let emb = doc_vecs[doc_id as usize].clone();
            (doc_id, sim, emb)
        })
        .collect();

    let mmr_config = MmrConfig {
        lambda: 0.5,
        top_k: k,
    };
    let reranked = mmr_embeddings(&candidates, mmr_config);

    // ── 7. Print before/after ────────────────────────────────────────────
    println!("\nMMR reranked (lambda={}):", mmr_config.lambda);
    for (rank, (doc_id, score)) in reranked.iter().enumerate() {
        let tokens = &documents[*doc_id as usize];
        println!("  {rank}: doc {doc_id:>2} (mmr={score:.4})  tokens={tokens:?}");
    }

    // Show which documents moved.
    let ann_order: Vec<u32> = ann_results.iter().map(|&(id, _)| id).collect();
    let mmr_order: Vec<u32> = reranked.iter().map(|(id, _)| *id).collect();
    let changed = ann_order != mmr_order;
    println!(
        "\nRanking {}.",
        if changed {
            "changed (MMR promoted diversity)"
        } else {
            "unchanged"
        }
    );
}
