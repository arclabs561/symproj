//! Embed-then-search pipeline: symproj produces vectors, vicinity finds neighbors.
//!
//! Demonstrates:
//!   1. Build a Codebook from hand-crafted embeddings
//!   2. Wrap it in a Projection (tokenizer + codebook)
//!   3. Encode phrases into vectors
//!   4. Insert vectors into a vicinity HNSW index
//!   5. Query with a new phrase and print nearest neighbors

use std::collections::HashMap;
use symproj::{l2_normalize_in_place, Codebook, Projection};
use textprep::BpeTokenizer;
use vicinity::hnsw::HNSWIndex;

fn main() {
    // -- Vocabulary ----------------------------------------------------------
    // Each word maps to a token ID. The codebook below assigns a vector to
    // each ID. In a real system these come from trained embeddings; here we
    // use hand-picked 4-D vectors that cluster by semantic domain.
    let vocab: HashMap<String, u32> = [
        ("cat", 0),
        ("dog", 1),
        ("fish", 2),
        ("bird", 3),
        ("car", 4),
        ("truck", 5),
        ("bike", 6),
        ("bus", 7),
        ("red", 8),
        ("blue", 9),
        ("fast", 10),
        ("slow", 11),
    ]
    .into_iter()
    .map(|(w, id)| (w.to_string(), id))
    .collect();

    // -- Codebook (4-D embeddings) -------------------------------------------
    // Vectors are laid out so that animals cluster together, vehicles cluster
    // together, and modifiers sit between.
    #[rustfmt::skip]
    let matrix: Vec<f32> = vec![
        // animals (high dim-0, low dim-1)
         0.9,  0.1,  0.2,  0.1, // 0: cat
         0.8,  0.2,  0.1,  0.2, // 1: dog
         0.7,  0.0,  0.5,  0.1, // 2: fish
         0.8,  0.1,  0.4,  0.3, // 3: bird
        // vehicles (low dim-0, high dim-1)
         0.1,  0.9,  0.1,  0.3, // 4: car
         0.2,  0.8,  0.2,  0.4, // 5: truck
         0.1,  0.7,  0.3,  0.1, // 6: bike
         0.2,  0.9,  0.1,  0.2, // 7: bus
        // modifiers (mid-range)
         0.4,  0.4,  0.8,  0.1, // 8: red
         0.3,  0.3,  0.1,  0.9, // 9: blue
         0.5,  0.5,  0.3,  0.6, // 10: fast
         0.4,  0.4,  0.6,  0.3, // 11: slow
    ];

    let dim = 4;
    let codebook = Codebook::new(matrix, dim).expect("valid codebook");
    let tokenizer = BpeTokenizer::from_vocab(vocab);
    let proj = Projection::new(tokenizer, codebook);

    // -- Corpus --------------------------------------------------------------
    let phrases = [
        "cat dog",
        "fish bird",
        "cat fish",
        "car truck",
        "bike bus",
        "car bus",
        "red cat",
        "blue car",
        "fast dog",
        "slow truck",
        "fast bike",
        "red bird",
    ];

    // -- Encode and insert into HNSW -----------------------------------------
    // HNSWIndex::new(dimension, m, m_max)
    let mut index = HNSWIndex::new(dim, 8, 16).expect("valid HNSW params");

    for (i, phrase) in phrases.iter().enumerate() {
        let mut v = proj.encode(phrase);
        l2_normalize_in_place(&mut v);
        index.add(i as u32, v).expect("add should succeed");
    }
    index.build().expect("build should succeed");

    println!("Indexed {} phrases into HNSW (dim={dim})\n", phrases.len());

    // -- Query ---------------------------------------------------------------
    let queries = ["dog cat", "truck bus", "blue fish", "fast car"];
    let k = 4;
    let ef_search = 32;

    for query in &queries {
        let mut qv = proj.encode(query);
        l2_normalize_in_place(&mut qv);

        let results = index.search(&qv, k, ef_search).expect("search ok");

        println!("Query: \"{query}\"");
        for (rank, &(doc_id, dist)) in results.iter().enumerate() {
            let phrase = phrases[doc_id as usize];
            println!("  {rank}: \"{phrase}\"  (distance={dist:.4})");
        }
        println!();
    }
}
