//! Compare symproj's `l2_normalize_in_place` with innr's SIMD-accelerated `normalize`.
//!
//! Both should produce identical unit vectors. innr dispatches to NEON/AVX2/AVX-512
//! at runtime; symproj uses scalar code. This example verifies they agree.

use innr::{norm, normalize as innr_normalize};
use symproj::{l2_normalize_in_place, Codebook};

fn main() {
    let dim = 64;
    let vocab_size = 8;

    // Deterministic pseudo-random matrix (LCG).
    let mut matrix = Vec::with_capacity(vocab_size * dim);
    let mut state: u64 = 42;
    for _ in 0..vocab_size * dim {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
        matrix.push(val);
    }

    let codebook = Codebook::new(matrix, dim).expect("valid codebook");
    println!(
        "Codebook: vocab_size={}, dim={dim}\n",
        codebook.vocab_size()
    );

    // Encode a few token-id sequences and normalize both ways.
    let sequences: &[&[u32]] = &[&[0, 1, 2], &[3, 4, 5, 6, 7], &[0, 7], &[2]];

    for ids in sequences {
        let raw = codebook.encode_ids(ids);

        // symproj path
        let mut v_symproj = raw.clone();
        l2_normalize_in_place(&mut v_symproj);

        // innr SIMD path
        let mut v_innr = raw.clone();
        innr_normalize(&mut v_innr);

        // Both should be unit vectors.
        let norm_symproj = norm(&v_symproj);
        let norm_innr = norm(&v_innr);

        // Element-wise max absolute difference.
        let max_diff: f32 = v_symproj
            .iter()
            .zip(v_innr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("ids={ids:?}");
        println!("  norm (symproj): {norm_symproj:.8}");
        println!("  norm (innr):    {norm_innr:.8}");
        println!("  max element diff: {max_diff:.2e}");
        assert!(
            max_diff < 1e-6,
            "normalization methods diverged: max_diff={max_diff}"
        );
        println!("  PASSED\n");
    }

    println!("All vectors match between symproj and innr normalization.");
}
