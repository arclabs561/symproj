use proptest::prelude::*;
use symproj::{l2_normalize_in_place, remove_component_in_place, sif_weight, Codebook};

/// Strategy for a codebook with `vocab_size` tokens of dimension `dim`.
fn codebook_strategy(vocab_size: usize, dim: usize) -> impl Strategy<Value = Codebook> {
    proptest::collection::vec(-10.0f32..10.0, vocab_size * dim)
        .prop_map(move |matrix| Codebook::new(matrix, dim).unwrap())
}

/// Strategy for a non-zero f32 vector of given length.
fn nonzero_vec_strategy(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, len)
        .prop_filter("need non-zero vector", |v| v.iter().any(|&x| x != 0.0))
}

// ── Encoding properties ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn single_token_equals_embedding(
        cb in codebook_strategy(4, 3),
        id in 0u32..4,
    ) {
        let encoded = cb.encode_ids(&[id]);
        let expected = cb.get(id).unwrap();
        for (a, b) in encoded.iter().zip(expected.iter()) {
            prop_assert!((a - b).abs() < 1e-6, "single token encoding mismatch");
        }
    }

    #[test]
    fn strict_equals_lenient_for_valid_ids(
        cb in codebook_strategy(4, 3),
    ) {
        // All valid IDs
        let ids: Vec<u32> = (0..4).collect();
        let lenient = cb.encode_ids(&ids);
        let strict = cb.encode_ids_strict(&ids).unwrap();
        for (a, b) in lenient.iter().zip(strict.iter()) {
            prop_assert!((a - b).abs() < 1e-6, "strict vs lenient mismatch: {a} vs {b}");
        }
    }
}

// ── Normalization properties ────────────────────────────────────────────────

proptest! {
    #[test]
    fn l2_normalize_produces_unit_norm(v in nonzero_vec_strategy(5)) {
        let mut v = v;
        l2_normalize_in_place(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn l2_normalize_idempotent(v in nonzero_vec_strategy(5)) {
        let mut v = v;
        l2_normalize_in_place(&mut v);
        let after_first: Vec<f32> = v.clone();
        l2_normalize_in_place(&mut v);
        for (a, b) in v.iter().zip(after_first.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-5,
                "normalize not idempotent: {a} vs {b}"
            );
        }
    }
}

// ── remove_component properties ─────────────────────────────────────────────

proptest! {
    #[test]
    fn remove_component_reduces_dot_to_zero(
        v in nonzero_vec_strategy(5),
        u in nonzero_vec_strategy(5),
    ) {
        let mut u_unit = u;
        l2_normalize_in_place(&mut u_unit);

        let mut v_proj = v;
        remove_component_in_place(&mut v_proj, &u_unit).unwrap();

        let dot: f32 = v_proj.iter().zip(u_unit.iter()).map(|(a, b)| a * b).sum();
        prop_assert!(
            dot.abs() < 1e-4,
            "dot product after removal should be ~0, got {dot}"
        );
    }
}

// ── SIF weight properties ───────────────────────────────────────────────────

proptest! {
    #[test]
    fn sif_weight_in_zero_one(p in 0.0f32..1.0, a in 0.001f32..1.0) {
        let w = sif_weight(p, a);
        prop_assert!(w >= 0.0 && w <= 1.0, "sif_weight out of [0,1]: {w}");
    }

    #[test]
    fn sif_weight_monotonically_decreasing(
        p1 in 0.0f32..0.5,
        delta in 0.001f32..0.5,
        a in 0.001f32..1.0,
    ) {
        let p2 = p1 + delta;
        let w1 = sif_weight(p1, a);
        let w2 = sif_weight(p2, a);
        prop_assert!(
            w1 >= w2,
            "sif_weight should decrease with p: w({p1})={w1} < w({p2})={w2}"
        );
    }
}
