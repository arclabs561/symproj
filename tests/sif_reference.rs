//! SIF sentence vectors (Arora, Liang & Ma 2017, Algorithm 1) against a numpy
//! reference.
//!
//! The expected values come from a float64 numpy run of Algorithm 1 on the
//! fixture below: `v_s = (1/|s|) sum_t a/(a+p_t) E_t`, then remove the
//! projection on the first right singular vector `u` of the uncentered
//! sentence matrix. symproj does not compute `u`, so the test passes numpy's.

use symproj::{remove_component_in_place, sif_weight, Codebook};

const EMBEDDINGS: [f32; 15] = [
    1.0, 0.0, 0.5, //
    0.2, 1.0, 0.0, //
    0.0, 0.3, 1.0, //
    0.7, 0.7, 0.1, //
    0.4, -0.2, 0.9, //
];
const PROBABILITIES: [f32; 5] = [0.05, 0.001, 0.02, 0.3, 0.0005];
const A: f32 = 1e-3;

const NUMPY_SENTENCE_VECTORS: [[f64; 3]; 4] = [
    [3.9869281046e-02, 1.7142857143e-01, 1.9140989729e-02],
    [3.4883720930e-02, 1.6821705426e-01, 2.2148394241e-04],
    [1.4313725490e-01, -6.6666666667e-02, 3.0490196078e-01],
    [9.2248062016e-02, 9.5819490587e-02, 1.6198781838e-01],
];
const NUMPY_FIRST_SINGULAR_VECTOR: [f64; 3] = [-0.4532678107, -0.0524953707, -0.8898272461];
const NUMPY_SIF: [[f64; 3]; 4] = [
    [0.0198788823, 0.1691133764, -0.0201029210],
    [0.0236248355, 0.1669131025, -0.0218812615],
    [-0.0076603636, -0.0841313458, 0.0088654314],
    [0.0056810124, 0.0857936973, -0.0079552430],
];

fn sentences() -> [&'static [u32]; 4] {
    [&[0, 1, 2], &[3, 3, 1], &[4, 0], &[2, 4, 3, 1]]
}

/// Arora's `(1/|s|) sum w E` from symproj's weighted mean `sum w E / sum w`.
fn arora_sentence_vector(codebook: &Codebook, ids: &[u32]) -> Vec<f32> {
    let weights: Vec<f32> = ids
        .iter()
        .map(|&id| sif_weight(PROBABILITIES[id as usize], A))
        .collect();
    let mean = codebook.encode_ids_weighted_strict(ids, &weights).unwrap();
    let scale = weights.iter().sum::<f32>() / ids.len() as f32;
    mean.iter().map(|x| x * scale).collect()
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    for (g, w) in got.iter().zip(want) {
        assert!(
            (f64::from(*g) - w).abs() < 1e-6,
            "{what}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn sif_matches_numpy_reference() {
    let codebook = Codebook::new(EMBEDDINGS.to_vec(), 3).unwrap();
    for (s, ids) in sentences().iter().enumerate() {
        let mut v = arora_sentence_vector(&codebook, ids);
        assert_close(&v, &NUMPY_SENTENCE_VECTORS[s], "sentence vector");

        let u: Vec<f32> = NUMPY_FIRST_SINGULAR_VECTOR
            .iter()
            .map(|&x| x as f32)
            .collect();
        remove_component_in_place(&mut v, &u).unwrap();
        assert_close(&v, &NUMPY_SIF[s], "after first-component removal");
    }
}

#[test]
fn weighted_mean_is_invariant_to_weight_scale() {
    // Scaling every weight cannot turn the weighted mean into Arora's
    // 1/|s| convention; the output has to be rescaled instead.
    let codebook = Codebook::new(EMBEDDINGS.to_vec(), 3).unwrap();
    let ids = [0u32, 1, 2];
    let weights = [0.2f32, 0.5, 0.3];
    let scaled: Vec<f32> = weights.iter().map(|w| w * 7.0).collect();
    let a = codebook.encode_ids_weighted_strict(&ids, &weights).unwrap();
    let b = codebook.encode_ids_weighted_strict(&ids, &scaled).unwrap();
    let b: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
    assert_close(&a, &b, "scaled weights");
}
