//! # symproj
//!
//! Symbolic projection and embeddings.
//!
//! Maps discrete symbols to continuous vectors using a Codebook.
//!
//! **Naming note**: this crate was previously named `proj`, but `proj` is already taken on crates.io
//! by GeoRust's PROJ bindings (geospatial). We publish this crate as `symproj`.
//!
//! ## Intuition First
//!
//! Imagine a library where every book has a call number. The call number
//! isn't just a label; it tells you where the book sits in a 3D space.
//! `symproj` is the system that maps "book names" (tokens) to "library coordinates" (vectors).
//!
//! ## Provenance (minimal citations)
//!
//! What this crate implements is the long-lived primitive:
//! \[
//! (t_1,\dots,t_n)\mapsto \mathbb{R}^d
//! \]
//! via (1) embedding lookup (a codebook) and (2) pooling (mean).
//!
//! - **Word embeddings / lookup tables**: Mikolov et al. (word2vec), 2013. [`arXiv:1301.3781`](https://arxiv.org/abs/1301.3781)
//! - **Subword tokenization**:
//!   - BPE for NMT: Sennrich et al., 2016. [`P16-1162`](https://aclanthology.org/P16-1162/)
//!   - SentencePiece / Unigram LM: Kudo, 2018. [`arXiv:1808.06226`](https://arxiv.org/abs/1808.06226)
//! - **Sentence embeddings baseline**: Arora et al. (SIF), 2017. [`ICLR OpenReview`](https://openreview.net/forum?id=SyK00v5xx)
//! - **Modern sentence embedding fine-tuning**:
//!   - SBERT: Reimers & Gurevych, 2019. [`D19-1410`](https://aclanthology.org/D19-1410/)
//!   - SimCSE: Gao et al., 2021. [`EMNLP 2021`](https://aclanthology.org/2021.emnlp-main.552/)
//! - **Retrieval context (token vectors + pooling/compression)**:
//!   - ColBERT (late interaction): Khattab & Zaharia, 2020. [`arXiv:2004.12832`](https://arxiv.org/abs/2004.12832)
//!
//! ## Nearby Rust ecosystem crates (context, not dependencies)
//!
//! - `tokenizers` (Hugging Face tokenization): <https://docs.rs/tokenizers/>
//! - `sentencepiece` (SentencePiece model loading): <https://crates.io/crates/sentencepiece>
//! - `finalfusion` / `rust2vec` (word embedding formats): <https://docs.rs/finalfusion/> / <https://docs.rs/rust2vec/>
//! - `fastembed` (embedding generation via ONNX): <https://docs.rs/fastembed/>
//! - `candle` (Rust ML runtime): <https://github.com/huggingface/candle>

use textprep::SubwordTokenizer;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Token not found in codebook: {0}")]
    TokenNotFound(u32),
    #[error("Weight length mismatch: expected {expected}, got {got}")]
    WeightLenMismatch { expected: usize, got: usize },
    #[error("dimension cannot be zero")]
    ZeroDimension,
    #[error("matrix length {len} is not a multiple of dimension {dim}")]
    InvalidMatrixShape { len: usize, dim: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// A Codebook maps token IDs to dense vectors.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Flattened embedding matrix [vocab_size * dim]
    matrix: Vec<f32>,
    /// Dimension of each vector
    dim: usize,
}

impl Codebook {
    /// Create a new Codebook from a flattened matrix and dimension.
    pub fn new(matrix: Vec<f32>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::ZeroDimension);
        }
        if !matrix.len().is_multiple_of(dim) {
            return Err(Error::InvalidMatrixShape {
                len: matrix.len(),
                dim,
            });
        }
        Ok(Self { matrix, dim })
    }

    /// Get the vector for a token ID.
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        let start = (id as usize) * self.dim;
        let end = start + self.dim;
        if end <= self.matrix.len() {
            Some(&self.matrix[start..end])
        } else {
            None
        }
    }

    /// The raw flattened embedding matrix `[vocab_size * dim]`.
    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of token IDs representable by this codebook.
    pub fn vocab_size(&self) -> usize {
        self.matrix.len() / self.dim
    }
}

impl Codebook {
    /// Encode a token-id sequence into a single vector using mean pooling.
    ///
    /// This is **lenient**: token IDs not present in the codebook are skipped.
    pub fn encode_ids(&self, ids: &[u32]) -> Vec<f32> {
        if ids.is_empty() {
            return vec![0.0; self.dim];
        }

        let embeddings: Vec<&[f32]> = ids.iter().filter_map(|&id| self.get(id)).collect();
        if embeddings.is_empty() {
            return vec![0.0; self.dim];
        }

        let mut out = vec![0.0; self.dim];
        let count = embeddings.len() as f32;
        for emb in &embeddings {
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o += e;
            }
        }
        for o in out.iter_mut() {
            *o /= count;
        }
        out
    }

    /// Encode token IDs into a single vector using mean pooling (strict).
    ///
    /// Unlike [`Self::encode_ids`], this returns an error if any token ID is not present in the
    /// codebook. This is useful when you need a “closed vocabulary” contract.
    pub fn encode_ids_strict(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }

        let mut embeddings: Vec<&[f32]> = Vec::with_capacity(ids.len());
        for &id in ids {
            let emb = self.get(id).ok_or(Error::TokenNotFound(id))?;
            embeddings.push(emb);
        }

        let mut out = vec![0.0; self.dim];
        let count = embeddings.len() as f32;
        for emb in &embeddings {
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o += e;
            }
        }
        for o in out.iter_mut() {
            *o /= count;
        }
        Ok(out)
    }

    /// Encode token IDs into a single vector using a weighted mean (strict).
    ///
    /// We compute:
    /// \[
    /// v = \frac{\sum_i w_i \, E\[t_i\]}{\sum_i w_i}
    /// \]
    /// with the convention that if \(\sum_i w_i \le 0\), we return the zero vector.
    ///
    /// Weighting is one route toward SIF-style baselines (Arora et al., 2017).
    pub fn encode_ids_weighted_strict(&self, ids: &[u32], weights: &[f32]) -> Result<Vec<f32>> {
        if ids.len() != weights.len() {
            return Err(Error::WeightLenMismatch {
                expected: ids.len(),
                got: weights.len(),
            });
        }
        if ids.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }

        let dim = self.dim;
        let mut out = vec![0.0f32; dim];
        let mut sum_w = 0.0f32;

        for (&id, &w) in ids.iter().zip(weights.iter()) {
            let emb = self.get(id).ok_or(Error::TokenNotFound(id))?;
            if w == 0.0 {
                continue;
            }
            sum_w += w;
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o += w * e;
            }
        }

        if sum_w <= 0.0 {
            return Ok(vec![0.0; dim]);
        }

        for o in out.iter_mut() {
            *o /= sum_w;
        }
        Ok(out)
    }

    /// Encode token ids into a sequence of vectors (no pooling).
    pub fn encode_sequence_ids(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(emb) = self.get(id) {
                result.push(emb.to_vec());
            }
        }
        result
    }
}

/// SIF (Smooth Inverse Frequency) weight from Arora et al. (2017):
/// \[
/// w(p) = \frac{a}{a + p}
/// \]
/// where \(p\) is token probability and \(a\) is a small smoothing constant (often \(10^{-3}\)).
///
/// Returns `0.0` when `a <= 0` or `p < 0`. Negative probabilities are not meaningful;
/// this function treats them as a no-op rather than panicking, but a `debug_assert`
/// fires in debug builds to help catch upstream bugs.
#[inline]
pub fn sif_weight(p: f32, a: f32) -> f32 {
    debug_assert!(p >= 0.0, "sif_weight: p must be non-negative, got {p}");
    if a <= 0.0 {
        return 0.0;
    }
    if p < 0.0 {
        return 0.0;
    }
    a / (a + p)
}

/// L2-normalize a vector in place.
///
/// If the input has norm 0, this is a no-op.
pub fn l2_normalize_in_place(v: &mut [f32]) {
    let mut ss = 0.0f32;
    for &x in v.iter() {
        ss += x * x;
    }
    if ss <= 0.0 {
        return;
    }
    let inv = 1.0f32 / ss.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Remove a (unit) component direction \(u\) from a vector \(v\):
/// \[
/// v \leftarrow v - u \,(u \cdot v)
/// \]
///
/// This is the “remove top PC” post-step used in SIF-style baselines (when \(u\) is the top PC).
pub fn remove_component_in_place(v: &mut [f32], u_unit: &[f32]) -> Result<()> {
    if v.len() != u_unit.len() {
        return Err(Error::DimensionMismatch {
            expected: v.len(),
            got: u_unit.len(),
        });
    }
    debug_assert!(
        (u_unit.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 0.01,
        "u_unit should be approximately unit norm"
    );
    let mut dot = 0.0f32;
    for i in 0..v.len() {
        dot += u_unit[i] * v[i];
    }
    for i in 0..v.len() {
        v[i] -= u_unit[i] * dot;
    }
    Ok(())
}

/// A Projection combines a Tokenizer and a Codebook.
pub struct Projection<T: SubwordTokenizer> {
    tokenizer: T,
    codebook: Codebook,
}

impl<T: SubwordTokenizer> Projection<T> {
    /// Create a new Projection.
    pub fn new(tokenizer: T, codebook: Codebook) -> Self {
        Self {
            tokenizer,
            codebook,
        }
    }

    /// Encode text into a single vector using mean pooling.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_ids(&tokens)
    }

    /// Encode text into a sequence of vectors (no pooling).
    pub fn encode_sequence(&self, text: &str) -> Vec<Vec<f32>> {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_sequence_ids(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use textprep::VocabTokenizer;

    #[test]
    fn test_projection_basic() {
        let mut vocab = HashMap::new();
        vocab.insert("apple".to_string(), 0);
        vocab.insert("pie".to_string(), 1);
        let tokenizer = VocabTokenizer::from_vocab(vocab);

        let matrix = vec![
            1.0, 0.0, 0.0, // apple
            0.0, 1.0, 0.0, // pie
        ];
        let codebook = Codebook::new(matrix, 3).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let vec = proj.encode("apple pie");
        // Mean pooling: ( [1,0,0] + [0,1,0] ) / 2 = [0.5, 0.5, 0]
        assert!((vec[0] - 0.5).abs() < 1e-6);
        assert!((vec[1] - 0.5).abs() < 1e-6);
        assert!((vec[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_codebook_rejects_zero_dim() {
        let err = Codebook::new(vec![1.0, 2.0, 3.0], 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dimension cannot be zero"), "got: {msg}");
    }

    #[test]
    fn test_codebook_rejects_non_multiple() {
        let err = Codebook::new(vec![1.0, 2.0, 3.0], 2).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a multiple of dimension"), "got: {msg}");
    }

    #[test]
    fn codebook_strict_errors_on_missing_token() {
        let codebook = Codebook::new(vec![1.0, 2.0], 2).unwrap(); // vocab_size=1
        let err = codebook.encode_ids_strict(&[0, 9]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Token not found"), "got: {msg}");
    }

    #[test]
    fn weighted_mean_matches_unweighted_mean_when_all_weights_equal() {
        let matrix = vec![
            1.0, 0.0, // id=0
            0.0, 1.0, // id=1
        ];
        let codebook = Codebook::new(matrix, 2).unwrap();
        let ids = [0u32, 1u32];
        let w = [1.0f32, 1.0f32];
        let v = codebook.encode_ids_weighted_strict(&ids, &w).unwrap();
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_has_unit_norm_when_nonzero() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    #[test]
    fn single_token_equals_embedding() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids(&[0]);
        assert_eq!(&v[..], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn encode_ids_all_missing_returns_zero_vector() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids(&[999]);
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn weighted_zero_weights_returns_zero_vector() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids_weighted_strict(&[0], &[0.0]).unwrap();
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_noop_on_zero_vector() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn remove_component_dimension_mismatch_errors() {
        let mut v = vec![1.0f32, 2.0];
        let u = vec![0.0f32, 1.0, 0.0];
        assert!(remove_component_in_place(&mut v, &u).is_err());
    }

    #[test]
    fn multilingual_vocab_smoke() {
        // Minimal multilingual coverage (scripts + diacritics).
        let mut vocab = HashMap::new();
        vocab.insert("東京".to_string(), 0);
        vocab.insert("Москва".to_string(), 1);
        vocab.insert("التقى".to_string(), 2);
        vocab.insert("राम".to_string(), 3);
        vocab.insert("François".to_string(), 4);
        let tokenizer = VocabTokenizer::from_vocab(vocab);

        // 5 tokens, 1-D vectors for simplicity.
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let codebook = Codebook::new(matrix, 1).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let v = proj.encode("東京 Москва التقى राम François");
        assert!((v[0] - 3.0).abs() < 1e-6, "got={:?}", v);
    }
}
