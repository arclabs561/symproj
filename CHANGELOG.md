# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2026-07-09

### Added

- `Codebook::from_arc` for shared embedding matrix storage.
- `Codebook::encode_ids_into` and `Codebook::encode_ids_with_stats` for buffer reuse and vocabulary coverage reporting.
- `Codebook::encode_sequence_ids_flat` and `Projection::encode_sequence_flat` for row-major multi-vector output.
- `Projection::encode_with_stats` for codebook coverage reporting after tokenization.
- `remove_component_unit_in_place` for callers that already have a normalized component direction.
- Optional `simd` feature that delegates L2 normalization to `innr`.

### Changed

- `remove_component_in_place` now accepts arbitrary non-zero component directions instead of assuming unit norm.
- `Codebook` now stores shared matrix storage internally while preserving `Codebook::new(Vec<f32>, dim)`.
- Documented that weighted strict encoding divides by `sum_w`, not sentence length.
- Declared MSRV 1.89 in `Cargo.toml`.
- Updated dependency metadata to `textprep` 0.1.6 and `innr` 0.6.3.
- Updated the cross-crate example to exercise `rankops` 0.2 flat ColBERT scoring.

### Fixed

- Non-unit component directions no longer produce residual vectors that remain aligned with the removed direction.
- Lenient encoding can now distinguish empty input from all-missing input when callers use the stats API.

## [0.1.3] - 2026-06-11

### Added

- Documented the ColBERT late-interaction workflow on `encode_sequence_ids`.

### Changed

- Bumped `innr` to 0.4 and `vicinity` to 0.5.
- Raised MSRV to 1.89 for AVX-512 support via the `vicinity`/`innr` chain.

### Fixed

- Replaced the deprecated `BpeTokenizer` with `VocabTokenizer` in the example.

## [0.1.2] - 2026-04-06

### Added

- `Codebook::matrix()` accessor.
- Property-based tests for encoding and normalization, plus edge-case tests.
- Cross-crate integration examples, including `embed_and_search`.
- Documentation of `sif_weight` behavior.

### Changed

- Merged `CodebookProjection` into `Codebook`.
- Marked `Error` as `#[non_exhaustive]`.
- Raised MSRV to 1.83 for `is_multiple_of`.

### Removed

- Unused dependencies.

[0.1.4]: https://github.com/arclabs561/symproj/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/arclabs561/symproj/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/arclabs561/symproj/releases/tag/v0.1.2
