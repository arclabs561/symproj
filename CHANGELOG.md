# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.3]: https://github.com/arclabs561/symproj/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/arclabs561/symproj/releases/tag/v0.1.2
