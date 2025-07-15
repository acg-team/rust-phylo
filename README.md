# Phylo

A high-performance Rust library for phylogenetic analysis and multiple sequence alignment under maximum likelihood and parsimony optimality criteria.

<!-- [![Crates.io](https://img.shields.io/crates/v/phylo.svg)](https://crates.io/crates/phylo)
[![Documentation](https://docs.rs/phylo/badge.svg)](https://docs.rs/phylo) -->
[![Licence](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/acg-team/rust-phylo#license) [![CI](https://github.com/acg-team/rust-phylo/actions/workflows/tests.yml/badge.svg)](https://github.com/acg-team/rust-phylo/actions) [![codecov](https://codecov.io/gh/acg-team/rust-phylo/branch/main/graph/badge.svg)](https://codecov.io/gh/acg-team/rust-phylo)

[Current functionality](#current-functionality) • [Getting started](#getting-started) • [Crate features](#crate-features) • [Roadmap](#roadmap) • [Contributing](#contributing) • [Related projects](#related-projects) • [Support](#support) • [Citation](#citation) • [Licence and Attributions](#licence-and-attributions)

## Current functionality

- **Maximum Likelihood Phylogenetic Analysis**: Efficient implementation of phylogenetic tree inference using SPR moves using likelihood or parsimony cost functions;
- **Multiple Sequence Alignment (MSA)**: Support for Multiple Sequence Alignment using the IndelMaP algorithm ([paper]( https://academic.oup.com/mbe/article/41/7/msae109/7688856 ), [python implementation]( https://github.com/acg-team/indelMaP ));
- **Sequence Evolution Models**: Support for various DNA (JC69, K80, TN93, HKY, GTR) and protein (WAG, HIVB, BLOSUM62) substitution models as well as the Poisson Indel Process (PIP) ([paper]( https://www.pnas.org/doi/10.1073/pnas.1220450110 )) model;
- **High Performance**: Optimised tree search with optional parallel processing capabilities.

## Getting started

<!-- Add this to your `Cargo.toml`:

```toml
[dependencies]
phylo = "0.1.0"
``` -->

**Note**: This crate is not yet published on crates.io. To use it directly from GitHub, add this to your `Cargo.toml`:

```toml
[dependencies]
phylo = { git = "https://github.com/acg-team/rust-phylo", package = "phylo" }
```

Once published on crates.io, you'll be able to use:

```toml
[dependencies]
phylo = "0.1.0"
```

**Minimum Supported Rust Version**: 1.82.0

MSRV detected using [`cargo-msrv`]( https://github.com/foresterre/cargo-msrv ).

### Example

```rust
use std::path::Path;
    
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::TopologyOptimiser;
use phylo::phylo_info::PhyloInfoBuilder;
use phylo::substitution_models::{SubstModel, SubstitutionCostBuilder, K80};

fn main() -> std::result::Result<(), anyhow::Error> {
    // Note: This example uses test data from the repository
    let info = PhyloInfoBuilder::new(Path::new("./examples/data/K80.fasta").to_path_buf()).build()?;
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SubstitutionCostBuilder::new(k80, info).build()?;
    let unopt_cost = c.cost();
    let optimiser = TopologyOptimiser::new(c);
    let result = optimiser.run()?;
    assert_eq!(unopt_cost, result.initial_cost);
    assert!(result.final_cost > result.initial_cost);
    assert!(result.iterations <= 100);
    assert_eq!(result.cost.tree().len(), 9); // The initial tree has 9 nodes, 5 leaves and 4 internal nodes, and so should the resulting tree.
    Ok(()) 
}
```

## Crate features

This crate supports several optional features:

- `par-regraft`: Enable parallel regrafting operations using Rayon;
- `par-regraft-chunk`: Enable chunked parallel regrafting;
- `par-regraft-manual`: Enable manual parallel regrafting control;
- `deterministic`: Ensure deterministic behaviour for reproducible results (required for running tests);
- `precomputed-test-results`: Speed up test runs with precomputed results (for local development).

Enable features in your Cargo.toml:

```toml
[dependencies]
phylo = { git = "https://github.com/acg-team/rust-phylo", package = "phylo", features = ["par-regraft"] }
```

<!-- 
## Documentation
Full documentation is available at docs.rs/phylo.
-->

<!-- 
## Performance benchmarks
Will add Luca's benchmarks when possible.
-->

## Roadmap

This crate is new and in active development at the moment. The basic existing functionality is mentioned above, but the following features are being currently implemented or planned:

- Simultaneous tree and alignment estimation under the PIP model ([paper]( https://www.pnas.org/doi/10.1073/pnas.1220450110 )); 
- Maximum likelihood tree search using NNI moves under the TKF92 long indel model ([paper]( https://link.springer.com/article/10.1007/BF00163848 ));
- Extension to the PIP model that includes long insertions (manuscript in preparation);
- Ancestral state reconstruction using PIP ([paper]( https://pubmed.ncbi.nlm.nih.gov/35866991/ )), TKF92 and IndelMaP ([paper]( https://academic.oup.com/mbe/article/41/7/msae109/7688856 ));
- Randomised starting trees for tree inference;
- Generalisation of the tree structure for easier use in other crates.

Other minor features/improvements are documented on the [GitHub issues page]( https://github.com/acg-team/rust-phylo/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement ).

## Contributing

This is a new library that is currently in active development. Contributions are highly welcome!

**API Stability**: As this crate is in active development, the API may change between versions until we reach 1.0. We'll follow semantic versioning and document breaking changes in release notes.

### Running Tests

To run the test suite, you'll need to enable the `deterministic` feature:

```bash
cargo test --features deterministic
```

For faster test runs during development, you can also use:

```bash
cargo test --features "deterministic,precomputed-test-results"
```

### Submitting Changes

When you submit a pull request, it will be automatically tested with GitHub Actions. In addition to running the tests, GitHub Actions runs clippy and rustfmt on each PR.

We recommend reading [Rust crate guidelines]( https://rust-lang.github.io/api-guidelines/about.html ) to make sure your contribution is up to scratch!

### Current contributors:

- Jūlija Pečerska ([GitHub]( https://github.com/junniest ), [email]( mailto:julija.pecerska@zhaw.ch ));
- Mattes Mrzik ([GitHub]( https://github.com/MattesMrzik ), [email]( mailto:mattes.mrzik@zhaw.ch ));
- Dmitrii Iartsev ([GitHub]( https://github.com/jarcev ), [email]( mailto:dmitrii.iartsev@zhaw.ch ));
- Merlin Maggi ([GitHub]( https://github.com/merlinio2000 ));
- Luca Müller ([GitHub]( https://github.com/lucasperception )).

## Related projects

- [JATI (joint alignment tree inference)]( https://github.com/acg-team/JATI )

## Support

For questions, bug reports, or feature requests, please go to [rust-phylo discussion page]( https://github.com/acg-team/rust-phylo/discussions ) and/or open an issue on [GitHub]( https://github.com/acg-team/rust-phylo/issues ).

## Citation

If you use this library in your research, please consider citing:

```bibtex
@software{phylo_rust,
  title = {Phylo: A Rust library for phylogenetic analysis},
  author = {Pečerska, Jūlija and Mrzik, Mattes and Iartsev, Dmitrii and Gil, Manuel and Anisimova, Maria},
  url = {https://github.com/acg-team/rust-phylo},
  year = {2025}
}
```

## Licence and Attributions

### Licence

This project is licensed under either of
- Apache Licence, Version 2.0 ([LICENSE-APACHE]( LICENSE-APACHE ) or [www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )), or
- MIT Licence ([LICENSE-MIT]( LICENSE-MIT ) or [opensource.org/licenses/MIT]( http://opensource.org/licenses/MIT ))

at your option.

### Benchmarking datasets

Datasets for benchmarking were taken from:
- Zhou, Xiaofan (2017). Single-gene alignments. figshare. Dataset. [Link]( https://doi.org/10.6084/m9.figshare.5477749.v1 );
- Zhou, Xiaofan (2017). Supermatrices. figshare. Dataset. [Link]( https://doi.org/10.6084/m9.figshare.5477746.v1 ).

The datasets are licensed under [CC BY 4.0]( https://creativecommons.org/licenses/by/4.0/ ).

The datasets were modified by normalising invalid/unrecognised sequence characters since the exact sequences are less relevant
for pure performance measurements.
