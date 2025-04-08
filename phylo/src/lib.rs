#![cfg_attr(coverage, feature(coverage_attribute))]

use anyhow::Error;

pub mod alignment;
pub mod alphabets;
pub mod evolutionary_models;
pub mod io;
pub mod likelihood;
pub mod optimisers;
pub mod parsimony;
pub mod phylo_info;
pub mod pip_model;
pub mod substitution_models;
pub mod tree;

// TODO: didn't find a good way to make this non-consumer public
// because benches/ live in an individual separate crate
pub mod bench_helpers;
pub(crate) mod test_macros;

type Result<T> = std::result::Result<T, Error>;

pub(crate) const MAX_BLEN: f64 = 1e5f64;
