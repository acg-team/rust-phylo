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

pub(crate) mod test_macros;

type Result<T> = std::result::Result<T, Error>;

#[allow(non_camel_case_types)]
type ord_f64 = ordered_float::OrderedFloat<f64>;

pub(crate) fn cmp_f64() -> impl Fn(&f64, &f64) -> std::cmp::Ordering {
    |a, b| a.partial_cmp(b).unwrap()
}
