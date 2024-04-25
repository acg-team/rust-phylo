use anyhow::Error;

pub mod alignment;
pub mod evolutionary_models;
pub mod io;
pub mod likelihood;
pub mod phylo_info;
pub mod pip_model;
pub mod sequences;
pub mod substitution_models;
pub mod tree;

type Result<T> = std::result::Result<T, Error>;

#[allow(non_camel_case_types)]
type f64_h = ordered_float::OrderedFloat<f64>;

pub struct Rounding {
    pub round: bool,
    pub digits: usize,
}
impl Rounding {
    pub fn zero() -> Self {
        Rounding {
            round: true,
            digits: 0,
        }
    }
    pub fn four() -> Self {
        Rounding {
            round: true,
            digits: 4,
        }
    }
    pub fn none() -> Self {
        Rounding {
            round: false,
            digits: 0,
        }
    }
}

#[cfg(test)]
pub(crate) fn cmp_f64() -> impl Fn(&f64, &f64) -> std::cmp::Ordering {
    |a, b| a.partial_cmp(b).unwrap()
}
