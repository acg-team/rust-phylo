use anyhow::Error;

pub mod alignment;
pub mod io;
pub mod likelihood;
pub mod phylo_info;
pub mod sequences;
pub mod substitution_models;
pub mod tree;

type Result<T> = std::result::Result<T, Error>;

#[allow(non_camel_case_types)]
type f64_h = ordered_float::OrderedFloat<f64>;

#[cfg(test)]
pub(crate) fn cmp_f64() -> impl Fn(&f64, &f64) -> std::cmp::Ordering {
    |a, b| a.partial_cmp(b).unwrap()
}

pub fn assert_float_relative_slice_eq(actual: &[f64], expected: &[f64], epsilon: f64) {
    use approx::relative_eq;
    assert_eq!(
        actual.len(),
        expected.len(),
        "Must have the same number of entries."
    );
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            relative_eq!(act, exp, epsilon = epsilon),
            "Entries at position {} do not match, actual: {}, expected: {}",
            i,
            act,
            exp,
        );
    }
}
