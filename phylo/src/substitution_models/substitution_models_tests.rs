use rstest::*;

use std::collections::HashMap;
use std::iter::repeat;
use std::ops::Mul;

use approx::assert_relative_eq;
use nalgebra::dvector;

use crate::evolutionary_models::EvolutionaryModel;
use crate::substitution_models::ParsimonyModel;
use crate::substitution_models::{
    dna_models::DNASubstModel,
    protein_models::{
        ProteinSubstArray, ProteinSubstModel, BLOSUM_PI_ARR, HIVB_PI_ARR, WAG_PI_ARR,
    },
    FreqVector, SubstMatrix,
};
use crate::{assert_float_relative_slice_eq, Rounding as R};

#[cfg(test)]
fn check_pi_convergence(substmat: SubstMatrix, pi: &[f64], epsilon: f64) {
    assert_eq!(substmat.row(0).len(), pi.len());
    for row in substmat.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = epsilon);
        assert_float_relative_slice_eq(&row.iter().cloned().collect::<Vec<f64>>(), pi, epsilon);
    }
}

#[cfg(test)]
pub(crate) fn gtr_char_probs_data() -> (Vec<f64>, HashMap<u8, FreqVector>) {
    (
        [0.21, 0.30, 0.34, 0.15]
            .into_iter()
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
        HashMap::from([
            (b'T', dvector![1.0, 0.0, 0.0, 0.0]),
            (b'C', dvector![0.0, 1.0, 0.0, 0.0]),
            (b'A', dvector![0.0, 0.0, 1.0, 0.0]),
            (b'G', dvector![0.0, 0.0, 0.0, 1.0]),
            (b'X', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'N', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'Z', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'P', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'V', dvector![0.0, 0.37974684, 0.43037975, 0.18987342]),
            (b'D', dvector![0.3, 0.0, 0.48571429, 0.21428571]),
            (b'B', dvector![0.31818182, 0.45454545, 0.0, 0.22727273]),
            (b'H', dvector![0.24705882, 0.35294118, 0.4, 0.0]),
            (b'M', dvector![0.0, 0.46875, 0.53125, 0.0]),
            (b'R', dvector![0.0, 0.0, 0.69387755, 0.30612245]),
            (b'W', dvector![0.38181818, 0.0, 0.61818182, 0.0]),
            (b'S', dvector![0.0, 0.66666667, 0.0, 0.33333333]),
            (b'Y', dvector![0.41176471, 0.58823529, 0.0, 0.0]),
            (b'K', dvector![0.58333333, 0.0, 0.0, 0.41666667]),
        ]),
    )
}

#[cfg(test)]
pub(crate) fn protein_char_probs_data(pi: &[f64]) -> HashMap<u8, FreqVector> {
    HashMap::from([
        (b'A', compile_aa_probability(&['A'], pi)),
        (b'R', compile_aa_probability(&['R'], pi)),
        (b'W', compile_aa_probability(&['W'], pi)),
        (b'B', compile_aa_probability(&['D', 'N'], pi)),
        (b'Z', compile_aa_probability(&['E', 'Q'], pi)),
        (b'J', compile_aa_probability(&['I', 'L'], pi)),
        (b'X', FreqVector::from_column_slice(pi)),
    ])
}

#[cfg(test)]
fn compile_aa_probability(chars: &[char], pi: &[f64]) -> FreqVector {
    use crate::sequences::AMINOACIDS_STR;
    let mut char_probs = FreqVector::from_column_slice(&[0.0; 20]);
    if chars.len() == 1 {
        let position = AMINOACIDS_STR.find(chars[0]).unwrap();
        char_probs[position] = 1.0;
        char_probs
    } else {
        for c in chars {
            let position = AMINOACIDS_STR.find(*c).unwrap();
            char_probs[position] = pi[position];
        }
        char_probs.scale_mut(1.0 / char_probs.sum());
        char_probs
    }
}

#[test]
fn dna_jc69_correct() {
    let jc69 = DNASubstModel::new("jc69", &Vec::new(), true).unwrap();
    let jc69_2 = DNASubstModel::new("JC69", &[1.0, 2.0], true).unwrap();
    assert_eq!(jc69, jc69_2);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'A', b'A'), -1.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'A', b'C'), 1.0 / 3.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'G', b'T'), 1.0 / 3.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&jc69),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
    let jc69_3 = DNASubstModel::new("JC69", &[4.0], true).unwrap();
    assert_eq!(jc69.q, jc69_3.q);
    assert_eq!(jc69.pi, jc69_3.pi);
    let jc69_4 = DNASubstModel::new("JC69", &[4.0], false).unwrap();
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69_4, b'A', b'A'), -3.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69_4, b'A', b'C'), 1.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69_4, b'G', b'T'), 1.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&jc69_4),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn dna_k80_correct() {
    let k80 = DNASubstModel::new("k80", &Vec::new(), true).unwrap();
    let k801 = DNASubstModel::new("k80", &[2.0], true).unwrap();
    let k802 = DNASubstModel::new("k80", &[2.0, 1.0], true).unwrap();
    let k803 = DNASubstModel::new("k80", &[2.0, 1.0, 3.0, 6.0], true).unwrap();
    assert_eq!(k80, k801);
    assert_eq!(k80, k802);
    assert_eq!(k80, k803);
    assert_eq!(k802, k803);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'A', b'A'), -1.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'T', b'A'), 1.0 * 0.25);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'A', b'G'), 2.0 * 0.25);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&k80),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn dna_hky_incorrect() {
    let hky = DNASubstModel::new("hky", &[2.0, 1.0, 3.0, 6.0], false);
    assert!(hky.is_err());
    let hky = DNASubstModel::new("hky", &[2.0, 1.0, 3.0, 6.0, 0.5], false);
    assert!(hky.is_err());
    let hky = DNASubstModel::new("hky", &[0.22, 0.26, 0.33, 0.19, 0.5, 0.6, 0.7], false);
    assert!(hky.is_err());
}

#[test]
fn dna_hky_correct() {
    let hky = DNASubstModel::new("hky", &[0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&hky),
        &dvector![0.22, 0.26, 0.33, 0.19]
    );
    let hky_2 = DNASubstModel::new("hky", &[0.22, 0.26, 0.33, 0.19, 0.5], true).unwrap();
    assert_relative_eq!(
        hky_2
            .q
            .diagonal()
            .component_mul(&dvector![0.22, 0.26, 0.33, 0.19])
            .sum(),
        -1.0
    );
}

#[test]
fn dna_gtr_correct() {
    let gtr = DNASubstModel::new(
        "gtr",
        &repeat(0.25)
            .take(4)
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
        true,
    )
    .unwrap();
    assert_eq!(gtr.pi, dvector![0.25, 0.25, 0.25, 0.25]);
    assert_eq!(gtr.q[(0, 0)], -1.0);
    let gtr2 = DNASubstModel::new(
        "gtr",
        &repeat(0.25)
            .take(4)
            .chain(repeat(1.0).take(6))
            .collect::<Vec<f64>>(),
        true,
    )
    .unwrap();
    assert_relative_eq!(gtr.q, gtr2.q);
    assert!(EvolutionaryModel::get_rate(&gtr, b'T', b'T') < 0.0);
    assert!(EvolutionaryModel::get_rate(&gtr, b'A', b'A') < 0.0);
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&gtr, b'T', b'C'),
        EvolutionaryModel::get_rate(&gtr, b'C', b'T')
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&gtr, b'A', b'G'),
        EvolutionaryModel::get_rate(&gtr, b'G', b'A')
    );
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&gtr),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn dna_gtr_incorrect() {
    let gtr = DNASubstModel::new("gtr", &[2.0, 1.0, 3.0, 6.0], false);
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new("gtr", &[0.22, 0.26, 0.33, 0.19, 0.5, 0.6, 0.7], false);
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new(
        "gtr",
        &repeat(0.3)
            .take(4)
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
        false,
    );
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new(
        "gtr",
        &repeat(0.25)
            .take(4)
            .chain(repeat(0.7).take(7))
            .collect::<Vec<f64>>(),
        false,
    );
    assert!(gtr.is_err());
}

#[test]
fn dna_tn93_correct() {
    let tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    let expected_pi = dvector![0.22, 0.26, 0.33, 0.19];
    let expected_q = SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -0.15594579,
            0.15524379,
            0.0004455,
            0.0002565,
            0.13136013,
            -0.13206213,
            0.0004455,
            0.0002565,
            0.000297,
            0.000351,
            -0.056516265,
            0.055868265,
            0.000297,
            0.000351,
            0.097034355,
            -0.097682355,
        ],
    );
    assert_relative_eq!(tn93.q, expected_q);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&tn93),
        &expected_pi
    );
    assert_relative_eq!(EvolutionaryModel::get_rate(&tn93, b'T', b'T'), -0.15594579);
    assert_relative_eq!(EvolutionaryModel::get_rate(&tn93, b'T', b'C'), 0.15524379);
    assert_relative_eq!(EvolutionaryModel::get_rate(&tn93, b'C', b'T'), 0.13136013);
    assert_relative_eq!(EvolutionaryModel::get_rate(&tn93, b'G', b'T'), 0.000297);
    assert_relative_eq!(EvolutionaryModel::get_rate(&tn93, b'G', b'A'), 0.097034355);
    let tn93_2 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        true,
    )
    .unwrap();
    assert_relative_eq!(-1.0, tn93_2.q.diagonal().component_mul(&expected_pi).sum());
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&tn93_2),
        &expected_pi
    );
}

#[test]
fn dna_tn93_incorrect() {
    let tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435],
        false,
    );
    assert!(tn93.is_err());
    let tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 1.19, 0.5970915, 0.2940435, 0.00135],
        false,
    );
    assert!(tn93.is_err());
    let tn93 = DNASubstModel::new(
        "tn93",
        &[
            0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135, 0.00135,
        ],
        false,
    );
    assert!(tn93.is_err());
}

#[test]
fn dna_model_incorrect() {
    assert!(DNASubstModel::new("jc70", &Vec::new(), false).is_err());
    assert!(DNASubstModel::new("wag", &Vec::new(), false).is_err());
    assert!(DNASubstModel::new("gtr", &repeat(0.25).take(7).collect::<Vec<f64>>(), false).is_err());
    assert!(
        DNASubstModel::new("gtr", &repeat(0.25).take(11).collect::<Vec<f64>>(), false).is_err()
    );
    assert!(DNASubstModel::new("gtr", &repeat(0.4).take(10).collect::<Vec<f64>>(), false).is_err());
}

#[test]
fn dna_p_matrix() {
    let jc69 = DNASubstModel::new("jc69", &Vec::new(), false).unwrap();
    let p_inf = EvolutionaryModel::get_p(&jc69, 200000.0);
    assert_eq!(p_inf.nrows(), 4);
    assert_eq!(p_inf.ncols(), 4);
    check_pi_convergence(p_inf, jc69.pi.as_slice(), 1e-5);
}

#[test]
fn dna_normalisation() {
    let mut jc69 = DNASubstModel::new("jc69", &Vec::new(), false).unwrap();
    jc69.normalise();
    assert_eq!((jc69.q.diagonal().transpose().mul(jc69.pi))[(0, 0)], -1.0);
    let mut k80 = DNASubstModel::new("k80", &[3.0, 1.5], false).unwrap();
    k80.normalise();
    assert_eq!((k80.q.diagonal().transpose().mul(k80.pi))[(0, 0)], -1.0);
    let mut gtr = DNASubstModel::new(
        "gtr",
        &[0.22, 0.26, 0.33, 0.19]
            .into_iter()
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
        false,
    )
    .unwrap();
    gtr.normalise();
    assert_eq!((gtr.q.diagonal().transpose().mul(gtr.pi))[(0, 0)], -1.0);
    let mut tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    tn93.normalise();
    assert_relative_eq!((tn93.q.diagonal().transpose().mul(tn93.pi))[(0, 0)], -1.0);
}

#[test]
fn dna_char_probabilities() {
    let (params, char_probs) = gtr_char_probs_data();
    let mut gtr = DNASubstModel::new("gtr", &params, false).unwrap();
    gtr.normalise();
    for (&char, expected) in char_probs.iter() {
        let actual = gtr.get_char_probability(char);
        assert_relative_eq!(actual.sum(), 1.0);
        assert_relative_eq!(actual, expected, epsilon = 1e-4);
    }
}

#[rstest]
#[case::wag("wag", &WAG_PI_ARR, 1e-8)]
#[case::blosum("blosum", &BLOSUM_PI_ARR, 1e-5)]
#[case::hivb("hivb", &HIVB_PI_ARR, 1e-8)]
fn protein_char_probabilities(#[case] input: &str, #[case] pi_array: &[f64], #[case] epsilon: f64) {
    let mut model = ProteinSubstModel::new(input, &[], false).unwrap();
    model.normalise();
    let expected = protein_char_probs_data(pi_array);
    for (char, expected_probs) in expected.into_iter() {
        let actual = model.get_char_probability(char);
        assert_relative_eq!(actual.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(actual, expected_probs, epsilon = epsilon);
    }
}

#[rstest]
#[case::wag("wag")]
#[case::blosum("blosum")]
#[case::hivb("hivb")]
fn protein_weird_char_probabilities(#[case] input: &str) {
    let mut model = ProteinSubstModel::new(input, &[], false).unwrap();
    model.normalise();
    assert_eq!(
        EvolutionaryModel::get_char_probability(&model, b'.'),
        EvolutionaryModel::get_char_probability(&model, b'X')
    );
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn dna_weird_char_probabilities(#[case] input: &str, #[case] params: &[f64]) {
    let model = DNASubstModel::new(input, params, true).unwrap();
    assert_eq!(
        EvolutionaryModel::get_char_probability(&model, b'.'),
        EvolutionaryModel::get_char_probability(&model, b'X')
    );
}

#[test]
fn protein_model_correct() {
    let wag = ProteinSubstModel::new("WAG", &[], false).unwrap();
    let wag2 = ProteinSubstModel::new("wag", &[], false).unwrap();
    assert_eq!(wag, wag2);
    EvolutionaryModel::get_rate(&wag, b'A', b'L');
    EvolutionaryModel::get_rate(&wag, b'H', b'K');
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&wag).sum(),
        1.0,
        epsilon = 1e-4
    );
    let blos = ProteinSubstModel::new("Blosum", &[], false).unwrap();
    let blos2 = ProteinSubstModel::new("bLoSuM", &[], false).unwrap();
    assert_eq!(blos, blos2);
    EvolutionaryModel::get_rate(&blos, b'R', b'N');
    EvolutionaryModel::get_rate(&blos, b'M', b'K');
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&blos).sum(),
        1.0,
        epsilon = 1e-3
    );
    let hivb = ProteinSubstModel::new("hivB", &[], false).unwrap();
    let hivb2 = ProteinSubstModel::new("HIVb", &[], false).unwrap();
    assert_eq!(hivb, hivb2);
    EvolutionaryModel::get_rate(&hivb, b'L', b'P');
    EvolutionaryModel::get_rate(&hivb, b'C', b'Q');
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&hivb).sum(),
        1.0,
        epsilon = 1e-3
    );
}

#[test]
#[should_panic]
fn protein_model_incorrect_access() {
    let wag = ProteinSubstModel::new("WAG", &[], false).unwrap();
    EvolutionaryModel::get_rate(&wag, b'H', b'J');
    EvolutionaryModel::get_rate(&wag, b'-', b'L');
}

#[test]
#[should_panic]
fn protein_model_gap() {
    let wag = ProteinSubstModel::new("WAG", &[], false).unwrap();
    EvolutionaryModel::get_rate(&wag, b'-', b'L');
}

#[test]
fn protein_model_incorrect() {
    assert!(ProteinSubstModel::new("jc69", &[], false).is_err());
    assert!(ProteinSubstModel::new("waq", &[], false).is_err());
    assert!(ProteinSubstModel::new("HIV", &[], false).is_err());
}

#[rstest]
#[case::wag("wag", 1e-2)]
#[case::blosum("blosum", 1e-3)]
// FIXME: This test fails for HIVB
// #[case::hivb("hivb", 1e-3)]
fn protein_p_matrix(#[case] input: &str, #[case] epsilon: f64) {
    let model = ProteinSubstModel::new(input, &[], false).unwrap();
    let p_inf = EvolutionaryModel::get_p(&model, 1000000.0);
    assert_eq!(p_inf.nrows(), 20);
    assert_eq!(p_inf.ncols(), 20);
    check_pi_convergence(p_inf, model.pi.as_slice(), epsilon);
}

#[rstest]
#[case::wag("wag", 1e-10)]
#[case::blosum("blosum", 1e-10)]
#[case::hivb("hivb", 1e-10)]
fn protein_normalisation(#[case] input: &str, #[case] epsilon: f64) {
    let mut model = ProteinSubstModel::new(input, &[], false).unwrap();
    model.normalise();
    assert_relative_eq!(
        (model.q.diagonal().transpose().mul(model.pi))[(0, 0)],
        -1.0,
        epsilon = epsilon
    );
}

const TRUE_MATRIX: ProteinSubstArray = [
    0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0, 4.0, 9.0, 7.0,
    4.0, 5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0, 6.0, 6.0,
    7.0, 6.0, 5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0, 7.0, 4.0, 4.0,
    9.0, 6.0, 6.0, 5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0, 8.0, 8.0, 6.0, 5.0,
    6.0, 9.0, 7.0, 7.0, 5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0, 7.0,
    5.0, 6.0, 7.0, 6.0, 5.0, 5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0, 5.0, 7.0, 5.0, 4.0, 6.0, 8.0,
    5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 6.0, 6.0, 3.0, 9.0, 4.0, 0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0,
    8.0, 6.0, 5.0, 5.0, 8.0, 7.0, 5.0, 4.0, 6.0, 5.0, 5.0, 7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0,
    8.0, 8.0, 7.0, 5.0, 6.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0,
    5.0, 7.0, 6.0, 6.0, 5.0, 6.0, 8.0, 4.0, 7.0, 6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0,
    4.0, 6.0, 5.0, 5.0, 8.0, 6.0, 5.0, 8.0, 6.0, 3.0, 6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0,
    4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0,
    6.0, 6.0, 6.0, 0.0, 6.0, 8.0, 6.0, 5.0, 5.0, 8.0, 8.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0,
    6.0, 7.0, 4.0, 3.0, 5.0, 0.0, 5.0, 7.0, 6.0, 5.0, 7.0, 6.0, 4.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0,
    8.0, 8.0, 6.0, 5.0, 4.0, 7.0, 6.0, 0.0, 7.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 6.0, 7.0, 6.0, 8.0,
    6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 5.0, 4.0, 5.0,
    6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0, 4.0, 7.0, 6.0, 6.0, 4.0, 6.0, 5.0,
    6.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0, 0.0, 9.0, 7.0, 5.0, 7.0, 5.0,
    8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0, 7.0, 0.0, 5.0, 6.0, 6.0,
    6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0, 6.0, 6.0, 0.0, 6.0,
    4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 8.0, 7.0,
    0.0,
];

#[rstest]
#[case::jc69("jc69", &[], &[0.1, 0.3, 0.5, 0.7], &R::four())]
#[case::k80("k80", &[], &[0.01], &R::zero())]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5], &[0.1, 0.2, 0.3], &R::none())]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135], &[0.1, 0.3, 0.5, 0.7], &R::zero())]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0], &[0.2, 0.8], &R::four())]
fn dna_scoring_matrices(
    #[case] input: &str,
    #[case] params: &[f64],
    #[case] times: &[f64],
    #[case] rounding: &R,
) {
    let mut model = DNASubstModel::new(input, params, false).unwrap();
    model.normalise();
    let scorings = ParsimonyModel::generate_scorings(&model, times, false, rounding);
    for &time in times {
        let (_, avg_0) = ParsimonyModel::get_scoring_matrix(&model, time, rounding);
        let (_, avg_1) = scorings.get(&ordered_float::OrderedFloat(time)).unwrap();
        assert_relative_eq!(avg_0, avg_1);
    }
}

#[test]
fn protein_scoring_matrices() {
    let mut model = ProteinSubstModel::new("wag", &[], false).unwrap();
    model.normalise();
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.1, &R::zero());
    for (row, true_row) in mat.row_iter().zip(true_matrix_01.row_iter()) {
        assert_eq!(row, true_row);
    }
    assert_relative_eq!(avg, 5.7675);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.3, &R::zero());
    assert_relative_eq!(avg, 4.7475);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.5, &R::zero());
    assert_relative_eq!(avg, 4.2825);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.7, &R::zero());
    assert_relative_eq!(avg, 4.0075);
}

#[test]
fn generate_protein_scorings() {
    let mut model = ProteinSubstModel::new("wag", &[], false).unwrap();
    model.normalise();
    let scorings =
        ParsimonyModel::generate_scorings(&model, &[0.1, 0.3, 0.5, 0.7], false, &R::zero());
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat_01, avg_01) = scorings.get(&ordered_float::OrderedFloat(0.1)).unwrap();
    for (row, true_row) in mat_01.row_iter().zip(true_matrix_01.row_iter()) {
        assert_eq!(row, true_row);
    }
    assert_relative_eq!(*avg_01, 5.7675);
    let (_, avg_03) = scorings.get(&ordered_float::OrderedFloat(0.3)).unwrap();
    assert_relative_eq!(*avg_03, 4.7475);
    let (_, avg_05) = scorings.get(&ordered_float::OrderedFloat(0.5)).unwrap();
    assert_relative_eq!(*avg_05, 4.2825);
    let (_, avg_07) = scorings.get(&ordered_float::OrderedFloat(0.7)).unwrap();
    assert_relative_eq!(*avg_07, 4.0075);
}

#[test]
fn matrix_entry_rounding() {
    let model = DNASubstModel::new("K80", &[1.0, 2.0], false).unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
    let model = ProteinSubstModel::new("HIVB", &[], false).unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
}

#[test]
fn matrix_zero_diagonals() {
    let model = ProteinSubstModel::new("HIVB", &[], false).unwrap();
    let (mat_zeros, avg_zeros) = model.get_scoring_matrix_corrected(0.5, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.5, false, &R::zero());
    assert_ne!(avg_zeros, avg);
    assert!(avg_zeros < avg);
    assert_ne!(mat_zeros, mat);
    for &element in mat_zeros.diagonal().iter() {
        assert_eq!(element, 0.0);
    }
}
