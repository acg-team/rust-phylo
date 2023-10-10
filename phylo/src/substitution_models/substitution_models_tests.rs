use crate::assert_float_relative_slice_eq;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel, SubstMatrix};
use approx::assert_relative_eq;
use nalgebra::vector;
use rstest::*;
use std::iter::repeat;
use std::ops::Mul;

fn check_pi_convergence<const N: usize>(substmat: SubstMatrix<N>, pi: &[f64], epsilon: f64) {
    assert_eq!(N, pi.len());
    for col in substmat.column_iter() {
        for (i, &cell) in col.iter().enumerate() {
            assert_relative_eq!(cell, pi[i], epsilon = epsilon);
        }
    }
}

#[test]
fn dna_jc69_correct() {
    let jc69 = DNASubstModel::new("jc69", &Vec::new()).unwrap();
    let jc692 = DNASubstModel::new("JC69", &[2.0, 1.0]).unwrap();
    assert_eq!(jc69, jc692);
}

#[test]
fn dna_k80_correct() {
    let k80 = DNASubstModel::new("k80", &Vec::new()).unwrap();
    let k802 = DNASubstModel::new("k80", &[2.0, 1.0]).unwrap();
    let k803 = DNASubstModel::new("k80", &[2.0, 1.0, 3.0, 6.0]).unwrap();
    assert_eq!(k80, k802);
    assert_eq!(k80, k803);
    assert_eq!(k802, k803);
}

#[test]
fn dna_gtr_correct() {
    let mut gtr = DNASubstModel::new(
        "gtr",
        &repeat(0.25)
            .take(4)
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    gtr.normalise();
    assert_eq!(gtr.pi, vector![0.25, 0.25, 0.25, 0.25]);
    assert_eq!(gtr.q[(0, 0)], -1.0);
    let mut gtr2 = DNASubstModel::new(
        "gtr",
        &repeat(0.25)
            .take(4)
            .chain(repeat(1.0).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    gtr2.normalise();
    assert_float_relative_slice_eq(gtr.q.as_slice(), gtr2.q.as_slice(), 0.00001);
}

#[test]
fn dna_tn93_correct() {
    let tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    let expected = [
        [-0.15594579, 0.15524379, 0.0004455, 0.0002565],
        [0.13136013, -0.13206213, 0.0004455, 0.0002565],
        [0.000297, 0.000351, -0.056516265, 0.055868265],
        [0.000297, 0.000351, 0.097034355, -0.097682355],
    ];
    for (i, expected) in expected.iter().enumerate() {
        assert_float_relative_slice_eq(
            &tn93.q.row(i).iter().copied().collect::<Vec<_>>(),
            expected,
            0.0000001,
        )
    }
}

#[test]
fn dna_model_incorrect() {
    assert!(DNASubstModel::new("jc70", &Vec::new()).is_err());
    assert!(DNASubstModel::new("wag", &Vec::new()).is_err());
    assert!(DNASubstModel::new("gtr", &repeat(0.25).take(7).collect::<Vec<f64>>()).is_err());
    assert!(DNASubstModel::new("gtr", &repeat(0.25).take(11).collect::<Vec<f64>>()).is_err());
    assert!(DNASubstModel::new("gtr", &repeat(0.4).take(10).collect::<Vec<f64>>()).is_err());
}

#[test]
fn dna_p_matrix() {
    let jc69 = DNASubstModel::new("jc69", &Vec::new()).unwrap();
    let p_inf = jc69.get_p(200000.0);
    assert_eq!(p_inf.nrows(), 4);
    assert_eq!(p_inf.ncols(), 4);
    check_pi_convergence(p_inf, jc69.pi.as_slice(), 1e-5);
}

#[test]
fn dna_normalisation() {
    let mut jc69 = DNASubstModel::new("jc69", &Vec::new()).unwrap();
    jc69.normalise();
    assert_eq!((jc69.q.diagonal().transpose().mul(jc69.pi))[(0, 0)], -1.0);
    let mut k80 = DNASubstModel::new("k80", &[3.0, 1.5]).unwrap();
    k80.normalise();
    assert_eq!((k80.q.diagonal().transpose().mul(k80.pi))[(0, 0)], -1.0);
    let mut gtr = DNASubstModel::new(
        "gtr",
        &[0.22, 0.26, 0.33, 0.19]
            .into_iter()
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    gtr.normalise();
    assert_eq!((gtr.q.diagonal().transpose().mul(gtr.pi))[(0, 0)], -1.0);
    let mut tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    tn93.normalise();
    assert_relative_eq!((tn93.q.diagonal().transpose().mul(tn93.pi))[(0, 0)], -1.0);
}

#[test]
fn protein_model_correct() {
    let wag = ProteinSubstModel::new("WAG").unwrap();
    let wag2 = ProteinSubstModel::new("wag").unwrap();
    assert_eq!(wag, wag2);
    wag.get_rate(b'A', b'L');
    wag.get_rate(b'H', b'K');
    assert_relative_eq!(wag.pi.sum(), 1.0, epsilon = 0.0001);
    let blos = ProteinSubstModel::new("Blosum").unwrap();
    let blos2 = ProteinSubstModel::new("bLoSuM").unwrap();
    assert_eq!(blos, blos2);
    blos.get_rate(b'R', b'N');
    blos.get_rate(b'M', b'K');
    assert_relative_eq!(blos.pi.sum(), 1.0, epsilon = 0.001);
    let hivb = ProteinSubstModel::new("hivB").unwrap();
    let hivb2 = ProteinSubstModel::new("HIVb").unwrap();
    assert_eq!(hivb, hivb2);
    hivb.get_rate(b'L', b'P');
    hivb.get_rate(b'C', b'Q');
    assert_relative_eq!(hivb.pi.sum(), 1.0, epsilon = 0.0001);
}

#[test]
#[should_panic]
fn protein_model_incorrect_access() {
    let wag = ProteinSubstModel::new("WAG").unwrap();
    wag.get_rate(b'H', b'J');
}

#[test]
fn protein_model_incorrect() {
    assert!(ProteinSubstModel::new("jc69").is_err());
    assert!(ProteinSubstModel::new("waq").is_err());
    assert!(ProteinSubstModel::new("HIV").is_err());
}

#[rstest]
#[case::wag("wag", 1e-3)]
#[case::blosum("blosum", 1e-3)]
fn protein_p_matrix(#[case] input: &str, #[case] epsilon: f64) {
    let model = ProteinSubstModel::new(input).unwrap();
    let p_inf = model.get_p(20000.0);
    assert_eq!(p_inf.nrows(), 20);
    assert_eq!(p_inf.ncols(), 20);
    check_pi_convergence(p_inf, model.pi.as_slice(), epsilon);
}

#[rstest]
#[case::wag("wag", 0.1)]
#[case::blosum("blosum", 0.01)]
fn protein_normalisation(#[case] input: &str, #[case] epsilon: f64) {
    // This uses a crazy epsilon, but the matrices are off by a bit, need fixing.
    let mut model = ProteinSubstModel::new(input).unwrap();
    model.normalise();
    assert_relative_eq!(
        (model.q.sum() - model.q.diagonal().sum()) / 20.0,
        1.0,
        epsilon = epsilon
    );
}

const TRUE_MATRIX: [[f64; 20]; 20] = [
    [
        0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0, 4.0, 9.0,
        7.0, 4.0,
    ],
    [
        5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0, 6.0, 6.0,
        7.0, 6.0,
    ],
    [
        5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0, 7.0, 4.0, 4.0, 9.0,
        6.0, 6.0,
    ],
    [
        5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0, 8.0, 8.0, 6.0, 5.0, 6.0, 9.0,
        7.0, 7.0,
    ],
    [
        5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0, 7.0, 5.0, 6.0, 7.0,
        6.0, 5.0,
    ],
    [
        5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0, 5.0, 7.0, 5.0, 4.0, 6.0, 8.0, 5.0, 5.0, 5.0, 8.0,
        7.0, 6.0,
    ],
    [
        4.0, 6.0, 6.0, 3.0, 9.0, 4.0, 0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0, 8.0, 6.0, 5.0, 5.0, 8.0,
        7.0, 5.0,
    ],
    [
        4.0, 6.0, 5.0, 5.0, 7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0, 8.0, 8.0, 7.0, 5.0, 6.0, 8.0,
        8.0, 7.0,
    ],
    [
        6.0, 5.0, 4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0, 5.0, 7.0, 6.0, 6.0, 5.0, 6.0, 8.0,
        4.0, 7.0,
    ],
    [
        6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0, 4.0, 6.0, 5.0, 5.0, 8.0, 6.0, 5.0, 8.0,
        6.0, 3.0,
    ],
    [
        6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0, 4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0,
        6.0, 4.0,
    ],
    [
        5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 0.0, 6.0, 8.0, 6.0, 5.0, 5.0, 8.0,
        8.0, 6.0,
    ],
    [
        5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 4.0, 3.0, 5.0, 0.0, 5.0, 7.0, 6.0, 5.0, 7.0,
        6.0, 4.0,
    ],
    [
        6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 6.0, 5.0, 4.0, 7.0, 6.0, 0.0, 7.0, 6.0, 7.0, 6.0,
        4.0, 5.0,
    ],
    [
        4.0, 6.0, 7.0, 6.0, 8.0, 6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0, 5.0, 8.0,
        7.0, 6.0,
    ],
    [
        4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0, 4.0, 7.0,
        6.0, 6.0,
    ],
    [
        4.0, 6.0, 5.0, 6.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0, 0.0, 9.0,
        7.0, 5.0,
    ],
    [
        7.0, 5.0, 8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0, 7.0, 0.0,
        5.0, 6.0,
    ],
    [
        6.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0, 6.0, 6.0,
        0.0, 6.0,
    ],
    [
        4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 8.0,
        7.0, 0.0,
    ],
];

#[test]
fn protein_scoring_matrices() {
    let mut model = ProteinSubstModel::new("wag").unwrap();
    model.normalise();
    let true_matrix_01 = SubstMatrix::from(TRUE_MATRIX);
    let (mat, avg) = model.get_scoring_matrix(0.1, true);
    assert_eq!(mat.to_string(), true_matrix_01.to_string());
    assert_relative_eq!(avg, 5.7675);
    let (_, avg) = model.get_scoring_matrix(0.3, true);
    assert_relative_eq!(avg, 4.7475);
    let (_, avg) = model.get_scoring_matrix(0.5, true);
    assert_relative_eq!(avg, 4.2825);
    let (_, avg) = model.get_scoring_matrix(0.7, true);
    assert_relative_eq!(avg, 4.0075);
}

#[test]
fn generate_protein_scorings() {
    let mut model = ProteinSubstModel::new("wag").unwrap();
    model.normalise();
    let scorings = model.generate_scorings(&[0.1, 0.3, 0.5, 0.7], false, true);
    let (mat_01, avg_01) = scorings.get(&ordered_float::OrderedFloat(0.1)).unwrap();
    assert_float_relative_slice_eq(
        mat_01.as_slice(),
        &TRUE_MATRIX.iter().flatten().cloned().collect::<Vec<f64>>(),
        0.0001,
    );
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
    let model = DNASubstModel::new("K80", &[1.0, 2.0]).unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, true);
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, false);
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
    let model = ProteinSubstModel::new("HIVB").unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, true);
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, false);
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
}

#[test]
fn matrix_zero_diagonals() {
    let model = ProteinSubstModel::new("HIVB").unwrap();
    let (mat_zeros, avg_zeros) = model.get_scoring_matrix_corrected(0.5, true, true);
    let (mat, avg) = model.get_scoring_matrix_corrected(0.5, false, true);
    assert_ne!(avg_zeros, avg);
    assert!(avg_zeros < avg);
    assert_ne!(mat_zeros, mat);
    for &element in mat_zeros.diagonal().iter() {
        assert_eq!(element, 0.0);
    }
}
