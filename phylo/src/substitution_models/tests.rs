use std::fmt::Display;
use std::iter::repeat;
use std::ops::Mul;

use approx::assert_relative_eq;
use nalgebra::dvector;
use rand::Rng;

use crate::alphabets::AMINOACIDS;
use crate::evolutionary_models::EvoModel;
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, ParsimonyModel, SubstMatrix, SubstModel,
};
use crate::Rounding as R;

use super::QMatrix;

#[cfg(test)]
fn check_freq_convergence(substmat: SubstMatrix, pi: &FreqVector, epsilon: f64) {
    assert_eq!(substmat.nrows(), pi.len());
    assert_eq!(substmat.ncols(), pi.len());
    for row in substmat.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(row, pi.transpose().as_view(), epsilon = epsilon);
    }
}

#[test]
fn dna_jc69_correct() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let jc69_2 = SubstModel::<JC69>::new(&[], &[1.0, 2.0]).unwrap();
    assert_eq!(jc69, jc69_2);
    assert_relative_eq!(jc69.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(jc69.rate(b'A', b'C'), 1.0 / 3.0);
    assert_relative_eq!(jc69.rate(b'G', b'T'), 1.0 / 3.0);
    assert_relative_eq!(jc69.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    let jc69_3 = SubstModel::<JC69>::new(&[], &[4.0]).unwrap();
    assert_eq!(jc69.q(), jc69_3.q());
    assert_eq!(jc69.freqs(), jc69_3.freqs());
}

#[test]
fn dna_j69_params() {
    let jc69 = SubstModel::<JC69>::new(&[0.1, 0.4, 0.75, 1.5], &[0.1, 0.4, 0.75, 1.5]).unwrap();
    assert_relative_eq!(jc69.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(format!("{}", jc69), format!("JC69"));
}

#[test]
fn dna_k80_correct() {
    let k80 = SubstModel::<K80>::new(&[], &[]).unwrap();
    let k801 = SubstModel::<K80>::new(&[], &[2.0]).unwrap();
    let k802 = SubstModel::<K80>::new(&[], &[2.0, 1.0]).unwrap();
    let k803 = SubstModel::<K80>::new(&[], &[2.0, 1.0, 3.0, 6.0]).unwrap();
    assert_eq!(k80, k801);
    assert_eq!(k80, k802);
    assert_eq!(k80, k803);
    assert_eq!(k802, k803);
    assert_relative_eq!(k80.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(k80.rate(b'T', b'A'), 1.0 * 0.25);
    assert_relative_eq!(k80.rate(b'A', b'G'), 2.0 * 0.25);
    assert_relative_eq!(k80.freqs(), &dvector![0.25, 0.25, 0.25, 0.25]);
}

#[cfg(test)]
fn infinity_p_template<Q: QMatrix + PartialEq + Display>(freqs: &[f64], params: &[f64]) {
    let model: SubstModel<Q> = SubstModel::<Q>::new(freqs, params).unwrap();
    let p_inf = model.p(1000.0);
    assert_eq!(p_inf.nrows(), model.n());
    assert_eq!(p_inf.ncols(), model.n());
    check_freq_convergence(p_inf, model.freqs(), 1e-5);
}

#[test]
fn dna_infinity_p() {
    infinity_p_template::<JC69>(&[], &[]);
    infinity_p_template::<K80>(&[], &[]);
    infinity_p_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    infinity_p_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    infinity_p_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn protein_infinity_p() {
    infinity_p_template::<WAG>(&[], &[]);
    infinity_p_template::<HIVB>(&[], &[]);
    infinity_p_template::<BLOSUM>(&[], &[]);
}

#[test]
fn dna_k80_params() {
    let k80 = SubstModel::<K80>::new(&[0.1, 0.4, 0.75, 1.5], &[0.1, 0.4, 0.75, 1.5]).unwrap();
    assert_relative_eq!(k80.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(format!("{}", k80), format!("K80 with [kappa = {:.5}]", 0.1));
}

#[test]
fn dna_hky_incorrect() {
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[2.0]);
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[0.5]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[0.5]);
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[0.5, 1.0]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[0.5]);
}

#[test]
fn dna_hky_correct() {
    let hky = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap();
    assert_relative_eq!(hky.freqs(), &dvector![0.22, 0.26, 0.33, 0.19]);
    let hky2 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 1.0]).unwrap();
    assert_relative_eq!(hky2.freqs(), &dvector![0.22, 0.26, 0.33, 0.19]);
    assert_eq!(hky, hky2);
    let hky3 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[]).unwrap();
    let hky4 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[2.0, 1.0]).unwrap();
    assert_relative_eq!(
        hky3.q()
            .diagonal()
            .component_mul(&frequencies!(&[0.22, 0.26, 0.33, 0.19]))
            .sum(),
        -1.0
    );
    assert_eq!(hky3, hky4);
}

#[test]
fn dna_gtr_equal_rates_correct() {
    let gtr = SubstModel::<GTR>::new(
        &repeat(0.25).take(4).collect::<Vec<f64>>(),
        &repeat(1.0).take(5).collect::<Vec<f64>>(),
    )
    .unwrap();
    assert_eq!(gtr.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(gtr.q()[(0, 0)], -1.0);
    let gtr2 = SubstModel::<GTR>::new(&[], &repeat(1.0).take(5).collect::<Vec<f64>>()).unwrap();
    assert_relative_eq!(gtr.q(), gtr2.q());
    assert!(gtr.rate(b'T', b'T') < 0.0);
    assert!(gtr.rate(b'A', b'A') < 0.0);
    assert_relative_eq!(gtr.rate(b'T', b'C'), gtr.rate(b'C', b'T'));
    assert_relative_eq!(gtr.rate(b'A', b'G'), gtr.rate(b'G', b'A'));
    assert_relative_eq!(gtr.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
}

#[test]
fn dna_gtr_defaults() {
    let gtr_no_freqs = SubstModel::<GTR>::new(&[], &[2.0, 1.0, 3.0, 6.0, 0.5]).unwrap();
    assert_relative_eq!(gtr_no_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(gtr_no_freqs.params(), &[2.0, 1.0, 3.0, 6.0, 0.5]);

    let gtr_missing_params =
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.6, 0.7]).unwrap();
    assert_relative_eq!(
        gtr_missing_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(gtr_missing_params.params(), &[0.5, 0.6, 0.7, 1.0, 1.0]);

    let gtr_incorrect_freqs = SubstModel::<GTR>::new(&[0.3; 4], &[0.5; 5]).unwrap();
    assert_relative_eq!(gtr_incorrect_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(gtr_incorrect_freqs.params(), &[0.5; 5]);

    let gtr_too_many_params = SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7; 6]).unwrap();
    assert_relative_eq!(
        gtr_too_many_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(gtr_too_many_params.params(), &[0.7; 6]);
}

#[test]
fn dna_tn93_correct() {
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135])
        .unwrap();
    let expected_pi = frequencies!(&[0.22, 0.26, 0.33, 0.19]);
    let expected_q = SubstMatrix::from_column_slice(
        4,
        4,
        &[
            -1.4732124694954951,
            1.2409529074850258,
            0.002805744890196536,
            0.002805744890196536,
            1.4665807088459397,
            -1.2475846681345815,
            0.0033158803247777245,
            0.0033158803247777245,
            0.0042086173352948045,
            0.0042086173352948045,
            -0.5339064704940854,
            0.9166789418005613,
            0.002423143314260645,
            0.002423143314260645,
            0.5277848452791111,
            -0.9228005670155355,
        ],
    );
    assert_relative_eq!(tn93.q(), &expected_q);
    assert_relative_eq!(tn93.q().diagonal().component_mul(&expected_pi).sum(), -1.0);
    assert_relative_eq!(tn93.freqs(), &expected_pi);
    assert_relative_eq!(
        tn93.rate(b'T', b'C') * expected_pi[0],
        tn93.rate(b'C', b'T') * expected_pi[1],
    );
    assert_relative_eq!(
        tn93.rate(b'A', b'G') * expected_pi[2],
        tn93.rate(b'G', b'A') * expected_pi[3],
    );
    assert_relative_eq!(
        tn93.rate(b'T', b'A') * expected_pi[0],
        tn93.rate(b'A', b'T') * expected_pi[2],
    );
    assert_relative_eq!(
        tn93.rate(b'C', b'G') * expected_pi[1],
        tn93.rate(b'G', b'C') * expected_pi[3],
    );
}

#[test]
fn dna_tn93_incorrect() {
    let tn93_too_few_params =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap();
    assert_relative_eq!(
        tn93_too_few_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(tn93_too_few_params.params(), &[0.5970915, 0.2940435, 1.0]);

    let tn93_incorrect_freqs =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 1.19], &[0.5, 0.6, 0.3]).unwrap();
    assert_relative_eq!(tn93_incorrect_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(tn93_incorrect_freqs.params(), &[0.5, 0.6, 0.3]);

    let tn93_too_many_params =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.6, 0.3, 0.56]).unwrap();
    assert_relative_eq!(
        tn93_too_many_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(tn93_too_many_params.params(), &[0.5, 0.6, 0.3]);

    let tn93_no_freqs = SubstModel::<TN93>::new(&[], &[2.0, 1.0, 3.0]).unwrap();
    assert_relative_eq!(tn93_no_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(tn93_no_freqs.params(), &[2.0, 1.0, 3.0]);
}

#[test]
fn dna_normalisation() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    assert_relative_eq!(jc69.q().diagonal().component_mul(jc69.freqs()).sum(), -1.0);
    let k80 = SubstModel::<K80>::new(&[], &[3.0, 1.5]).unwrap();
    assert_relative_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    let gtr = SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7; 5]).unwrap();
    assert_relative_eq!(gtr.q().diagonal().component_mul(gtr.freqs()).sum(), -1.0);
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.59, 0.29, 0.0013]).unwrap();
    assert_relative_eq!(tn93.q().diagonal().component_mul(tn93.freqs()).sum(), -1.0);
}

#[test]
fn dna_normalised_param_change() {
    let mut k80 = SubstModel::<K80>::new(&[], &[3.0]).unwrap();
    assert_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    let k80_old = k80.clone();
    assert_eq!(k80.params(), &[3.0]);

    k80.set_param(0, 10.0);
    assert_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    assert_eq!(k80.params(), &[10.0]);
    assert_ne!(k80.params(), k80_old.params());
    assert_ne!(k80.q(), k80_old.q());
}

#[cfg(test)]
fn protein_correct_access_template<Q: QMatrix + PartialEq + Display>(epsilon: f64) {
    let model_1 = SubstModel::<Q>::new(&[], &[]).unwrap();
    let model_2 = SubstModel::<Q>::new(&[], &[]).unwrap();
    assert_relative_eq!(model_1.q(), model_2.q());
    for _ in 0..10 {
        let mut rng = rand::thread_rng();
        let query1 = AMINOACIDS[rng.gen_range(0..AMINOACIDS.len())];
        let query2 = AMINOACIDS[rng.gen_range(0..AMINOACIDS.len())];
        model_1.rate(query1, query2);
    }
    assert_relative_eq!(model_1.freqs().sum(), 1.0, epsilon = epsilon);
}

#[test]
fn protein_correct_access() {
    protein_correct_access_template::<WAG>(1e-4);
    protein_correct_access_template::<HIVB>(1e-3);
    protein_correct_access_template::<BLOSUM>(1e-3);
}

#[cfg(test)]
fn protein_gap_access_template<Q: QMatrix + PartialEq + Display>() {
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    model.rate(b'-', b'L');
}

#[test]
#[should_panic]
fn protein_incorrect_access_wag() {
    protein_gap_access_template::<WAG>();
}

#[test]
#[should_panic]
fn protein_incorrect_access_hivb() {
    protein_gap_access_template::<HIVB>();
}

#[test]
#[should_panic]
fn protein_incorrect_access_blosum() {
    protein_gap_access_template::<BLOSUM>();
}

#[cfg(test)]
fn normalised_template<Q: QMatrix + PartialEq + Display>() {
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    assert_relative_eq!(
        (model.q().diagonal().transpose().mul(model.freqs()))[(0, 0)],
        -1.0,
        epsilon = 1e-10
    );
}

#[test]
fn protein_normalised() {
    normalised_template::<WAG>();
    normalised_template::<HIVB>();
    normalised_template::<BLOSUM>();
}

#[test]
fn dna_normalised() {
    normalised_template::<JC69>();
    normalised_template::<K80>();
    normalised_template::<HKY>();
    normalised_template::<TN93>();
    normalised_template::<GTR>();
}

#[cfg(test)]
fn dna_scoring_matrices_template<Q: QMatrix + PartialEq + Display>(
    freqs: &[f64],
    params: &[f64],
    times: &[f64],
    rounding: &R,
) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let scorings = ParsimonyModel::generate_scorings(&model, times, false, rounding);
    for &time in times {
        let (_, avg_0) = ParsimonyModel::scoring_matrix(&model, time, rounding);
        let (_, avg_1) = scorings.get(&ordered_float::OrderedFloat(time)).unwrap();
        assert_relative_eq!(avg_0, avg_1);
    }
}

#[test]
fn dna_scoring_matrices() {
    dna_scoring_matrices_template::<JC69>(&[], &[], &[0.1, 0.3, 0.5, 0.7], &R::four());
    dna_scoring_matrices_template::<K80>(&[], &[], &[0.01], &R::zero());
    dna_scoring_matrices_template::<HKY>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5],
        &[0.1, 0.2, 0.3],
        &R::none(),
    );
    dna_scoring_matrices_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
        &[0.1, 0.3, 0.5, 0.7],
        &R::zero(),
    );
    dna_scoring_matrices_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0],
        &[0.2, 0.8],
        &R::four(),
    );
}

const TRUE_MATRIX: [f64; 400] = [
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

#[test]
fn protein_scoring_matrices() {
    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat, avg) = ParsimonyModel::scoring_matrix(&model, 0.1, &R::zero());

    assert_relative_eq!(mat, true_matrix_01);

    assert_relative_eq!(avg, 5.7675);
    let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.3, &R::zero());
    assert_relative_eq!(avg, 4.7475);
    let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.5, &R::zero());
    assert_relative_eq!(avg, 4.2825);
    let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.7, &R::zero());
    assert_relative_eq!(avg, 4.0075);
}

#[test]
fn generate_protein_scorings() {
    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let scorings =
        ParsimonyModel::generate_scorings(&model, &[0.1, 0.3, 0.5, 0.7], false, &R::zero());
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat_01, avg_01) = scorings.get(&ordered_float::OrderedFloat(0.1)).unwrap();

    assert_relative_eq!(*mat_01, true_matrix_01);

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
    let model = SubstModel::<K80>::new(&[], &[1.0, 2.0]).unwrap();
    let (mat_round, avg_round) = model.scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
    let model = SubstModel::<HIVB>::new(&[], &[]).unwrap();
    let (mat_round, avg_round) = model.scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
}

#[test]
fn matrix_zero_diagonals() {
    let model = SubstModel::<HIVB>::new(&[], &[]).unwrap();
    let (mat_zeros, avg_zeros) = model.scoring_matrix_corrected(0.5, true, &R::zero());
    let (mat, avg) = model.scoring_matrix_corrected(0.5, false, &R::zero());
    assert_ne!(avg_zeros, avg);
    assert!(avg_zeros < avg);
    assert_ne!(mat_zeros, mat);
    for &element in mat_zeros.diagonal().iter() {
        assert_eq!(element, 0.0);
    }
}

#[test]
fn designation() {
    let jc69_model_desc = format!("{}", SubstModel::<JC69>::new(&[], &[2.0]).unwrap());
    assert!(jc69_model_desc.contains("JC69"));
    assert!(!jc69_model_desc.contains("2.0"));

    let k80_model_desc = format!("{}", SubstModel::<K80>::new(&[], &[2.0]).unwrap());
    assert!(k80_model_desc.contains("K80"));
    assert!(k80_model_desc.contains("kappa = 2.0"));

    let hky_model_desc = format!("{}", SubstModel::<HKY>::new(&[], &[2.5]).unwrap());
    assert!(hky_model_desc.contains("HKY"));
    assert!(hky_model_desc.contains("kappa = 2.5"));

    let tn93_model_desc = format!(
        "{}",
        SubstModel::<TN93>::new(&[], &[2.5, 0.3, 0.1]).unwrap()
    );
    assert!(tn93_model_desc.contains("TN93"));
    assert!(tn93_model_desc.contains("2.5"));
    assert!(tn93_model_desc.contains("0.3"));
    assert!(tn93_model_desc.contains("0.1"));

    let gtr_model_desc = format!(
        "{}",
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[1.5, 3.0, 1.25, 0.45, 0.1]).unwrap()
    );
    assert!(gtr_model_desc.contains("GTR"));
    assert!(gtr_model_desc.contains("rtc = 1.5"));
    assert!(gtr_model_desc.contains("rta = 3.0"));
    assert!(gtr_model_desc.contains("rtg = 1.25"));
    assert!(gtr_model_desc.contains("rca = 0.45"));
    assert!(gtr_model_desc.contains("rcg = 0.1"));
    assert!(gtr_model_desc.contains("rag = 1.0"));
    assert!(gtr_model_desc.contains("0.22"));
    assert!(gtr_model_desc.contains("0.26"));
    assert!(gtr_model_desc.contains("0.33"));
    assert!(gtr_model_desc.contains("0.19"));

    let wag_model_desc = format!("{}", SubstModel::<WAG>::new(&[], &[]).unwrap());
    assert!(wag_model_desc.contains("WAG"));

    let hivb_model_desc = format!("{}", SubstModel::<HIVB>::new(&[], &[]).unwrap());
    assert!(hivb_model_desc.contains("HIVB"));

    let blosum_model_desc = format!("{}", SubstModel::<BLOSUM>::new(&[], &[]).unwrap());
    assert!(blosum_model_desc.contains("BLOSUM"));
}
