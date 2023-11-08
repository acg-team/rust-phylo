use approx::assert_relative_eq;
use nalgebra::dvector;

use crate::evolutionary_models::EvolutionaryModel;
use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::dna_models::DNASubstModel;
use crate::substitution_models::substitution_models_tests::gtr_char_probs;
use crate::substitution_models::SubstitutionModel;
use crate::{pip_model::PIPModel, substitution_models::SubstMatrix};

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(pip_jc69.pi, dvector![0.25, 0.25, 0.25, 0.25, 0.0]);
    let jc96 = DNASubstModel::new("jc69", &[]).unwrap();
    compare_pip_subst_rates(&pip_jc69, &jc96, pip_jc69.mu);
}

#[test]
fn pip_dna_tn93_correct() {
    let pip_tn93 = PIPModel::<4>::new(
        "tn93",
        &[
            0.2, 0.5, 0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135,
        ],
    )
    .unwrap();
    let expected = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            -0.15594579 - 0.5,
            0.15524379,
            0.0004455,
            0.0002565,
            0.5,
            0.13136013,
            -0.13206213 - 0.5,
            0.0004455,
            0.0002565,
            0.5,
            0.000297,
            0.000351,
            -0.056516265 - 0.5,
            0.055868265,
            0.5,
            0.000297,
            0.000351,
            0.097034355,
            -0.097682355 - 0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    );
    assert_relative_eq!(pip_tn93.q, expected);
    assert_relative_eq!(pip_tn93.pi, dvector![0.22, 0.26, 0.33, 0.19, 0.0]);
}

#[test]
fn pip_dna_too_few_params() {
    let result = PIPModel::<4>::new("jc69", &[0.2]);
    assert!(result.is_err());
    let result = PIPModel::<4>::new("tn93", &[0.2, 0.5, 0.22, 0.26, 0.33, 0.19]);
    assert!(result.is_err());
    let result = PIPModel::<4>::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    );
    assert!(result.is_err());
}

#[test]
fn pip_dna_incorrect_dna_model() {
    let result = PIPModel::<4>::new("jc68", &[0.2, 0.5]);
    assert!(result.is_err());
    let result = PIPModel::<4>::new(
        "blosum",
        &[
            0.2, 0.5, 0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135,
        ],
    );
    assert!(result.is_err());
}

#[test]
fn pip_char_frequencies() {
    let (params, char_probs) = gtr_char_probs();
    let pip_gtr = PIPModel::<4>::new("gtr", &[vec![0.2, 0.4], params].concat()).unwrap();
    for (&char, expected_gtr) in char_probs.iter() {
        let expected_pip = expected_gtr.clone().insert_row(4, 0.0);
        let actual = pip_gtr.get_char_probability(char);
        assert_eq!(actual.len(), 5);
        assert_relative_eq!(actual.sum(), 1.0);
        assert_relative_eq!(actual, expected_pip, epsilon = 1e-4);
    }
    let actual = pip_gtr.get_char_probability(b'-');
    assert_eq!(actual.len(), 5);
    assert_relative_eq!(actual.sum(), 1.0);
    assert_relative_eq!(actual, dvector![0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn pip_rates() {
    let pip_params = vec![0.2, 0.5];
    let tn93_params = vec![0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135];
    let pip_tn93 = PIPModel::<4>::new("tn93", &[pip_params, tn93_params.clone()].concat()).unwrap();
    let tn93 = DNASubstModel::new("tn93", &tn93_params).unwrap();
    compare_pip_subst_rates(&pip_tn93, &tn93, pip_tn93.mu);
}

#[cfg(test)]
fn compare_pip_subst_rates(pip_model: &PIPModel<4>, subst_model: &SubstitutionModel<4>, mu: f64) {
    for char in charify(NUCLEOTIDES_STR) {
        assert!(pip_model.get_rate(char, char) < 0.0);
        assert_relative_eq!(
            pip_model
                .q
                .row(pip_model.index[char as usize] as usize)
                .sum(),
            0.0
        );
        for other_char in charify(NUCLEOTIDES_STR) {
            if char == other_char {
                continue;
            }
            assert_relative_eq!(
                pip_model.get_rate(char, other_char),
                subst_model.get_rate(char, other_char)
            );
        }
        assert_relative_eq!(pip_model.get_rate(char, b'-'), mu);
        assert_relative_eq!(pip_model.get_rate(b'-', char), 0.0);
    }
}

#[test]
fn pip_p_matrix_inf() {
    let pip_params = vec![0.2, 0.5];
    let pip_jc69 = PIPModel::<4>::new("jc69", &pip_params).unwrap();
    let time = 10000000.0;
    let p = pip_jc69.get_p(time);
    let expected = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(p, expected, epsilon = 1e-2);
}

#[test]
fn pip_p_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let pip_params = vec![0.5, 0.25];
    let hky_params = vec![0.22, 0.26, 0.33, 0.19, 0.5];
    let pip_hky = PIPModel::<4>::new("hky", &[pip_params, hky_params.clone()].concat()).unwrap();
    let expected_q = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            -0.9, 0.13, 0.33, 0.19, 0.25, 0.11, -0.88, 0.33, 0.19, 0.25, 0.22, 0.26, -0.825, 0.095,
            0.25, 0.22, 0.26, 0.165, -0.895, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    assert_eq!(pip_hky.q, expected_q);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.225, 0.109, 0.173, 0.0996, 0.393, 0.0922, 0.242, 0.173, 0.0996, 0.393, 0.115, 0.136,
            0.276, 0.0792, 0.393, 0.115, 0.136, 0.138, 0.217, 0.393, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.get_p(2.0), expected_p, epsilon = 1e-3);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.437, 0.0859, 0.162, 0.0935, 0.221, 0.0727, 0.45, 0.162, 0.0935, 0.221, 0.108, 0.128,
            0.48, 0.0625, 0.221, 0.108, 0.128, 0.108, 0.434, 0.221, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.get_p(1.0), expected_p, epsilon = 1e-3);
}
