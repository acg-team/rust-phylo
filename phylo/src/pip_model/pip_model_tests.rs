use rstest::*;

use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dvector, DMatrix, DVector};
use nalgebra::{Const, DimMin};

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{setup_phylogenetic_info, PhyloInfo};
use crate::pip_model::{PIPLikelihoodCost, PIPModel, PIPModelInfo};
use crate::sequences::{charify, AMINOACIDS_STR, NUCLEOTIDES_STR};
use crate::substitution_models::{
    dna_models::DNASubstModel,
    protein_models::{ProteinSubstModel, BLOSUM_PI_ARR, HIVB_PI_ARR, WAG_PI_ARR},
    substitution_models_tests::{gtr_char_probs_data, protein_char_probs_data},
    FreqVector, SubstMatrix, SubstitutionModel,
};
use crate::tree::tree_parser;

#[cfg(test)]
fn compare_pip_subst_rates<const N: usize>(
    chars: &str,
    pip_model: &PIPModel<N>,
    subst_model: &SubstitutionModel<N>,
) where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    for char in charify(chars) {
        assert!(pip_model.get_rate(char, char) < 0.0);
        assert_relative_eq!(
            pip_model
                .q
                .row(pip_model.index[char as usize] as usize)
                .sum(),
            0.0,
            epsilon = 1e-10
        );
        for other_char in charify(chars) {
            if char == other_char {
                continue;
            }
            assert_relative_eq!(
                pip_model.get_rate(char, other_char),
                subst_model.get_rate(char, other_char)
            );
        }
        assert_relative_eq!(pip_model.get_rate(char, b'-'), pip_model.mu);
        assert_relative_eq!(pip_model.get_rate(b'-', char), 0.0);
    }
}

#[rstest]
#[case::wag("wag", &[0.1, 0.4], &WAG_PI_ARR)]
#[case::blosum("blosum", &[0.8, 0.25], &BLOSUM_PI_ARR)]
#[case::hivb("hivb", &[1.1, 12.4], &HIVB_PI_ARR)]
fn protein_pip_correct(
    #[case] model_name: &str,
    #[case] model_params: &[f64],
    #[case] pi_array: &[f64],
) {
    let pip_model = PIPModel::<20>::new(model_name, model_params, false).unwrap();
    assert_eq!(pip_model.lambda, model_params[0]);
    assert_eq!(pip_model.mu, model_params[1]);
    let frequencies = FreqVector::from_column_slice(pi_array).insert_row(20, 0.0);
    assert_eq!(pip_model.pi, frequencies);
    let subst_model = ProteinSubstModel::new(model_name, &[], false).unwrap();
    compare_pip_subst_rates(AMINOACIDS_STR, &pip_model, &subst_model);
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4], false).unwrap();
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(pip_jc69.pi, dvector![0.25, 0.25, 0.25, 0.25, 0.0]);
    let jc96 = DNASubstModel::new("jc69", &[], false).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES_STR, &pip_jc69, &jc96);
}

#[test]
fn pip_dna_tn93_correct() {
    let pip_tn93 = PIPModel::<4>::new(
        "tn93",
        &[
            0.2, 0.5, 0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135,
        ],
        false,
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

#[rstest]
#[case::jc69("jc69", &[0.2])]
#[case::k80("k80", &[0.2])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93_too_few_for_subst("tn93", &[0.2, 0.5, 0.22, 0.26, 0.33, 0.19])]
#[case::tn93_too_few_for_pip("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_dna_too_few_params(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let result = PIPModel::<4>::new(model_name, model_params, false);
    assert!(result.is_err());
}

#[rstest]
#[case::wag_no_params("wag", &[])]
#[case::blosum("k80", &[0.2])]
#[case::hivb("hivb", &[0.22])]
#[case::wag_one_param("wag", &[0.1])]
fn pip_protein_too_few_params(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let result = PIPModel::<20>::new(model_name, model_params, false);
    assert!(result.is_err());
}

#[test]
fn pip_dna_incorrect_dna_model() {
    let result = PIPModel::<4>::new("jc68", &[0.2, 0.5], false);
    assert!(result.is_err());
    let result = PIPModel::<4>::new(
        "blosum",
        &[
            0.2, 0.5, 0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135,
        ],
        false,
    );
    assert!(result.is_err());
}

#[test]
fn pip_dna_char_frequencies() {
    let (params, char_probs) = gtr_char_probs_data();
    let pip_gtr = PIPModel::<4>::new("gtr", &[vec![0.2, 0.4], params].concat(), false).unwrap();
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

#[rstest]
#[case::wag("wag", &WAG_PI_ARR, 1e-8)]
#[case::blosum("blosum", &BLOSUM_PI_ARR, 1e-5)]
#[case::hivb("hivb", &HIVB_PI_ARR, 1e-8)]
fn protein_char_probabilities(
    #[case] model_name: &str,
    #[case] pi_array: &[f64],
    #[case] epsilon: f64,
) {
    let mut pip = PIPModel::<20>::new(model_name, &[0.4, 0.1], false).unwrap();
    pip.normalise();
    let expected = protein_char_probs_data(pi_array);
    for (char, expected_prot) in expected.into_iter() {
        let expected_pip = expected_prot.clone().insert_row(20, 0.0);
        let actual = pip.get_char_probability(char);
        assert_eq!(actual.len(), 21);
        assert_relative_eq!(actual.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(actual, expected_pip, epsilon = epsilon);
    }
    let actual = pip.get_char_probability(b'-');
    assert_eq!(actual.len(), 21);
    assert_relative_eq!(actual.sum(), 1.0);
    assert_relative_eq!(
        actual,
        FreqVector::from_column_slice(&[0.0; 20]).insert_row(20, 1.0)
    );
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_rates(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let pip_params = [0.2, 0.15];
    let pip_model = PIPModel::<4>::new(
        model_name,
        &[&pip_params, model_params.clone()].concat(),
        false,
    )
    .unwrap();
    let subst_model = DNASubstModel::new(model_name, model_params, false).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES_STR, &pip_model, &subst_model);
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_p_matrix_inf(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let pip_params = vec![0.2, 0.5];
    let pip = PIPModel::<4>::new(model_name, &[&pip_params, model_params].concat(), false).unwrap();
    let p = pip.get_p(10000000.0);
    let expected = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(p, expected, epsilon = 1e-10);
}

#[test]
fn pip_p_example_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let epsilon = 1e-3;
    let pip_params = vec![0.5, 0.25];
    let hky_params = vec![0.22, 0.26, 0.33, 0.19, 0.5];
    let pip_hky =
        PIPModel::<4>::new("hky", &[pip_params, hky_params.clone()].concat(), false).unwrap();
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
    assert_relative_eq!(pip_hky.get_p(2.0), expected_p, epsilon = epsilon);
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
