use rstest::*;

use std::f64::consts::PI;
use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dvector, DMatrix, DVector};
use nalgebra::{Const, DimMin};

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{phyloinfo_from_files, phyloinfo_from_sequences_newick, PhyloInfo};
use crate::pip_model::{PIPLikelihoodCost, PIPModel, PIPModelInfo};
use crate::sequences::{charify, AMINOACIDS_STR, NUCLEOTIDES_STR};
use crate::substitution_models::{
    dna_models::DNASubstModel,
    protein_models::{ProteinSubstModel, BLOSUM_PI_ARR, HIVB_PI_ARR, WAG_PI_ARR},
    substitution_models_tests::{gtr_char_probs_data, protein_char_probs_data},
    FreqVector, SubstMatrix, SubstitutionModel,
};

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
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_model),
        &frequencies
    );
    let subst_model = ProteinSubstModel::new(model_name, &[], false).unwrap();
    compare_pip_subst_rates(AMINOACIDS_STR, &pip_model, &subst_model);
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4], false).unwrap();
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    let jc96 = DNASubstModel::new("jc69", &[], false).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES_STR, &pip_jc69, &jc96);
}

#[test]
fn pip_dna_jc69_normalised() {
    let mut pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4], true).unwrap();
    EvolutionaryModel::normalise(&mut pip_jc69);
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    for char in charify(NUCLEOTIDES_STR) {
        assert_eq!(EvolutionaryModel::get_rate(&pip_jc69, char, char), -1.0);
        assert_relative_eq!(
            pip_jc69.q.row(pip_jc69.index[char as usize] as usize).sum(),
            0.0,
        );
        assert_relative_eq!(
            EvolutionaryModel::get_rate(&pip_jc69, char, b'-'),
            0.4 / 1.4
        );
        assert_relative_eq!(EvolutionaryModel::get_rate(&pip_jc69, b'-', char), 0.0);
    }
}

#[test]
fn pip_protein_wag_normalised() {
    let mut pip_wag = PIPModel::<20>::new("wag", &[0.1, 0.4], true).unwrap();
    EvolutionaryModel::normalise(&mut pip_wag);
    assert_eq!(pip_wag.lambda, 0.1);
    assert_eq!(pip_wag.mu, 0.4);
    let stat_dist = DVector::from_column_slice(&WAG_PI_ARR).insert_row(20, 0.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_wag),
        &stat_dist
    );
    assert_relative_eq!(pip_wag.q.sum(), 0.0, epsilon = 1e-10);
    for char in charify(NUCLEOTIDES_STR) {
        assert_relative_eq!(
            pip_wag.q.row(pip_wag.index[char as usize] as usize).sum(),
            0.0,
        );
        assert_relative_eq!(
            EvolutionaryModel::get_rate(&pip_wag, char, b'-'),
            0.4 / 1.4,
            epsilon = 1e-5
        );
        assert_relative_eq!(EvolutionaryModel::get_rate(&pip_wag, b'-', char), 0.0);
    }
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
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_tn93),
        &dvector![0.22, 0.26, 0.33, 0.19, 0.0]
    );
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
    EvolutionaryModel::normalise(&mut pip);
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
    let pip_model =
        PIPModel::<4>::new(model_name, &[&pip_params, model_params].concat(), false).unwrap();
    let subst_model = DNASubstModel::new(model_name, model_params, false).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES_STR, &pip_model, &subst_model);
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_dna_p_matrix_inf(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let pip_params = vec![0.2, 0.5];
    let pip = PIPModel::<4>::new(model_name, &[&pip_params, model_params].concat(), false).unwrap();
    let p = EvolutionaryModel::get_p(&pip, 10000000.0);
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

#[rstest]
#[case::wag("wag", &[1.2, 0.95])]
#[case::blosum("blosum", &[0.2, 0.5])]
#[case::hivb("hivb", &[0.1, 0.04])]
fn pip_protein_p_matrix_inf(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let pip = PIPModel::<20>::new(model_name, model_params, false).unwrap();
    let p = EvolutionaryModel::get_p(&pip, 10000000.0);
    let mut expected = SubstMatrix::zeros(21, 21);
    expected.fill_column(20, 1.0);
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
    assert_relative_eq!(
        EvolutionaryModel::get_p(&pip_hky, 2.0),
        expected_p,
        epsilon = epsilon
    );
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.437, 0.0859, 0.162, 0.0935, 0.221, 0.0727, 0.45, 0.162, 0.0935, 0.221, 0.108, 0.128,
            0.48, 0.0625, 0.221, 0.108, 0.128, 0.108, 0.434, 0.221, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(
        EvolutionaryModel::get_p(&pip_hky, 1.0),
        expected_p,
        epsilon = epsilon
    );
}

#[test]
fn pip_likelihood_no_msa() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .unwrap();
    let model_wag = PIPModel::<4>::new("jc69", &[0.5, 0.25], false).unwrap();
    assert!(PIPModelInfo::<4>::new(&info, &model_wag).is_err());
}

#[cfg(test)]
fn setup_example_phylo_info() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A", None, b"-A--"),
        Record::with_attrs("B", None, b"CA--"),
        Record::with_attrs("C", None, b"-A-G"),
        Record::with_attrs("D", None, b"-CAA"),
    ];
    phyloinfo_from_sequences_newick(&sequences, "((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;").unwrap()
}

#[test]
fn pip_hky_likelihood_example_leaf_values() {
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let info = setup_example_phylo_info();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    let nu = 7.5;
    let iota = 0.1333;
    let beta = 0.78694;
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("A").unwrap().into());
    assert_values(
        &cost,
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        nu * iota * beta,
        &[[0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.0],
        &[0.0, 0.33 * nu * iota * beta, 0.0, 0.0],
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("B").unwrap().into());
    assert_values(
        &cost,
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        nu * iota * beta,
        &[[1.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.26, 0.33, 0.0, 0.0],
        &[0.26 * nu * iota * beta, 0.33 * nu * iota * beta, 0.0, 0.0],
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("C").unwrap().into());
    let iota = 0.0667;
    let beta = 0.8848;
    assert_values(
        &cost,
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        nu * iota * beta,
        &[[0.0, 1.0, 0.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.19],
        &[0.0, 0.33 * nu * iota * beta, 0.0, 0.19 * nu * iota * beta],
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("D").unwrap().into());
    assert_values(
        &cost,
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        nu * iota * beta,
        &[[0.0, 1.0, 1.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.26, 0.33, 0.33],
        &[
            0.0,
            0.26 * nu * iota * beta,
            0.33 * nu * iota * beta,
            0.33 * nu * iota * beta,
        ],
    );
}

#[test]
fn pip_hky_likelihood_example_internals() {
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let info = setup_example_phylo_info();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    for i in 0..4 {
        cost.set_leaf_values(i);
    }
    let nu = 7.5;
    let iota = 0.1333;
    let beta = 0.78694;
    let idx = usize::from(cost.info.tree.get_idx_by_id("E").unwrap());
    cost.set_internal_values(idx);
    assert_values(
        &cost,
        idx + 4,
        nu * iota * beta,
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0619, 0.0431, 0.154, 0.154],
        &[
            0.0619 * nu * iota * beta + 0.26 * nu * iota * beta,
            0.0431 * nu * iota * beta,
            0.0,
            0.0,
        ],
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("F").unwrap());
    cost.set_internal_values(idx);
    let nu = 7.5;
    let iota_f = 0.2;
    let beta_f = 0.70351;
    let iota_d = 0.0667;
    let beta_d = 0.8848;
    assert_values(
        &cost,
        idx + 4,
        nu * iota_f * beta_f,
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0488, 0.0449, 0.0567, 0.0261],
        &[
            0.0,
            0.0449 * nu * iota_f * beta_f,
            0.0567 * nu * iota_f * beta_f + 0.33 * nu * iota_d * beta_d,
            0.0261 * nu * iota_f * beta_f,
        ],
    );

    let nu = 7.5;
    let iota = 0.2667;
    let beta = 1.0;
    let iota_e = 0.1333;
    let beta_e = 0.78694;
    let idx = usize::from(cost.info.tree.get_idx_by_id("R").unwrap());
    cost.set_root_values(idx);
    assert_values(
        &cost,
        idx + 4,
        nu * iota * beta,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        &[0.0207, 0.000557, 0.013, 0.00598],
        &[
            0.0207 * nu * iota * beta + 0.0619 * nu * iota_e * beta_e + 0.26 * nu * iota_e * beta_e,
            0.000557 * nu * iota * beta,
            0.013 * nu * iota * beta + 0.0567 * nu * iota_f * beta_f + 0.33 * nu * iota_d * beta_d,
            0.00598 * nu * iota * beta + 0.0261 * nu * iota_f * beta_f,
        ],
    );
}

#[cfg(test)]
fn assert_values<const N: usize>(
    cost: &PIPLikelihoodCost<N>,
    node_id: usize,
    exp_survins: f64,
    exp_anc: &[f64],
    exp_f: &[f64],
    exp_pnu: &[f64],
) {
    let e = 1e-3;
    assert_relative_eq!(cost.tmp.surv_ins_weights[node_id], exp_survins, epsilon = e);
    assert_relative_eq!(
        cost.tmp.f[node_id],
        DVector::<f64>::from_column_slice(exp_f),
        epsilon = e
    );
    assert_eq!(cost.tmp.anc[node_id].nrows(), exp_f.len());
    assert_eq!(cost.tmp.anc[node_id].ncols(), 3);
    assert_relative_eq!(
        cost.tmp.anc[node_id].as_slice(),
        DMatrix::<f64>::from_column_slice(exp_anc.len(), 1, exp_anc).as_slice(),
    );
    assert_relative_eq!(
        cost.tmp.pnu[node_id],
        DVector::<f64>::from_column_slice(exp_pnu),
        epsilon = e
    );
}

#[test]
fn pip_hky_likelihood_example_c0() {
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let info = setup_example_phylo_info();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("A").unwrap().into());
    assert_c0_values(
        &cost,
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        -1.0,
        7.5 * 0.028408 - 1.0,
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("B").unwrap().into());
    assert_c0_values(
        &cost,
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        -1.0,
        7.5 * 0.028408 - 1.0,
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("C").unwrap().into());
    assert_c0_values(
        &cost,
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        -1.0,
        7.5 * 0.00768 - 0.5,
    );
    cost.set_leaf_values(cost.info.tree.get_idx_by_id("D").unwrap().into());
    assert_c0_values(
        &cost,
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        -1.0,
        7.5 * 0.00768 - 0.5,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("E").unwrap());
    cost.set_internal_values(idx);
    assert_c0_values(
        &cost,
        idx + 4,
        -0.84518,
        7.5 * 0.044652 + 7.5 * 0.028408 * 2.0 - 3.0,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("F").unwrap());
    cost.set_internal_values(idx);
    assert_c0_values(
        &cost,
        idx + 4,
        -0.9512,
        7.5 * 0.066164 + 7.5 * 0.00768 * 2.0 - 2.5,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("R").unwrap());
    cost.set_root_values(idx);
    assert_c0_values(
        &cost,
        idx + 4,
        -0.732,
        7.5 * 0.071467
            + 7.5 * 0.044652
            + 7.5 * 0.028408 * 2.0
            + 7.5 * 0.066164
            + 7.5 * 0.00768 * 2.0
            - 7.5,
    )
}

#[cfg(test)]
fn assert_c0_values<const N: usize>(
    cost: &PIPLikelihoodCost<N>,
    node_id: usize,
    exp_f1: f64,
    exp_pnu: f64,
) {
    let e = 1e-3;
    assert_relative_eq!(cost.tmp.c0_f1[node_id], exp_f1, epsilon = e);
    assert_relative_eq!(cost.tmp.c0_pnu[node_id], exp_pnu, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_final() {
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let info = setup_example_phylo_info();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    for i in 0..4 {
        cost.set_leaf_values(i);
    }
    cost.set_internal_values(1);
    cost.set_internal_values(2);
    cost.set_root_values(0);
    assert_relative_eq!(
        cost.tmp.pnu[4],
        DVector::from_column_slice(&[
            7.5 * 0.0392204949,
            7.5 * 0.000148719,
            7.5 * 0.03102171,
            7.5 * 0.00527154
        ]),
        epsilon = 1e-3
    );
    assert_relative_eq!(cost.tmp.c0_pnu[4], -5.591, epsilon = 1e-3);
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -20.769363665853653 - 0.709020450847471 + (2.0 * PI).ln() / 2.0,
        epsilon = 1e-3
    );
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -21.476307347643274 + (2.0 * PI).ln() / 2.0, // value from the python script
        epsilon = 1e-2
    );
}

#[cfg(test)]
fn setup_example_phylo_info_2() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A", None, b"--A--"),
        Record::with_attrs("B", None, b"-CA--"),
        Record::with_attrs("C", None, b"--A-G"),
        Record::with_attrs("D", None, b"T-CAA"),
    ];
    phyloinfo_from_sequences_newick(&sequences, "((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;").unwrap()
}

#[test]
fn pip_hky_likelihood_example_2() {
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let info = setup_example_phylo_info_2();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -24.9549393298 + (2.0 * PI).ln() / 2.0,
        epsilon = 1e-2
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let model =
        PIPModel::<4>::new("hky", &[0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5], false).unwrap();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -377.04579323450395 + (2.0 * PI).ln() / 2.0,
        epsilon = 1e-4
    );

    let model =
        PIPModel::<4>::new("hky", &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0], false).unwrap();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -362.93914579664727 + (2.0 * PI).ln() / 2.0, // value from the python script
        epsilon = 1e-1
    );

    let model = PIPModel::<4>::new(
        "gtr",
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
        false,
    )
    .unwrap();
    let temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model,
        tmp: temp_values,
    };
    assert_relative_eq!(
        cost.compute_log_likelihood(),
        -359.38117790814533 + (2.0 * PI).ln() / 2.0,
        epsilon = 1e-4
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example_model_comp() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let model_hky_as_jc69 =
        PIPModel::<4>::new("hky", &[1.1, 0.55, 0.25, 0.25, 0.25, 0.25, 1.0], true).unwrap();
    let temp_values_1 = PIPModelInfo::<4>::new(&info, &model_hky_as_jc69).unwrap();
    let mut cost_1 = PIPLikelihoodCost {
        info: &info,
        model: model_hky_as_jc69,
        tmp: temp_values_1,
    };
    let model_jc69 = PIPModel::<4>::new("jc69", &[1.1, 0.55], true).unwrap();
    let temp_values_2 = PIPModelInfo::<4>::new(&info, &model_jc69).unwrap();
    let mut cost_2 = PIPLikelihoodCost {
        info: &info,
        model: model_jc69,
        tmp: temp_values_2,
    };
    assert_relative_eq!(
        cost_1.compute_log_likelihood(),
        cost_2.compute_log_likelihood(),
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example_reroot() {
    let info_1 = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let model_gtr_1 = PIPModel::<4>::new(
        "gtr",
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
        false,
    )
    .unwrap();
    let temp_values_1 = PIPModelInfo::<4>::new(&info_1, &model_gtr_1).unwrap();
    let mut cost_1 = PIPLikelihoodCost {
        info: &info_1,
        model: model_gtr_1,
        tmp: temp_values_1,
    };

    let info_2 = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .unwrap();
    let model_gtr_2 = PIPModel::<4>::new(
        "gtr",
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
        false,
    )
    .unwrap();

    let temp_values_2 = PIPModelInfo::<4>::new(&info_2, &model_gtr_2).unwrap();
    let mut cost_2 = PIPLikelihoodCost {
        info: &info_2,
        model: model_gtr_2,
        tmp: temp_values_2,
    };
    assert_relative_eq!(
        cost_1.compute_log_likelihood(),
        cost_2.compute_log_likelihood(),
    );
}

#[test]
fn pip_likelihood_protein_example() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/phyml_protein_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
    )
    .unwrap();
    let model_wag = PIPModel::<20>::new("wag", &[0.5, 0.25], false).unwrap();
    let temp_values = PIPModelInfo::<20>::new(&info, &model_wag).unwrap();
    let mut cost = PIPLikelihoodCost {
        info: &info,
        model: model_wag,
        tmp: temp_values,
    };
    assert!(cost.compute_log_likelihood() <= 0.0);
}
