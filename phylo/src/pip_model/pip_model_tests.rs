use rstest::*;

use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dvector, Const, DMatrix, DVector, DimMin};

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::make_freqs;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::pip_model::{PIPLikelihoodCost, PIPModel, PIPModelInfo};
use crate::sequences::{AMINOACIDS, GAP, NUCLEOTIDES};
use crate::substitution_models::dna_models::DNA_GAP_SETS;
use crate::substitution_models::protein_models::PROTEIN_GAP_SETS;
use crate::substitution_models::{
    dna_models::DNASubstModel,
    protein_models::{ProteinSubstModel, BLOSUM_PI_ARR, HIVB_PI_ARR, WAG_PI_ARR},
    substitution_models_tests::{gtr_char_probs_data, protein_char_probs_data},
    FreqVector, SubstMatrix, SubstitutionModel,
};
use crate::tree::{tree_parser, Tree};

const UNNORMALIZED_PIP_HKY_Q: [f64; 25] = [
    -0.9, 0.11, 0.22, 0.22, 0.0, 0.13, -0.88, 0.26, 0.26, 0.0, 0.33, 0.33, -0.825, 0.165, 0.0,
    0.19, 0.19, 0.095, -0.895, 0.0, 0.25, 0.25, 0.25, 0.25, -0.0,
];
const PIP_HKY_PARAMS: [f64; 7] = [0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5];

#[cfg(test)]
fn tree_newick(newick: &str) -> Tree {
    tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap()
}

#[cfg(test)]
fn compare_pip_subst_rates<const N: usize>(
    chars: &[u8],
    pip_model: &PIPModel<N>,
    subst_model: &SubstitutionModel<N>,
) where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    for &char in chars {
        assert!(pip_model.get_rate(char, char) < 0.0);
        assert_relative_eq!(
            pip_model.q.row(pip_model.index[char as usize]).sum(),
            0.0,
            epsilon = 1e-10
        );
        for &other_char in chars {
            if char == other_char {
                continue;
            }
            assert_relative_eq!(
                pip_model.get_rate(char, other_char),
                subst_model.get_rate(char, other_char)
            );
        }
        assert_relative_eq!(pip_model.get_rate(char, b'-'), pip_model.mu);
        assert_relative_eq!(pip_model.get_rate(GAP, char), 0.0);
    }
}

#[rstest]
#[case::wag("wag", &[0.1, 0.4], &WAG_PI_ARR)]
#[case::blosum("blosum", &[0.8, 0.25], &BLOSUM_PI_ARR)]
#[case::hivb("hivb", &[1.1, 12.4], &HIVB_PI_ARR)]
fn protein_pip_correct(#[case] model_name: &str, #[case] params: &[f64], #[case] pi_array: &[f64]) {
    let pip_model = PIPModel::<20>::new(model_name, params).unwrap();
    assert_eq!(pip_model.lambda, params[0]);
    assert_eq!(pip_model.mu, params[1]);
    let frequencies = make_freqs!(pi_array).insert_row(20, 0.0);
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_model),
        &frequencies
    );
    let subst_model = ProteinSubstModel::new(model_name, &[]).unwrap();
    compare_pip_subst_rates(AMINOACIDS, &pip_model, &subst_model);
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    let jc96 = DNASubstModel::new("jc69", &[]).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_jc69, &jc96);
}

#[test]
fn pip_dna_jc69_normalised() {
    let pip_jc69 = PIPModel::<4>::new("jc69", &[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.lambda, 0.1);
    assert_eq!(pip_jc69.mu, 0.4);
    assert_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    for &char in NUCLEOTIDES {
        assert_eq!(
            EvolutionaryModel::get_rate(&pip_jc69, char, char),
            -1.0 - 0.4
        );
        assert_relative_eq!(pip_jc69.q.row(pip_jc69.index[char as usize]).sum(), 0.0,);
        assert_relative_eq!(EvolutionaryModel::get_rate(&pip_jc69, char, b'-'), 0.4);
        assert_relative_eq!(EvolutionaryModel::get_rate(&pip_jc69, b'-', char), 0.0);
    }
}

#[test]
fn pip_protein_wag_normalised() {
    let pip_wag = PIPModel::<20>::new("wag", &[0.1, 0.4]).unwrap();
    assert_eq!(pip_wag.lambda, 0.1);
    assert_eq!(pip_wag.mu, 0.4);
    let stat_dist = DVector::from_column_slice(&WAG_PI_ARR).insert_row(20, 0.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&pip_wag),
        &stat_dist
    );
    assert_relative_eq!(pip_wag.q.sum(), 0.0, epsilon = 1e-10);
    for &char in NUCLEOTIDES {
        assert_relative_eq!(pip_wag.q.row(pip_wag.index[char as usize]).sum(), 0.0,);
        assert_relative_eq!(
            EvolutionaryModel::get_rate(&pip_wag, char, b'-'),
            0.4,
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
    )
    .unwrap();
    let tn93 = DNASubstModel::new(
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    let mut diff = SubstMatrix::zeros(4, 4);
    diff.fill_diagonal(-0.5);
    diff = diff.insert_column(4, 0.5).insert_row(4, 0.0);
    let expected_q = tn93.q.insert_column(4, 0.0).insert_row(4, 0.0) + diff;
    assert_relative_eq!(pip_tn93.q, expected_q, epsilon = 1e-10);
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
fn pip_dna_too_few_params(#[case] model_name: &str, #[case] params: &[f64]) {
    let result = PIPModel::<4>::new(model_name, params);
    assert!(result.is_err());
}

#[rstest]
#[case::wag_no_params("wag", &[])]
#[case::blosum("k80", &[0.2])]
#[case::hivb("hivb", &[0.22])]
#[case::wag_one_param("wag", &[0.1])]
fn pip_protein_too_few_params(#[case] model_name: &str, #[case] params: &[f64]) {
    let result = PIPModel::<20>::new(model_name, params);
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
fn pip_dna_char_frequencies() {
    let (params, char_probs) = gtr_char_probs_data();
    let pip_gtr = PIPModel::<4>::new("gtr", &[vec![0.2, 0.4], params].concat()).unwrap();
    for (&char, expected_gtr) in char_probs.iter() {
        let expected_pip = expected_gtr.clone().insert_row(4, 0.0);
        let actual = pip_gtr.get_char_probability(&DNA_GAP_SETS[char as usize]);
        assert_eq!(actual.len(), 5);
        assert_relative_eq!(actual.sum(), 1.0);
        assert_relative_eq!(actual, expected_pip, epsilon = 1e-4);
    }
    let actual = pip_gtr.get_char_probability(&DNA_GAP_SETS[b'-' as usize]);
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
    let pip = PIPModel::<20>::new(model_name, &[0.4, 0.1]).unwrap();
    let expected = protein_char_probs_data(pi_array);
    for (char, expected_prot) in expected.into_iter() {
        let expected_pip = expected_prot.clone().insert_row(20, 0.0);
        let actual = pip.get_char_probability(&PROTEIN_GAP_SETS[char as usize]);
        assert_eq!(actual.len(), 21);
        assert_relative_eq!(actual.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(actual, expected_pip, epsilon = epsilon);
    }
    let actual = pip.get_char_probability(&PROTEIN_GAP_SETS[b'-' as usize]);
    assert_eq!(actual.len(), 21);
    assert_relative_eq!(actual.sum(), 1.0);
    assert_relative_eq!(actual, make_freqs!(&[0.0; 20]).insert_row(20, 1.0));
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_rates(#[case] model_name: &str, #[case] params: &[f64]) {
    let pip_params = [0.2, 0.15];
    let pip_model = PIPModel::<4>::new(model_name, &[&pip_params, params].concat()).unwrap();
    let subst_model = DNASubstModel::new(model_name, params).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_model, &subst_model);
    let pip_model = PIPModel::<4>::new(model_name, &[&pip_params, params].concat()).unwrap();
    let subst_model = DNASubstModel::new(model_name, params).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_model, &subst_model);
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_dna_p_matrix_inf(#[case] model_name: &str, #[case] params: &[f64]) {
    let pip_params = vec![0.2, 0.5];
    let pip: PIPModel<4> = PIPModel::<4>::new(model_name, &[&pip_params, params].concat()).unwrap();
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
fn pip_protein_p_matrix_inf(#[case] model_name: &str, #[case] params: &[f64]) {
    let pip = PIPModel::<20>::new(model_name, params).unwrap();
    let p = EvolutionaryModel::get_p(&pip, 10000000.0);
    let mut expected = SubstMatrix::zeros(21, 21);
    expected.fill_column(20, 1.0);
    assert_relative_eq!(p, expected, epsilon = 1e-10);
}

#[test]
fn pip_p_example_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let epsilon = 1e-3;
    let mut pip_hky = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    pip_hky.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
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
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let model_jc69 = PIPModel::<4>::new("jc69", &[0.5, 0.25]).unwrap();
    assert!(PIPModelInfo::<4>::new(&info, &model_jc69).is_err());
}

#[cfg(test)]
fn setup_example_phylo_info() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A", None, b"-A--"),
        Record::with_attrs("B", None, b"CA--"),
        Record::with_attrs("C", None, b"-A-G"),
        Record::with_attrs("D", None, b"-CAA"),
    ];
    PhyloInfo::from_sequences_tree(
        sequences,
        tree_newick("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;"),
        &GapHandling::Proper,
    )
    .unwrap()
}

#[test]
fn pip_hky_likelihood_example_leaf_values() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let mut temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let cost = PIPLikelihoodCost { info: &info };

    let iota = 0.133;
    let beta = 0.787;
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        &model,
        &mut temp_values,
    );
    assert_values(
        &temp_values,
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        iota,
        beta,
        &[[0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.0],
        &[0.0, 0.33 * iota * beta, 0.0, 0.0],
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        &model,
        &mut temp_values,
    );
    assert_values(
        &temp_values,
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        iota,
        beta,
        &[[1.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.26, 0.33, 0.0, 0.0],
        &[0.26 * iota * beta, 0.33 * iota * beta, 0.0, 0.0],
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        &model,
        &mut temp_values,
    );
    let iota = 0.067;
    let beta = 0.885;
    assert_values(
        &temp_values,
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        iota,
        beta,
        &[[0.0, 1.0, 0.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.19],
        &[0.0, 0.33 * iota * beta, 0.0, 0.19 * iota * beta],
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        &model,
        &mut temp_values,
    );
    assert_values(
        &temp_values,
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        iota,
        beta,
        &[[0.0, 1.0, 1.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.26, 0.33, 0.33],
        &[
            0.0,
            0.26 * iota * beta,
            0.33 * iota * beta,
            0.33 * iota * beta,
        ],
    );
}

#[test]
fn pip_hky_likelihood_example_internals() {
    let mut model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info();
    let mut temp_values = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let cost = PIPLikelihoodCost { info: &info };
    for i in 0..4 {
        cost.set_leaf_values(i, &model, &mut temp_values);
    }
    let iota = 0.133;
    let beta = 0.787;
    let idx = usize::from(cost.info.tree.get_idx_by_id("E").unwrap());
    cost.set_internal_values(idx, &model, &mut temp_values);
    assert_values(
        &temp_values,
        idx + 4,
        iota,
        beta,
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0619, 0.0431, 0.154, 0.154],
        &[
            0.0619 * iota * beta + 0.26 * iota * beta,
            0.0431 * iota * beta,
            0.0,
            0.0,
        ],
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("F").unwrap());
    cost.set_internal_values(idx, &model, &mut temp_values);
    let iota_f = 0.2;
    let beta_f = 0.704;
    let iota_d = 0.067;
    let beta_d = 0.885;
    assert_values(
        &temp_values,
        idx + 4,
        iota_f,
        beta_f,
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0488, 0.0449, 0.0567, 0.0261],
        &[
            0.0,
            0.0449 * iota_f * beta_f,
            0.0567 * iota_f * beta_f + 0.33 * iota_d * beta_d,
            0.0261 * iota_f * beta_f,
        ],
    );

    let iota = 0.267;
    let beta = 1.0;
    let iota_e = 0.133;
    let beta_e = 0.787;
    let idx = usize::from(cost.info.tree.get_idx_by_id("R").unwrap());
    cost.set_root_values(idx, &model, &mut temp_values);
    assert_values(
        &temp_values,
        idx + 4,
        iota,
        beta,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        &[0.0207, 0.000557, 0.013, 0.00598],
        &[
            0.0207 * iota * beta + 0.0619 * iota_e * beta_e + 0.26 * iota_e * beta_e,
            0.000557 * iota * beta,
            0.013 * iota * beta + 0.0567 * iota_f * beta_f + 0.33 * iota_d * beta_d,
            0.00598 * iota * beta + 0.0261 * iota_f * beta_f,
        ],
    );
}

#[cfg(test)]
fn assert_values<const N: usize>(
    tmp: &PIPModelInfo<N>,
    node_id: usize,
    exp_ins: f64,
    exp_surv: f64,
    exp_anc: &[f64],
    exp_f: &[f64],
    exp_p: &[f64],
) {
    let e = 1e-3;
    assert_relative_eq!(tmp.ins_probs[node_id], exp_ins, epsilon = e);
    assert_relative_eq!(tmp.surv_probs[node_id], exp_surv, epsilon = e);
    assert_relative_eq!(
        tmp.f[node_id],
        DVector::<f64>::from_column_slice(exp_f),
        epsilon = e
    );
    assert_eq!(tmp.anc[node_id].nrows(), exp_f.len());
    assert_eq!(tmp.anc[node_id].ncols(), 3);
    assert_relative_eq!(
        tmp.anc[node_id].as_slice(),
        DMatrix::<f64>::from_column_slice(exp_anc.len(), 1, exp_anc).as_slice(),
    );
    assert_relative_eq!(
        tmp.p[node_id],
        DVector::<f64>::from_column_slice(exp_p),
        epsilon = e
    );
}

#[test]
fn pip_hky_likelihood_example_c0() {
    let mut model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info();
    let mut tmp = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let cost = PIPLikelihoodCost { info: &info };
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        &model,
        &mut tmp,
    );
    assert_c0_values(
        &tmp,
        cost.info.tree.get_idx_by_id("A").unwrap().into(),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.028329,
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        &model,
        &mut tmp,
    );
    assert_c0_values(
        &tmp,
        cost.info.tree.get_idx_by_id("B").unwrap().into(),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.028329,
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        &model,
        &mut tmp,
    );
    assert_c0_values(
        &tmp,
        cost.info.tree.get_idx_by_id("C").unwrap().into(),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.007705,
    );
    cost.set_leaf_values(
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        &model,
        &mut tmp,
    );
    assert_c0_values(
        &tmp,
        cost.info.tree.get_idx_by_id("D").unwrap().into(),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.007705,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("E").unwrap());
    cost.set_internal_values(idx, &model, &mut tmp);
    assert_c0_values(
        &tmp,
        idx + 4,
        &[0.154, 0.154, 0.154, 0.154, 1.0],
        0.154,
        0.044448334 + 0.028329 * 2.0,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("F").unwrap());
    cost.set_internal_values(idx, &model, &mut tmp);
    assert_c0_values(
        &tmp,
        idx + 4,
        &[0.0488, 0.0488, 0.0488, 0.0488, 1.0],
        0.0488,
        0.06607104 + 0.007705 * 2.0,
    );
    let idx = usize::from(cost.info.tree.get_idx_by_id("R").unwrap());
    cost.set_root_values(idx, &model, &mut tmp);
    assert_c0_values(
        &tmp,
        idx + 4,
        &[0.268, 0.268, 0.268, 0.268, 1.0],
        0.268,
        0.071556 + 0.044448334 + 0.028329 * 2.0 + 0.06607104 + 0.007705 * 2.0,
    )
}

#[cfg(test)]
fn assert_c0_values<const N: usize>(
    tmp: &PIPModelInfo<N>,
    node_id: usize,
    exp_ftilde: &[f64],
    exp_f: f64,
    exp_p: f64,
) {
    let e = 1e-3;
    assert_relative_eq!(
        tmp.c0_ftilde[node_id],
        DMatrix::<f64>::from_column_slice(N + 1, 1, exp_ftilde),
        epsilon = e
    );
    assert_relative_eq!(tmp.c0_f[node_id], exp_f, epsilon = e);
    assert_relative_eq!(tmp.c0_p[node_id], exp_p, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_final() {
    let mut model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info();
    let mut tmp = PIPModelInfo::<4>::new(&info, &model).unwrap();
    let cost = PIPLikelihoodCost { info: &info };
    for i in 0..4 {
        cost.set_leaf_values(i, &model, &mut tmp);
    }
    cost.set_internal_values(1, &model, &mut tmp);
    cost.set_internal_values(2, &model, &mut tmp);
    cost.set_root_values(0, &model, &mut tmp);
    assert_relative_eq!(
        tmp.p[4],
        DVector::from_column_slice(&[0.0392204949, 0.000148719, 0.03102171, 0.00527154]),
        epsilon = 1e-3
    );
    assert_relative_eq!(tmp.c0_p[4], 0.254143374, epsilon = 1e-3);
    assert_relative_eq!(
        cost.compute_log_likelihood_with_tmp(&model, &mut tmp),
        -20.769363665853653 - 0.709020450847471,
        epsilon = 1e-3
    );
    assert_relative_eq!(
        cost.compute_log_likelihood_with_tmp(&model, &mut tmp),
        -21.476307347643274, // value from the python script
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
    PhyloInfo::from_sequences_tree(
        sequences,
        tree_newick("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;"),
        &GapHandling::Proper,
    )
    .unwrap()
}

#[test]
fn pip_hky_likelihood_example_2() {
    let mut model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info_2();
    let cost = PIPLikelihoodCost::<4> { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost, &model),
        -24.9549393298,
        epsilon = 1e-2
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let model = PIPModel::<4>::new("hky", &PIP_HKY_PARAMS).unwrap();
    let cost = PIPLikelihoodCost::<4> { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost, &model),
        -372.1419415285655,
        epsilon = 1e-4
    );

    let model = PIPModel::<4>::new("hky", &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    let cost = PIPLikelihoodCost::<4> { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost, &model),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );

    let model = PIPModel::<4>::new(
        "gtr",
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
    )
    .unwrap();
    let cost = PIPLikelihoodCost::<4> { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost, &model),
        -359.2343309917135,
        epsilon = 1e-4
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example_model_comp() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let model_hky_as_jc69 =
        PIPModel::<4>::new("hky", &[1.1, 0.55, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    let cost_1 = PIPLikelihoodCost::<4> { info: &info };
    let model_jc69 = PIPModel::<4>::new("jc69", &[1.1, 0.55]).unwrap();
    let cost_2 = PIPLikelihoodCost::<4> { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost_1, &model_hky_as_jc69),
        LikelihoodCostFunction::compute_log_likelihood(&cost_2, &model_jc69),
    );
}

#[test]
fn pip_likelihood_huelsenbeck_example_reroot() {
    let info_1 = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let model_gtr = PIPModel::<4>::new(
        "gtr",
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
    )
    .unwrap();
    let cost_1 = PIPLikelihoodCost::<4> { info: &info_1 };

    let info_2 = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let cost_2 = PIPLikelihoodCost::<4> { info: &info_2 };

    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost_1, &model_gtr),
        LikelihoodCostFunction::compute_log_likelihood(&cost_2, &model_gtr),
    );
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost_1, &model_gtr),
        -359.2343309917135,
        epsilon = 1e-4
    );
}

#[test]
fn pip_likelihood_protein_example() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/phyml_protein_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let model_wag = PIPModel::<20>::new("wag", &[0.5, 0.25]).unwrap();
    let cost = PIPLikelihoodCost::<20> { info: &info };
    assert!(LikelihoodCostFunction::compute_log_likelihood(&cost, &model_wag) <= 0.0);
}
