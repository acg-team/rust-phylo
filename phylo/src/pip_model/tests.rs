use rstest::*;

use std::cell::RefCell;
use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dvector, DMatrix, DVector};

use crate::alignment::Sequences;
use crate::alphabets::{AMINOACIDS, GAP, NUCLEOTIDES};
use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvoModel,
    ProteinModelType::{self, *},
};
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPDNAModel, PIPModel, PIPModelInfo, PIPParameter, PIPProteinModel};
use crate::substitution_models::{
    blosum_freqs, hivb_freqs, wag_freqs, DNASubstModel, FreqVector, ProteinSubstModel, SubstMatrix,
    SubstitutionModel,
};
use crate::tree::tree_parser::from_newick;

use crate::{frequencies, record_wo_desc as record, tree};

const UNNORMALIZED_PIP_HKY_Q: [f64; 25] = [
    -0.9, 0.11, 0.22, 0.22, 0.0, 0.13, -0.88, 0.26, 0.26, 0.0, 0.33, 0.33, -0.825, 0.165, 0.0,
    0.19, 0.19, 0.095, -0.895, 0.0, 0.25, 0.25, 0.25, 0.25, -0.0,
];
const PIP_HKY_PARAMS: [f64; 7] = [0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 0.5];

#[cfg(test)]
fn compare_pip_subst_rates<SM: SubstitutionModel + Clone>(
    chars: &[u8],
    pip_model: &PIPModel<SM>,
    subst_model: &SM,
) where
    SM::ModelType: Clone,
    PIPModel<SM>: EvoModel,
{
    for &char in chars {
        assert!(pip_model.rate(char, char) < 0.0);
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
                pip_model.rate(char, other_char),
                subst_model.rate(char, other_char),
                epsilon = 1e-10
            );
        }
        assert_relative_eq!(pip_model.rate(char, b'-'), pip_model.params.mu);
        assert_relative_eq!(pip_model.rate(GAP, char), 0.0);
    }
}

#[rstest]
#[case::wag(WAG, &[0.1, 0.4], wag_freqs())]
#[case::blosum(BLOSUM, &[0.8, 0.25], blosum_freqs())]
#[case::hivb(HIVB, &[1.1, 12.4], hivb_freqs())]
fn protein_pip_correct(
    #[case] model_type: ProteinModelType,
    #[case] params: &[f64],
    #[case] pi_array: FreqVector,
) {
    let pip_model = PIPProteinModel::new(model_type, params).unwrap();
    assert_eq!(pip_model.params.lambda, params[0]);
    assert_eq!(pip_model.params.mu, params[1]);
    let frequencies = pi_array.insert_row(20, 0.0);
    assert_eq!(EvoModel::freqs(&pip_model), &frequencies);
    let subst_model = <ProteinSubstModel as SubstitutionModel>::new(model_type, &[]).unwrap();
    compare_pip_subst_rates(AMINOACIDS, &pip_model, &subst_model);
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPDNAModel::new(JC69, &[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.params.lambda, 0.1);
    assert_eq!(pip_jc69.params.mu, 0.4);
    assert_eq!(
        EvoModel::freqs(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    let jc96 = <DNASubstModel as SubstitutionModel>::new(JC69, &[]).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_jc69, &jc96);
}

#[test]
fn pip_dna_jc69_normalised() {
    let pip_jc69 = PIPDNAModel::new(JC69, &[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.params.lambda, 0.1);
    assert_eq!(pip_jc69.params.mu, 0.4);
    assert_eq!(
        EvoModel::freqs(&pip_jc69),
        &dvector![0.25, 0.25, 0.25, 0.25, 0.0]
    );
    for &char in NUCLEOTIDES {
        assert_eq!(EvoModel::rate(&pip_jc69, char, char), -1.0 - 0.4);
        assert_relative_eq!(pip_jc69.q.row(pip_jc69.index[char as usize]).sum(), 0.0,);
        assert_relative_eq!(EvoModel::rate(&pip_jc69, char, b'-'), 0.4);
        assert_relative_eq!(EvoModel::rate(&pip_jc69, b'-', char), 0.0);
    }
}

#[test]
fn pip_protein_wag_normalised() {
    let pip_wag = PIPProteinModel::new(WAG, &[0.1, 0.4]).unwrap();
    assert_eq!(pip_wag.params.lambda, 0.1);
    assert_eq!(pip_wag.params.mu, 0.4);
    let stat_dist = wag_freqs().insert_row(20, 0.0);
    assert_relative_eq!(EvoModel::freqs(&pip_wag), &stat_dist);
    assert_relative_eq!(pip_wag.q.sum(), 0.0, epsilon = 1e-10);
    for &char in NUCLEOTIDES {
        assert_relative_eq!(
            pip_wag.q.row(pip_wag.index[char as usize]).sum(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(EvoModel::rate(&pip_wag, char, b'-'), 0.4, epsilon = 1e-5);
        assert_relative_eq!(EvoModel::rate(&pip_wag, b'-', char), 0.0);
    }
}

#[test]
fn pip_dna_tn93_correct() {
    let pip_tn93 = PIPDNAModel::new(
        TN93,
        &[
            0.2, 0.5, 0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135,
        ],
    )
    .unwrap();
    let tn93 = <DNASubstModel as SubstitutionModel>::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    let mut diff = SubstMatrix::zeros(4, 4);
    diff.fill_diagonal(-0.5);
    diff = diff.insert_column(4, 0.5).insert_row(4, 0.0);
    let expected_q = tn93.q.insert_column(4, 0.0).insert_row(4, 0.0) + diff;
    assert_relative_eq!(pip_tn93.q, expected_q, epsilon = 1e-10);
    assert_relative_eq!(
        EvoModel::freqs(&pip_tn93),
        &dvector![0.22, 0.26, 0.33, 0.19, 0.0]
    );
}

#[rstest]
#[case::jc69(JC69, &[0.2])]
#[case::k80(K80, &[0.2])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93_too_few_for_subst(TN93, &[0.2, 0.5, 0.22, 0.26, 0.33, 0.19])]
#[case::tn93_too_few_for_pip(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_dna_too_few_params(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let result = PIPDNAModel::new(model_type, params);
    assert!(result.is_err());
}

#[rstest]
#[case::wag_no_params(WAG, &[])]
#[case::blosum(BLOSUM, &[0.2])]
#[case::hivb(HIVB, &[0.22])]
#[case::wag_one_param(WAG, &[0.1])]
fn pip_protein_too_few_params(#[case] model_type: ProteinModelType, #[case] params: &[f64]) {
    let result = PIPProteinModel::new(model_type, params);
    assert!(result.is_err());
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_rates(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let pip_params = [0.2, 0.15];
    let pip_model = PIPDNAModel::new(model_type, &[&pip_params, params].concat()).unwrap();
    let subst_model = <DNASubstModel as SubstitutionModel>::new(model_type, params).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_model, &subst_model);
    let pip_model = PIPDNAModel::new(model_type, &[&pip_params, params].concat()).unwrap();
    let subst_model = <DNASubstModel as SubstitutionModel>::new(model_type, params).unwrap();
    compare_pip_subst_rates(NUCLEOTIDES, &pip_model, &subst_model);
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn pip_dna_p_matrix_inf(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let pip_params = vec![0.2, 0.5];
    let pip = PIPDNAModel::new(model_type, &[&pip_params, params].concat()).unwrap();
    let p = EvoModel::p(&pip, 10000000.0);
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
#[case::wag(WAG, &[1.2, 0.95])]
#[case::blosum(BLOSUM, &[0.2, 0.5])]
#[case::hivb(HIVB, &[0.1, 0.04])]
fn pip_protein_p_matrix_inf(#[case] model_type: ProteinModelType, #[case] params: &[f64]) {
    let pip = PIPProteinModel::new(model_type, params).unwrap();
    let p = EvoModel::p(&pip, 10000000.0);
    let mut expected = SubstMatrix::zeros(21, 21);
    expected.fill_column(20, 1.0);
    assert_relative_eq!(p, expected, epsilon = 1e-10);
}

#[test]
fn pip_p_example_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let epsilon = 1e-3;
    let mut pip_hky = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    pip_hky.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.225, 0.109, 0.173, 0.0996, 0.393, 0.0922, 0.242, 0.173, 0.0996, 0.393, 0.115, 0.136,
            0.276, 0.0792, 0.393, 0.115, 0.136, 0.138, 0.217, 0.393, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(EvoModel::p(&pip_hky, 2.0), expected_p, epsilon = epsilon);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.437, 0.0859, 0.162, 0.0935, 0.221, 0.0727, 0.45, 0.162, 0.0935, 0.221, 0.108, 0.128,
            0.48, 0.0625, 0.221, 0.108, 0.128, 0.108, 0.434, 0.221, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(EvoModel::p(&pip_hky, 1.0), expected_p, epsilon = epsilon);
}

#[cfg(test)]
fn setup_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("A", b"-A--"),
        record!("B", b"CA--"),
        record!("C", b"-A-G"),
        record!("D", b"-CAA"),
    ]);
    PIB::build_from_objects(sequences, tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;")).unwrap()
}

#[test]
fn pip_hky_likelihood_example_leaf_values() {
    let info = setup_example_phylo_info();
    let tree = &info.tree;
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    model.tmp = RefCell::new(PIPModelInfo::new(&info, &model));
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let iota = 0.133;
    let beta = 0.787;

    let _logl = model.logl(&info);
    assert_values(
        &model.tmp.borrow(),
        usize::from(tree.idx("A")),
        iota,
        beta,
        &[[0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.0],
        &[0.0, 0.33 * iota * beta, 0.0, 0.0],
    );
    assert_values(
        &model.tmp.borrow(),
        usize::from(tree.idx("B")),
        iota,
        beta,
        &[[1.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.26, 0.33, 0.0, 0.0],
        &[0.26 * iota * beta, 0.33 * iota * beta, 0.0, 0.0],
    );
    let iota = 0.067;
    let beta = 0.885;
    assert_values(
        &model.tmp.borrow(),
        usize::from(tree.idx("C")),
        iota,
        beta,
        &[[0.0, 1.0, 0.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.19],
        &[0.0, 0.33 * iota * beta, 0.0, 0.19 * iota * beta],
    );
    assert_values(
        &model.tmp.borrow(),
        usize::from(tree.idx("D")),
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
    let info = setup_example_phylo_info();
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    model.tmp = RefCell::new(PIPModelInfo::new(&info, &model));
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let _logl = model.logl(&info);
    let tmp = model.tmp.borrow();

    let iota = 0.133;
    let beta = 0.787;
    assert_values(
        &tmp,
        usize::from(info.tree.idx("E")),
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

    let iota_f = 0.2;
    let beta_f = 0.704;
    let iota_d = 0.067;
    let beta_d = 0.885;
    assert_values(
        &tmp,
        usize::from(info.tree.idx("F")),
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
    assert_values(
        &tmp,
        usize::from(info.tree.idx("R")),
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
fn assert_values<SM: SubstitutionModel>(
    tmp: &PIPModelInfo<SM>,
    idx: usize,
    exp_ins: f64,
    exp_surv: f64,
    exp_anc: &[f64],
    exp_f: &[f64],
    exp_p: &[f64],
) {
    let e = 1e-3;
    assert_relative_eq!(tmp.ins_probs[idx], exp_ins, epsilon = e);
    assert_relative_eq!(tmp.surv_probs[idx], exp_surv, epsilon = e);
    assert_relative_eq!(
        tmp.f[idx],
        DVector::<f64>::from_column_slice(exp_f),
        epsilon = e
    );
    assert_eq!(tmp.anc[idx].nrows(), exp_f.len());
    assert_eq!(tmp.anc[idx].ncols(), 3);
    assert_relative_eq!(
        tmp.anc[idx].as_slice(),
        DMatrix::<f64>::from_column_slice(exp_anc.len(), 1, exp_anc).as_slice(),
    );
    assert_relative_eq!(
        tmp.p[idx],
        DVector::<f64>::from_column_slice(exp_p),
        epsilon = e
    );
}

#[test]
fn pip_hky_likelihood_example_c0() {
    let info = setup_example_phylo_info();
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    model.tmp = RefCell::new(PIPModelInfo::new(&info, &model));
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let _logl = model.logl(&info);
    let tmp = model.tmp.borrow();
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("A")),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.028329,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("B")),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.028329,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("C")),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.007705,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("D")),
        &[0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        0.007705,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("E")),
        &[0.154, 0.154, 0.154, 0.154, 1.0],
        0.154,
        0.044448334 + 0.028329 * 2.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("F")),
        &[0.0488, 0.0488, 0.0488, 0.0488, 1.0],
        0.0488,
        0.06607104 + 0.007705 * 2.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("R")),
        &[0.268, 0.268, 0.268, 0.268, 1.0],
        0.268,
        0.071556 + 0.044448334 + 0.028329 * 2.0 + 0.06607104 + 0.007705 * 2.0,
    );
}

#[cfg(test)]
fn assert_c0_values<SM: SubstitutionModel>(
    tmp: &PIPModelInfo<SM>,
    idx: usize,
    exp_ftilde: &[f64],
    exp_f: f64,
    exp_p: f64,
) {
    let e = 1e-3;
    assert_relative_eq!(
        tmp.c0_ftilde[idx],
        DMatrix::<f64>::from_column_slice(SM::N + 1, 1, exp_ftilde),
        epsilon = e
    );
    assert_relative_eq!(tmp.c0_f[idx], exp_f, epsilon = e);
    assert_relative_eq!(tmp.c0_p[idx], exp_p, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_final() {
    let info = setup_example_phylo_info();
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    model.tmp = RefCell::new(PIPModelInfo::new(&info, &model));
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let _logl = model.logl(&info);
    let tmp = model.tmp.borrow();

    assert_relative_eq!(
        tmp.p[0],
        DVector::from_column_slice(&[0.0392204949, 0.000148719, 0.03102171, 0.00527154]),
        epsilon = 1e-3
    );
    assert_relative_eq!(tmp.c0_p[0], 0.254143374, epsilon = 1e-3);
    assert_relative_eq!(
        model.logl(&info),
        -20.769363665853653 - 0.709020450847471,
        epsilon = 1e-3
    );
    assert_relative_eq!(
        model.logl(&info),
        -21.476307347643274, // value from the python script
        epsilon = 1e-2
    );
}

#[cfg(test)]
fn setup_example_phylo_info_2() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("A", b"--A--"),
        record!("B", b"-CA--"),
        record!("C", b"--A-G"),
        record!("D", b"T-CAA"),
    ]);
    PIB::build_from_objects(sequences, tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;")).unwrap()
}

#[test]
fn pip_hky_likelihood_example_2() {
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info_2();
    assert_relative_eq!(model.cost(&info, false), -24.9549393298, epsilon = 1e-2);
}

#[test]
fn pip_likelihood_huelsenbeck_example() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let mut model = PIPDNAModel::new(HKY, &PIP_HKY_PARAMS).unwrap();
    assert_relative_eq!(model.cost(&info, false), -372.1419415285655, epsilon = 1e-4);

    // Check that model update works
    model.set_param(&PIPParameter::Lambda, 1.2);
    model.set_param(&PIPParameter::Mu, 0.45);
    model.set_freqs(frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    model.set_param(&PIPParameter::Rag, 1.0);

    assert_relative_eq!(
        model.cost(&info, false),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );

    let model = PIPDNAModel::new(
        GTR,
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
    )
    .unwrap();
    assert_relative_eq!(model.cost(&info, true), -359.2343309917135, epsilon = 1e-4);
}

#[test]
fn pip_likelihood_huelsenbeck_example_model_comp() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let hky_as_jc = PIPDNAModel::new(HKY, &[1.1, 0.55, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    let jc69 = PIPDNAModel::new(JC69, &[1.1, 0.55]).unwrap();
    assert_relative_eq!(hky_as_jc.cost(&info, false), jc69.cost(&info, true));
}

#[test]
fn pip_likelihood_huelsenbeck_example_reroot() {
    let phylo = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let model_gtr = PIPDNAModel::new(
        GTR,
        &[
            0.5, 0.25, 0.22, 0.26, 0.33, 0.19, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031,
        ],
    )
    .unwrap();
    let phylo_rerooted = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();

    assert_relative_eq!(
        model_gtr.cost(&phylo, false),
        model_gtr.cost(&phylo_rerooted, true),
        epsilon = 1e-4
    );
    assert_relative_eq!(
        model_gtr.cost(&phylo, true),
        -359.2343309917135,
        epsilon = 1e-4
    );
}

#[test]
fn pip_likelihood_protein_example() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/true_tree.newick"),
    )
    .build()
    .unwrap();
    let model_wag = PIPProteinModel::new(WAG, &[0.5, 0.25]).unwrap();
    assert!(model_wag.cost(&info, false) <= 0.0);
}

#[test]
fn designation() {
    let model = PIPDNAModel::new(K80, &[0.5, 0.25, 1.0, 2.0]).unwrap();
    assert_eq!(model.description(), "PIP with K80");
    let model = PIPProteinModel::new(WAG, &[0.5, 0.25]).unwrap();
    assert_eq!(model.description(), "PIP with WAG");
}
