use rstest::*;

use std::path::{Path, PathBuf};

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::alphabets::protein_alphabet;
use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvoModel,
    ProteinModelType::{self, *},
};
use crate::frequencies;
use crate::io::read_sequences_from_file;
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::pip_model::PIPModel;
use crate::substitution_models::{
    DNAParameter, DNASubstModel, FreqVector, ProteinSubstModel, SubstMatrix,
};
use crate::tree::{tree_parser::from_newick_string, Tree};

macro_rules! record {
    ($e1:expr,$e2:expr) => {
        Record::with_attrs($e1, None, $e2)
    };
}

#[cfg(test)]
fn tree_newick(newick: &str) -> Tree {
    from_newick_string(newick).unwrap().pop().unwrap()
}

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = Sequences::new(vec![record!("A0", b"A"), record!("B1", b"A")]);
    let tree = tree_newick(format!("((A0:{},B1:{}):1.0);", blen_i, blen_j).as_str());
    PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_simple_likelihood() {
    let info = &setup_simple_phylo_info(1.0, 1.0);
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    assert_relative_eq!(jc69.cost(info, false), -2.5832498829317445, epsilon = 1e-6);
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let info = &setup_simple_phylo_info(1.0, 2.0);
    assert_relative_eq!(jc69.cost(info, false), -2.719098272533848, epsilon = 1e-6);
}

#[rstest]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn change_likelihood_on_freq_change(#[case] mtype: DNAModelType, #[case] params: &[f64]) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let mut model = DNASubstModel::new(mtype, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_ne!(logl, model.cost(&info, true));
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[2.0, 0.5])]
fn same_likelihood_on_freq_change(#[case] mtype: DNAModelType, #[case] params: &[f64]) {
    // likelihood should stay the same when frequencies are changed in models with fixed
    let info = setup_cb_example_phylo_info();
    let mut model = DNASubstModel::new(mtype, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_eq!(logl, model.cost(&info, true));
}

#[rstest]
#[case::k80(K80, &[2.0, 0.5])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn change_likelihood_on_param_change(#[case] mtype: DNAModelType, #[case] params: &[f64]) {
    // likelihood should change when parameters are changed
    let info = setup_cb_example_phylo_info();
    let mut model = DNASubstModel::new(mtype, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_param(&DNAParameter::Rca, 100.0);
    assert_ne!(logl, model.cost(&info, true));
}

#[rstest]
#[case::jc69(JC69, &[])]
fn same_likelihood_on_param_change(#[case] mtype: DNAModelType, #[case] params: &[f64]) {
    // likelihood should not change when parameters are changed for jc69
    let info = setup_cb_example_phylo_info();
    let mut model = DNASubstModel::new(mtype, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_param(&DNAParameter::Rca, 100.0);
    assert_eq!(logl, model.cost(&info, true));
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn gaps_as_ambigs(#[case] mtype: DNAModelType, #[case] params: &[f64]) {
    let tree = tree_newick("((one:2,two:2):1,(three:1,four:1):2);");
    let sequences = Sequences::new(vec![
        record!("one", b"CCCCCCXX"),
        record!("two", b"XXAAAAAA"),
        record!("three", b"TTTNNTTT"),
        record!("four", b"GNGGGGNG"),
    ]);
    let info_ambig = &PhyloInfoBuilder::build_from_objects(sequences, tree.clone()).unwrap();
    let sequences = Sequences::new(vec![
        record!("one", b"CCCCCC--"),
        record!("two", b"--AAAAAA"),
        record!("three", b"TTT--TTT"),
        record!("four", b"G-GGGG-G"),
    ]);
    let info_gaps = &PhyloInfoBuilder::build_from_objects(sequences, tree.clone()).unwrap();

    let model = DNASubstModel::new(mtype, params).unwrap();
    assert_eq!(model.cost(info_ambig, true), model.cost(info_gaps, true));
}

#[cfg(test)]
fn setup_phylo_info_single_leaf() -> PhyloInfo {
    let sequences = Sequences::new(vec![record!("A0", b"AAAAAA")]);
    let tree = Tree::new(&sequences).unwrap();
    PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_likelihood_one_node() {
    let info = &setup_phylo_info_single_leaf();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    assert!(model.cost(info, false) < 0.0);
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("one", b"C"),
        record!("two", b"A"),
        record!("three", b"T"),
        record!("four", b"G"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap()
}

#[test]
fn dna_cb_example_likelihood() {
    let info = &setup_cb_example_phylo_info();
    let mut model = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    model.q = SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -0.15594579,
            0.15524379,
            0.00044550000000000004,
            0.0002565,
            0.13136013,
            -0.13206213,
            0.00044550000000000004,
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
    assert_relative_eq!(model.cost(info, false), -17.1035117087, epsilon = 1e-6);
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("one", b"T"),
        record!("two", b"C"),
        record!("three", b"A"),
        record!("four", b"C"),
        record!("five", b"C"),
    ]);
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap()
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = &setup_mol_evo_example_phylo_info();
    let model = DNASubstModel::new(K80, &[]).unwrap();
    assert_relative_eq!(model.cost(info, false), -7.581408, epsilon = 1e-6);
}

#[test]
fn dna_ambig_example_likelihood_tn93() {
    // Checks that likelihoods for different ambiguous characters are the same
    let tn93 = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    let info_w_x = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let info_w_n = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();
    assert_relative_eq!(tn93.cost(info_w_x, true), tn93.cost(info_w_n, true));
}

#[test]
fn dna_ambig_example_likelihood_k80() {
    // Checks that likelihoods for different ambiguous characters are the same
    let k80 = DNASubstModel::new(K80, &[2.0, 1.0]).unwrap();
    let info_w_x = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let info_w_n = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();
    assert_relative_eq!(
        k80.cost(info_w_x, true),
        -137.24280493914029,
        epsilon = 1e-6
    );
    assert_relative_eq!(k80.cost(info_w_x, true), k80.cost(info_w_n, true));
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let gtr = DNASubstModel::new(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0]).unwrap();
    assert_relative_eq!(gtr.cost(info, false), -216.234734, epsilon = 1e-3);
}

#[rstest]
#[case::wag(WAG, &[], -4505.736814460457, 1e-3)]
#[case::hivb(HIVB, &[], -4407.989226397638, 1e-5)]
#[case::blosum(BLOSUM, &[], -4587.71053, 1e-5)]
fn protein_example_likelihood(
    #[case] model_type: ProteinModelType,
    #[case] params: &[f64],
    #[case] expected_llik: f64,
    #[case] epsilon: f64,
) {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/true_tree.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    assert_relative_eq!(model.cost(info, false), expected_llik, epsilon = epsilon);
}

#[cfg(test)]
fn simple_dna_reroot_info() -> (PhyloInfo, PhyloInfo) {
    let sequences = Sequences::new(vec![
        record!("A", b"CTATATATAC"),
        record!("B", b"ATATATATAA"),
        record!("C", b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(
        sequences.clone(),
        tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
    )
    .unwrap();
    let info_rerooted = PhyloInfoBuilder::build_from_objects(
        sequences,
        tree_newick("(A:1.0,(B:2.0,C:3.0):1.0):0.0;"),
    )
    .unwrap();
    (info, info_rerooted)
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn simple_dna_likelihood_reversibility(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let (info, info_rerooted) = &simple_dna_reroot_info();
    let model = DNASubstModel::new(model_type, params).unwrap();
    assert_relative_eq!(
        model.cost(info, false),
        model.cost(info_rerooted, true),
        epsilon = 1e-10,
    );
}

#[cfg(test)]
fn simple_protein_reroot_info() -> (PhyloInfo, PhyloInfo) {
    let sequences = Sequences::new(vec![
        record!("A", b"CTATATATACIJL"),
        record!("B", b"ATATATATAAIHL"),
        record!("C", b"TTATATATATIJL"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(
        sequences.clone(),
        tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
    )
    .unwrap();
    let info_rerooted = PhyloInfoBuilder::build_from_objects(
        sequences,
        tree_newick("(A:1.0,(B:2.0,C:3.0):1.0):0.0;"),
    )
    .unwrap();
    (info, info_rerooted)
}

#[rstest]
#[case::wag(WAG, &[], 1e-8)]
#[case::hivb(HIVB, &[], 1e-8)]
#[case::blosum(BLOSUM, &[], 1e-3)]
fn simple_protein_likelihood_reversibility(
    #[case] model_type: ProteinModelType,
    #[case] params: &[f64],
    #[case] epsilon: f64,
) {
    let (info, info_rerooted) = &simple_protein_reroot_info();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    assert_relative_eq!(
        model.cost(info, false),
        model.cost(info_rerooted, true),
        epsilon = epsilon,
    );
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn huelsenbeck_example_dna_reversibility_likelihood(
    #[case] model_type: DNAModelType,
    #[case] params: &[f64],
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let info_rerooted = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(model_type, params).unwrap();
    assert_relative_eq!(
        model.cost(info, false),
        model.cost(info_rerooted, true),
        epsilon = 1e-10,
    );
}

#[test]

fn pip_likelihood_correct_after_reset() {
    use crate::evolutionary_models::EvoModel;
    let tree1 =
        from_newick_string("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);").unwrap()[0].clone();
    let tree2 =
        from_newick_string("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);").unwrap()[0].clone();
    let sequences = Sequences::new(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);
    let info1 = PhyloInfoBuilder::build_from_objects(sequences.clone(), tree1).unwrap();
    let info2 = PhyloInfoBuilder::build_from_objects(sequences, tree2).unwrap();

    let pip_wag = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    let c1 = pip_wag.cost(&info1, false);
    assert_relative_eq!(c1, -1004.2260753055999, epsilon = 1e-5);
    assert_relative_eq!(c1, pip_wag.cost(&info1, true), epsilon = 1e-5);

    let c2 = pip_wag.cost(&info2, true);
    assert_ne!(c1, c2);
    assert_relative_eq!(c2, -1425.1290016747846, epsilon = 1e-5);
    assert_relative_eq!(c2, pip_wag.cost(&info2, true), epsilon = 1e-5);
}

#[rstest]
#[case::wag(WAG, -7.488595394504073, -10.206456536551775)]
#[case::hivb(HIVB, -7.231597482410509, -9.865559545434952)]
#[case::blosum(BLOSUM, -7.4408154253528975, -10.33187874481282)]

fn likelihood_correct_after_reset(
    #[case] model_type: ProteinModelType,
    #[case] llik1: f64,
    #[case] llik2: f64,
) {
    let tree1 =
        from_newick_string("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);").unwrap()[0].clone();
    let tree2 =
        from_newick_string("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);").unwrap()[0].clone();
    let sequences = Sequences::new(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);
    let info1 = PhyloInfoBuilder::build_from_objects(sequences.clone(), tree1).unwrap();
    let info2 = PhyloInfoBuilder::build_from_objects(sequences, tree2).unwrap();

    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let c1 = model.cost(&info1, false);
    assert_relative_eq!(c1, llik1, epsilon = 1e-5);
    assert_relative_eq!(c1, model.cost(&info1, true), epsilon = 1e-5);

    let c2 = model.cost(&info2, true);
    assert_ne!(c1, c2);
    assert_relative_eq!(c2, llik2, epsilon = 1e-5);
    assert_relative_eq!(c2, model.cost(&info2, true), epsilon = 1e-5);
}

#[test]
fn only_one_site_one_char() {
    // This used to fail on leaf data creation when some of the sequences were empty
    let sequences = Sequences::new(vec![
        record!("one", b"C"),
        record!("two", b"-"),
        record!("three", b"-"),
        record!("four", b"-"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    let gtr =
        DNASubstModel::new(GTR, &[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let logl = gtr.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[ignore = "long test"]
#[test]
fn hiv_large_dataset_subset_valid_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PhyloInfoBuilder::new(alignment).build().unwrap();
    let gtr =
        DNASubstModel::new(GTR, &[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let logl = gtr.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[ignore = "long test"]
#[test]
fn hiv_large_dataset_substitution_subset_pip() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PhyloInfoBuilder::new(alignment).build().unwrap();
    let pip = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let logl = pip.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn dna_gaps_against_phyml() {
    let newick =
        "(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);".to_string();
    let sequences = Sequences::new(
        read_sequences_from_file(&Path::new("./data/").join("sequences_DNA1.fasta")).unwrap(),
    );
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();

    // Compare against value from PhyML
    assert_relative_eq!(jc69.cost(&info, false), -9.70406054783923);
}

#[test]
fn dna_single_char_gaps_against_phyml() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let newick =
        "(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);".to_string();

    let sequences = Sequences::new(vec![
        record!("A", b"A"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.920437792326963); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"X"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), 0.0, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -1.38629, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"-"),
        record!("B", b"-"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.77259, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"-"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.92044, epsilon = 1e-5); // Compare against PhyML
}

#[test]
fn dna_ambig_chars_against_phyml() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let newick =
        "(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.0:0.08716381);".to_string();

    let sequences = Sequences::new(vec![
        record!("A", b"B"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -21.28936836, epsilon = 1e-7);
}

#[test]
fn dna_x_simple_fully_likely() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let newick = "(A:0.05,B:0.0005):0.0;".to_string();

    let sequences = Sequences::new(vec![record!("A", b"X"), record!("B", b"X")]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), 0.0, epsilon = 1e-5);
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn dna_x_fully_likely(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let model = DNASubstModel::new(model_type, params).unwrap();
    let tree =
        from_newick_string("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);").unwrap()[0].clone();
    let sequences = Sequences::new(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"X"),
    ]);

    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(model.cost(&info, true), 0.0, epsilon = 1e-5);
}

#[rstest]
#[case::wag(WAG)]
#[case::hivb(HIVB)]
#[case::blosum(BLOSUM)]
fn protein_x_fully_likely(#[case] model_type: ProteinModelType) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let tree =
        from_newick_string("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);").unwrap()[0].clone();
    let sequences = Sequences::with_alphabet(
        vec![
            record!("A", b"X"),
            record!("B", b"X"),
            record!("C", b"X"),
            record!("D", b"X"),
        ],
        protein_alphabet(),
    );

    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(model.cost(&info, true), 0.0, epsilon = 1e-5);
}
