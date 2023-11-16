use rstest::*;

use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::likelihood::{setup_dna_likelihood, setup_protein_likelihood, LikelihoodCostFunction};
use crate::phylo_info::{
    phyloinfo_from_files, phyloinfo_from_sequences_newick, phyloinfo_from_sequences_tree, PhyloInfo,
};
use crate::tree::{NodeIdx::Leaf as L, Tree};

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A0", None, b"A"),
        Record::with_attrs("B1", None, b"A"),
    ];
    let mut tree = Tree::new(&sequences);
    tree.add_parent(0, L(0), L(1), blen_i, blen_j);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    phyloinfo_from_sequences_tree(&sequences, tree).unwrap()
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69", &[], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.5832498829317445,
        epsilon = 1e-6
    );
    let info = setup_simple_phylo_info(1.0, 2.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69", &[], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.719098272533848,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"C"),
        Record::with_attrs("two", None, b"A"),
        Record::with_attrs("three", None, b"T"),
        Record::with_attrs("four", None, b"G"),
    ];
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    phyloinfo_from_sequences_tree(&sequences, tree).unwrap()
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(
        &info,
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -17.1035117087,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"T"),
        Record::with_attrs("two", None, b"C"),
        Record::with_attrs("three", None, b"A"),
        Record::with_attrs("four", None, b"C"),
        Record::with_attrs("five", None, b"C"),
    ];
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    phyloinfo_from_sequences_tree(&sequences, tree).unwrap()
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(&info, "k80", &[], true).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -7.581408,
        epsilon = 1e-6
    );
}

#[test]
fn dna_ambig_example_likelihood() {
    let info_w_x = phyloinfo_from_files(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .unwrap();
    let mut likelihood_w_x = setup_dna_likelihood(
        &info_w_x,
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood_w_x.compute_log_likelihood(),
        -90.1367231323,
        epsilon = 1e-6
    );
    let info_w_n = phyloinfo_from_files(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .unwrap();
    let mut likelihood_w_n = setup_dna_likelihood(
        &info_w_n,
        "tn93",
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood_w_n.compute_log_likelihood(),
        -90.1367231323,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        likelihood_w_x.compute_log_likelihood(),
        likelihood_w_n.compute_log_likelihood()
    );
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let mut likelihood = setup_dna_likelihood(
        &info,
        "gtr",
        &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        true,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -216.234734,
        epsilon = 1e-3
    );
}

#[rstest]
#[case::wag("wag", &[], -4505.736814460457, 1e-4)]
#[case::hivb("hivb", &[], -4407.989226397638, 1e-5)]
#[case::blosum("blosum", &[], -4576.40850634098, 1e-5)] // PhyML likelihood under BLOSUM62 is -4587.71053
fn protein_example_likelihood(
    #[case] model_name: &str,
    #[case] model_params: &[f64],
    #[case] expected_llik: f64,
    #[case] epsilon: f64,
) {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/phyml_protein_nogap_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
    )
    .unwrap();
    let mut likelihood = setup_protein_likelihood(&info, model_name, model_params, true).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        expected_llik,
        epsilon = epsilon
    );
}

#[cfg(test)]
fn setup_simple_reversibility() -> Vec<PhyloInfo> {
    let mut res = Vec::<PhyloInfo>::new();
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    res.push(
        phyloinfo_from_sequences_newick(&sequences, "((A:2.0,B:2.0):1.0,C:2.0):0.0;").unwrap(),
    );
    res.push(
        phyloinfo_from_sequences_newick(&sequences, "(A:1.0,(B:2.0,C:3.0):1.0):0.0;").unwrap(),
    );
    res
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn simple_dna_likelihood_reversibility(#[case] model_name: &str, #[case] model_params: &[f64]) {
    let info = setup_simple_reversibility();
    let mut likelihood1 = setup_dna_likelihood(&info[0], model_name, model_params, false).unwrap();
    let mut likelihood2 = setup_dna_likelihood(&info[1], model_name, model_params, false).unwrap();
    assert_relative_eq!(
        likelihood1.compute_log_likelihood(),
        likelihood2.compute_log_likelihood(),
        epsilon = 1e-10,
    );
}

#[rstest]
#[case::wag("wag", &[], 1e-8)]
#[case::hivb("hivb", &[], 1e-8)]
#[case::blosum("blosum", &[], 1e-3)]
fn simple_protein_likelihood_reversibility(
    #[case] model_name: &str,
    #[case] model_params: &[f64],
    #[case] epsilon: f64,
) {
    let info = setup_simple_reversibility();
    let mut likelihood1 =
        setup_protein_likelihood(&info[0], model_name, model_params, true).unwrap();
    let mut likelihood2 =
        setup_protein_likelihood(&info[1], model_name, model_params, true).unwrap();
    assert_relative_eq!(
        likelihood1.compute_log_likelihood(),
        likelihood2.compute_log_likelihood(),
        epsilon = epsilon,
    );
}

#[rstest]
#[case::jc69("jc69", &[])]
#[case::k80("k80", &[])]
#[case::hky("hky", &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93("tn93", &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr("gtr", &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn huelsenbeck_example_dna_reversibility_likelihood(
    #[case] model_name: &str,
    #[case] model_params: &[f64],
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info1 = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let info2 = phyloinfo_from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .unwrap();
    let mut likelihood = setup_dna_likelihood(&info1, model_name, model_params, true).unwrap();
    let mut reroot_likelihood =
        setup_dna_likelihood(&info2, model_name, model_params, true).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        reroot_likelihood.compute_log_likelihood(),
        epsilon = 1e-10,
    );
}
