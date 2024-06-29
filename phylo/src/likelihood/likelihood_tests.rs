use rstest::*;

use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::evolutionary_models::{
    DNAModelType::*,
    EvolutionaryModel, EvolutionaryModelInfo,
    ModelType::{self, *},
    ProteinModelType::*,
};
use crate::likelihood::LikelihoodCostFunction;
use crate::make_freqs;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::substitution_models::dna_models::{DNALikelihoodCost, DNASubstModel, DNASubstModelInfo};
use crate::substitution_models::protein_models::{ProteinLikelihoodCost, ProteinSubstModel};
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::tree::{tree_parser, NodeIdx::Leaf as L, Tree};

#[cfg(test)]
fn tree_newick(newick: &str) -> Tree {
    tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap()
}

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A0", None, b"A"),
        Record::with_attrs("B1", None, b"A"),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, L(0), L(1), blen_i, blen_j);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfo::from_sequences_tree(sequences, tree, &GapHandling::Ambiguous).unwrap()
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let model = DNASubstModel::new(DNA(JC69), &[]).unwrap();
    let likelihood = DNALikelihoodCost { info: &info };

    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        -2.5832498829317445,
        epsilon = 1e-6
    );
    let info = setup_simple_phylo_info(1.0, 2.0);
    let likelihood = DNALikelihoodCost { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        -2.719098272533848,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_simple_phylo_info_no_alignment() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAAA"),
        Record::with_attrs("B1", None, b"A"),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, L(0), L(1), 2.0, 1.4);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfo::from_sequences_tree(sequences, tree, &GapHandling::Ambiguous).unwrap()
}

#[test]
fn dna_likelihood_no_msa() {
    let info = setup_simple_phylo_info_no_alignment();
    let model = DNASubstModel::new(DNA(JC69), &[]).unwrap();
    let tmp = DNASubstModelInfo::new(&info, &model);
    assert!(tmp.is_err());
}

#[cfg(test)]
fn setup_phylo_info_single_leaf() -> PhyloInfo {
    let sequences = vec![Record::with_attrs("A0", None, b"AAAAAA")];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfo::from_sequences_tree(sequences, tree, &GapHandling::Ambiguous).unwrap()
}

#[test]
fn dna_likelihood_one_node() {
    let info = setup_phylo_info_single_leaf();
    let likelihood = DNALikelihoodCost { info: &info };
    let model = DNASubstModel::new(DNA(JC69), &[]).unwrap();
    assert!(LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model) < 0.0);
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("one", None, b"C"),
        Record::with_attrs("two", None, b"A"),
        Record::with_attrs("three", None, b"T"),
        Record::with_attrs("four", None, b"G"),
    ];
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
        .unwrap()
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let likelihood = DNALikelihoodCost { info: &info };
    let mut model = DNASubstModel::new(
        DNA(TN93),
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
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        -17.1035117087,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("one", None, b"T"),
        Record::with_attrs("two", None, b"C"),
        Record::with_attrs("three", None, b"A"),
        Record::with_attrs("four", None, b"C"),
        Record::with_attrs("five", None, b"C"),
    ];
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
        .unwrap()
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let likelihood = DNALikelihoodCost { info: &info };
    let model = DNASubstModel::new(DNA(K80), &[]).unwrap();
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        -7.581408,
        epsilon = 1e-6
    );
}

#[test]
fn dna_ambig_example_likelihood() {
    let model = DNASubstModel::new(
        DNA(TN93),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();

    let info_w_x = PhyloInfo::from_files(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood_w_x = DNALikelihoodCost { info: &info_w_x };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_w_x, &model),
        -94.46514304131543,
        epsilon = 1e-6
    );

    let info_w_n = PhyloInfo::from_files(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood_w_n = DNALikelihoodCost { info: &info_w_n };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_w_n, &model),
        -94.46514304131543,
        epsilon = 1e-6
    );

    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_w_x, &model),
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_w_n, &model),
    );
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let model = DNASubstModel::new(
        DNA(GTR),
        &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };

    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        -216.234734,
        epsilon = 1e-3
    );
}

#[rstest]
#[case::wag(Protein(WAG), &[], -4505.736814460457, 1e-4)]
#[case::hivb(Protein(HIVB), &[], -4407.989226397638, 1e-5)]
#[case::blosum(Protein(BLOSUM), &[], -4576.40850634098, 1e-5)] // PhyML likelihood under BLOSUM62 is -4587.71053
fn protein_example_likelihood(
    #[case] model_type: ModelType,
    #[case] params: &[f64],
    #[case] expected_llik: f64,
    #[case] epsilon: f64,
) {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/phyml_protein_nogap_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    let likelihood = ProteinLikelihoodCost { info: &info };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
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
        PhyloInfo::from_sequences_tree(
            sequences.clone(),
            tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
            &GapHandling::Ambiguous,
        )
        .unwrap(),
    );
    res.push(
        PhyloInfo::from_sequences_tree(
            sequences,
            tree_newick("(A:1.0,(B:2.0,C:3.0):1.0):0.0;"),
            &GapHandling::Ambiguous,
        )
        .unwrap(),
    );
    res
}

#[rstest]
#[case::jc69(DNA(JC69), &[])]
#[case::k80(DNA(K80), &[])]
#[case::hky(DNA(HKY), &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(DNA(TN93), &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(DNA(GTR), &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn simple_dna_likelihood_reversibility(#[case] model_type: ModelType, #[case] params: &[f64]) {
    let info = setup_simple_reversibility();
    let model = DNASubstModel::new(model_type, params).unwrap();
    let likelihood = DNALikelihoodCost { info: &info[0] };
    let likelihood_rerooted = DNALikelihoodCost { info: &info[1] };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_rerooted, &model),
        epsilon = 1e-10,
    );
}

#[cfg(test)]
fn setup_simple_protein_reversibility() -> Vec<PhyloInfo> {
    let mut res = Vec::<PhyloInfo>::new();
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATATACIJL"),
        Record::with_attrs("B", None, b"ATATATATAAIHL"),
        Record::with_attrs("C", None, b"TTATATATATIJL"),
    ];
    res.push(
        PhyloInfo::from_sequences_tree(
            sequences.clone(),
            tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
            &GapHandling::Ambiguous,
        )
        .unwrap(),
    );
    res.push(
        PhyloInfo::from_sequences_tree(
            sequences,
            tree_newick("(A:1.0,(B:2.0,C:3.0):1.0):0.0;"),
            &GapHandling::Ambiguous,
        )
        .unwrap(),
    );
    res
}

#[rstest]
#[case::wag(Protein(WAG), &[], 1e-8)]
#[case::hivb(Protein(HIVB), &[], 1e-8)]
#[case::blosum(Protein(BLOSUM), &[], 1e-3)]
fn simple_protein_likelihood_reversibility(
    #[case] model_type: ModelType,
    #[case] params: &[f64],
    #[case] epsilon: f64,
) {
    let info = setup_simple_protein_reversibility();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    let likelihood = ProteinLikelihoodCost { info: &info[0] };
    let likelihood_rerooted = ProteinLikelihoodCost { info: &info[1] };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_rerooted, &model),
        epsilon = epsilon,
    );
}

#[rstest]
#[case::jc69(DNA(JC69), &[])]
#[case::k80(DNA(K80), &[])]
#[case::hky(DNA(HKY), &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(DNA(TN93), &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(DNA(GTR), &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn huelsenbeck_example_dna_reversibility_likelihood(
    #[case] model_type: ModelType,
    #[case] params: &[f64],
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info1 = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let info2 = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let model = DNASubstModel::new(model_type, params).unwrap();
    let likelihood = DNALikelihoodCost { info: &info1 };
    let likelihood_rerooted = DNALikelihoodCost { info: &info2 };
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model),
        LikelihoodCostFunction::compute_log_likelihood(&likelihood_rerooted, &model),
        epsilon = 1e-10,
    );
}

#[test]
fn empirical_frequencies_no_ambigs() {
    let sequences = vec![
        Record::with_attrs("one", None, b"CCCCCCCC"),
        Record::with_attrs("two", None, b"AAAAAAAA"),
        Record::with_attrs("three", None, b"TTTTTTTT"),
        Record::with_attrs("four", None, b"GGGGGGGG"),
    ];
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();
    assert_relative_eq!(freqs, make_freqs!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_x_or_n() {
    let sequences = vec![
        Record::with_attrs("on", None, b"XXXXXXXX"),
        Record::with_attrs("tw", None, b"XXXXXXXX"),
        Record::with_attrs("th", None, b"NNNNNNNN"),
        Record::with_attrs("fo", None, b"NNNNNNNN"),
    ];
    let newick = "((on:2,tw:2):1,(th:1,fo:1):2);".to_string();
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();
    assert_relative_eq!(freqs, make_freqs!(&[0.25; 4]), epsilon = 1e-6);
    let sequences = vec![
        Record::with_attrs("on", None, b"AAAAAAAAAA"),
        Record::with_attrs("tw", None, b"XXXXXXXXXX"),
        Record::with_attrs("th", None, b"CCCCCCCCCC"),
        Record::with_attrs("fo", None, b"NNNNNNNNNN"),
    ];
    let newick = "(((on:2,tw:2):1,th:1):4,fo:1);".to_string();
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();
    assert_relative_eq!(
        freqs,
        make_freqs!(&[0.125, 0.375, 0.375, 0.125]),
        epsilon = 1e-6
    );
}

#[test]
fn empirical_frequencies_ambig() {
    let sequences = vec![
        Record::with_attrs("A", None, b"VVVVVVVVV"),
        Record::with_attrs("B", None, b"TTT"),
    ];
    let newick = "(A:2,B:2):1.0;".to_string();
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();
    assert_relative_eq!(freqs, make_freqs!(&[0.25; 4]), epsilon = 1e-6);
    let sequences = vec![
        Record::with_attrs("A", None, b"SSSSSSSSSSSSSSSSSSSS"),
        Record::with_attrs("B", None, b"WWWWWWWWWWWWWWWWWWWW"),
    ];
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();
    assert_relative_eq!(freqs, make_freqs!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_no_aas() {
    let sequences = vec![Record::with_attrs("A", None, b"BBBBBBBBB")];
    let newick = "A:1.0;".to_string();
    let info =
        PhyloInfo::from_sequences_tree(sequences, tree_newick(&newick), &GapHandling::Ambiguous)
            .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let freqs = likelihood.get_empirical_frequencies();

    assert_relative_eq!(
        freqs,
        make_freqs!(&[3.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0, 3.0 / 10.0]),
        epsilon = 1e-6
    );
}
