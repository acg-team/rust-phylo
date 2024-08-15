use rstest::*;

use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvolutionaryModel,
    ProteinModelType::{self, *},
};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::substitution_models::dna_models::DNASubstModel;
use crate::substitution_models::protein_models::{ProteinLikelihoodCost, ProteinSubstModel};
use crate::substitution_models::{SubstMatrix, SubstitutionLikelihoodCost};
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"A"),
        Record::with_attrs("B1", None, b"A"),
    ]);
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, &L(0), &L(1), blen_i, blen_j);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);

    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        -2.5832498829317445,
        epsilon = 1e-6
    );
    let info = setup_simple_phylo_info(1.0, 2.0);
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);

    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        -2.719098272533848,
        epsilon = 1e-6
    );
}

#[test]
fn gaps_as_ambigs() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("one", None, b"CCCCCCXX"),
        Record::with_attrs("two", None, b"XXAAAAAA"),
        Record::with_attrs("three", None, b"TTTNNTTT"),
        Record::with_attrs("four", None, b"GNGGGGNG"),
    ]);
    let tree = tree_newick("((one:2,two:2):1,(three:1,four:1):2);");
    let info_ambig = PhyloInfoBuilder::build_from_objects(sequences, tree.clone()).unwrap();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let likelihood_ambig = SubstitutionLikelihoodCost::new(&info_ambig, &model)
        .logl()
        .0;
    let sequences = Sequences::new(vec![
        Record::with_attrs("one", None, b"CCCCCC--"),
        Record::with_attrs("two", None, b"--AAAAAA"),
        Record::with_attrs("three", None, b"TTT--TTT"),
        Record::with_attrs("four", None, b"G-GGGG-G"),
    ]);
    let tree = tree_newick("((one:2,two:2):1,(three:1,four:1):2);");
    let info_gaps = PhyloInfoBuilder::build_from_objects(sequences, tree.clone()).unwrap();
    let likelihood_gaps = SubstitutionLikelihoodCost::new(&info_gaps, &model).logl().0;
    assert_eq!(likelihood_ambig, likelihood_gaps);
}

#[cfg(test)]
fn setup_phylo_info_single_leaf() -> PhyloInfo {
    let sequences = Sequences::new(vec![Record::with_attrs("A0", None, b"AAAAAA")]);
    let mut tree = Tree::new(&sequences).unwrap();
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_likelihood_one_node() {
    let info = setup_phylo_info_single_leaf();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);
    assert!(LikelihoodCostFunction::logl(&likelihood) < 0.0);
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        Record::with_attrs("one", None, b"C"),
        Record::with_attrs("two", None, b"A"),
        Record::with_attrs("three", None, b"T"),
        Record::with_attrs("four", None, b"G"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap()
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
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
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        -17.1035117087,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        Record::with_attrs("one", None, b"T"),
        Record::with_attrs("two", None, b"C"),
        Record::with_attrs("three", None, b"A"),
        Record::with_attrs("four", None, b"C"),
        Record::with_attrs("five", None, b"C"),
    ]);
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap()
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let model = DNASubstModel::new(K80, &[]).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);

    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        -7.581408,
        epsilon = 1e-6
    );
}

#[test]
fn dna_ambig_example_likelihood() {
    let model = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();

    let info_w_x = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let likelihood_w_x = SubstitutionLikelihoodCost::new(&info_w_x, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood_w_x),
        -94.46514304131543,
        epsilon = 1e-6
    );

    let info_w_n = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let likelihood_w_n = SubstitutionLikelihoodCost::new(&info_w_n, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood_w_n),
        -94.46514304131543,
        epsilon = 1e-6
    );

    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood_w_x),
        LikelihoodCostFunction::logl(&likelihood_w_n),
    );
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let model =
        DNASubstModel::new(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0]).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);

    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        -216.234734,
        epsilon = 1e-3
    );
}

#[rstest]
#[case::wag(WAG, &[], -4505.736814460457, 1e-4)]
#[case::hivb(HIVB, &[], -4407.989226397638, 1e-5)]
#[case::blosum(BLOSUM, &[], -4576.40850634098, 1e-5)] // PhyML likelihood under BLOSUM62 is -4587.71053
fn protein_example_likelihood(
    #[case] model_type: ProteinModelType,
    #[case] params: &[f64],
    #[case] expected_llik: f64,
    #[case] epsilon: f64,
) {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_nogap_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    let likelihood = ProteinLikelihoodCost::new(&info, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        expected_llik,
        epsilon = epsilon
    );
}

#[cfg(test)]
fn simple_dna_reroot_info() -> (PhyloInfo, PhyloInfo) {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
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
    let (info, info_rerooted) = simple_dna_reroot_info();
    let model = DNASubstModel::new(model_type, params).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);
    let likelihood_rerooted = SubstitutionLikelihoodCost::new(&info_rerooted, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        LikelihoodCostFunction::logl(&likelihood_rerooted),
        epsilon = 1e-10,
    );
}

#[cfg(test)]
fn simple_protein_reroot_info() -> (PhyloInfo, PhyloInfo) {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATACIJL"),
        Record::with_attrs("B", None, b"ATATATATAAIHL"),
        Record::with_attrs("C", None, b"TTATATATATIJL"),
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
    let (info, info_rerooted) = simple_protein_reroot_info();
    let model = ProteinSubstModel::new(model_type, params).unwrap();
    let likelihood = ProteinLikelihoodCost::new(&info, &model);
    let likelihood_rerooted = ProteinLikelihoodCost::new(&info_rerooted, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        LikelihoodCostFunction::logl(&likelihood_rerooted),
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
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let info_rerooted = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(model_type, params).unwrap();
    let likelihood = SubstitutionLikelihoodCost::new(&info, &model);
    let likelihood_rerooted = SubstitutionLikelihoodCost::new(&info_rerooted, &model);
    assert_relative_eq!(
        LikelihoodCostFunction::logl(&likelihood),
        LikelihoodCostFunction::logl(&likelihood_rerooted),
        epsilon = 1e-10,
    );
}
