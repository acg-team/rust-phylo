use std::{fmt::Debug, fmt::Display, path::PathBuf};

use approx::assert_relative_eq;
use assert_matches::assert_matches;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::frequencies;
use crate::io::{read_sequences_from_file, DataError};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::substitution_models::FreqVector;
use crate::tree::tree_parser::{self, ParsingError};
use crate::tree::NodeIdx::{Internal as I, Leaf as L};
use crate::tree::Tree;

#[test]
fn empirical_frequencies_easy() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"AAAAA"),
        Record::with_attrs("B", None, b"CCCCC"),
        Record::with_attrs("C", None, b"GGGGG"),
        Record::with_attrs("D", None, b"TTTTT"),
    ]);
    let tree = tree_parser::from_newick_string("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);")
        .unwrap()
        .pop()
        .unwrap();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let freqs = info.freqs();
    assert_eq!(freqs, frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn empirical_frequencies() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"TT"),
        Record::with_attrs("B", None, b"CA"),
        Record::with_attrs("C", None, b"NN"),
        Record::with_attrs("D", None, b"NN"),
    ]);
    let tree = tree_parser::from_newick_string("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);")
        .unwrap()
        .pop()
        .unwrap();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let freqs = info.freqs();
    assert_eq!(
        freqs,
        frequencies!(&[3.0 / 8.0, 2.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0])
    );
    assert_eq!(freqs.sum(), 1.0);
}

#[cfg(test)]
fn tree_newick(newick: &str) -> Tree {
    tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap()
}

#[test]
fn setup_info_correct_unaligned() {
    let res_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(res_info.is_err());
}

#[test]
fn setup_info_mismatched_ids_missing_tips() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    let error_msg = downcast_error::<DataError>(&info).to_string();
    assert!(error_msg.contains("tree tip IDs: [\"C\", \"D\"]"));
}

#[test]
fn setup_info_mismatched_ids_missing_sequences() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_3.newick"),
    )
    .build();
    let error_msg = downcast_error::<DataError>(&info).to_string();
    assert!(error_msg.contains("sequence IDs: [\"E\", \"F\"]"));
}

#[test]
fn setup_info_missing_sequence_file() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA_nonexistent.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert_matches!(
        info.unwrap_err().to_string().as_str(),
        "Failed to read fasta from \"./data/sequences_DNA_nonexistent.fasta\""
    );
}

#[test]
fn setup_info_empty_sequence_file() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_empty.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No sequences provided, aborting."
    );
}

#[test]
fn setup_info_empty_tree_file() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_empty.newick"),
    )
    .build();
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No trees in the tree file, aborting."
    );
}

#[test]
fn setup_info_malformed_tree_file() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_malformed.newick"),
    )
    .build();
    assert!(downcast_error::<ParsingError>(&info)
        .to_string()
        .contains("Malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let res_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA1.fasta"),
        PathBuf::from("./data/tree_multiple.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(res_info.tree.leaves().len(), 4);
    assert_eq!(res_info.msa.len(), 4);
}

fn downcast_error<T: Display + Debug + Send + Sync + 'static>(
    result: &Result<PhyloInfo, anyhow::Error>,
) -> &T {
    (result.as_ref().unwrap_err()).downcast_ref::<T>().unwrap()
}

#[test]
fn setup_unaligned() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(info.is_err());
}

#[test]
fn setup_aligned_msa() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_DNA1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    assert!(!info.msa.is_empty());
    info.msa.seqs.iter().for_each(|rec| {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    });
    let mut sequences =
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap();
    sequences.sort();
    let mut aligned_sequences = info.msa.compile(None, &info.tree).unwrap();
    aligned_sequences.sort();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn test_aligned_check() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ]);
    let tree = tree_newick("((A0:2.0,B1:2.0):1.0,(C2:2.0,(D3:1.0,E4:2.5):3.0):2.0):0.0;");
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree.clone());
    assert!(info.is_err());
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A----"),
        Record::with_attrs("C2", None, b"AABCD"),
        Record::with_attrs("D3", None, b"AAAAA"),
        Record::with_attrs("E4", None, b"AAATT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree);
    assert!(info.is_ok());
}

#[test]
fn check_phyloinfo_creation_newick_msa() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(
        sequences,
        tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
    );
    assert!(info.is_ok());
    assert!(!info.unwrap().msa.is_empty());
}

#[test]
#[should_panic]
fn check_phyloinfo_creation_tree_no_msa() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATAAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ]);
    PhyloInfoBuilder::build_from_objects(sequences, tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"))
        .unwrap();
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = PhyloInfoBuilder::build_from_objects(
        Sequences::new(vec![]),
        tree_parser::from_newick_string("((A:2.0,B:2.0):1.0,C:2.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
    );
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("D", None, b"CTATATAAC"),
        Record::with_attrs("E", None, b"ATATATATAA"),
        Record::with_attrs("F", None, b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(
        sequences,
        tree_parser::from_newick_string("((A:2.0,B:2.0):1.0,C:2.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
    );
    assert!(info.is_err());
}

#[cfg(test)]
fn make_test_tree() -> Tree {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b""),
        Record::with_attrs("B", None, b""),
        Record::with_attrs("C", None, b""),
        Record::with_attrs("D", None, b""),
    ]);
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(4, &L(0), &L(1), 2.0, 2.0);
    tree.add_parent(5, &I(4), &L(2), 1.0, 2.0);
    tree.add_parent(6, &I(5), &L(3), 1.0, 2.0);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    tree
}

#[test]
fn check_phyloinfo_creation_tree_correct_no_msa() {
    let info = PhyloInfoBuilder::build_from_objects(
        Sequences::new(vec![
            Record::with_attrs("A", None, b"AAAAA"),
            Record::with_attrs("B", None, b"A"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_tree_correct_msa() {
    let info = PhyloInfoBuilder::build_from_objects(
        Sequences::new(vec![
            Record::with_attrs("A", None, b"AA"),
            Record::with_attrs("B", None, b"A-"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A-"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_ok());
    assert!(!info.unwrap().msa.is_empty());
}

#[test]
fn check_phyloinfo_creation_tree_mismatch_ids() {
    let info = PhyloInfoBuilder::build_from_objects(
        Sequences::new(vec![
            Record::with_attrs("D", None, b"CTATATAAC"),
            Record::with_attrs("E", None, b"ATATATATAA"),
            Record::with_attrs("F", None, b"TTATATATAT"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_err());
}

#[test]
fn check_empirical_frequencies() {
    let info = PhyloInfoBuilder::build_from_objects(
        Sequences::new(vec![
            Record::with_attrs("A", None, b"AAAAC"),
            Record::with_attrs("B", None, b"TTTCC"),
            Record::with_attrs("C", None, b"GGGCC"),
            Record::with_attrs("D", None, b"TTAAA"),
        ]),
        make_test_tree(),
    )
    .unwrap();
    let freqs = info.freqs();
    assert_eq!(freqs.clone().sum(), 1.0);
    assert_eq!(freqs, frequencies!(&[5.0, 5.0, 7.0, 3.0]).scale(1.0 / 20.0));
}

#[test]
fn empirical_frequencies_no_ambigs() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("one", None, b"CCCCCCCC"),
        Record::with_attrs("two", None, b"AAAAAAAA"),
        Record::with_attrs("three", None, b"TTTTTTTT"),
        Record::with_attrs("four", None, b"GGGGGGGG"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_x() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("on", None, b"XXXXXXXX"),
        Record::with_attrs("tw", None, b"XXXXXXXX"),
        Record::with_attrs("th", None, b"NNNNNNNN"),
        Record::with_attrs("fo", None, b"NNNNNNNN"),
    ]);
    let newick = "((on:2,tw:2):1,(th:1,fo:1):2);".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_n() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("on", None, b"AAAAAAAAAA"),
        Record::with_attrs("tw", None, b"XXXXXXXXXX"),
        Record::with_attrs("th", None, b"CCCCCCCCCC"),
        Record::with_attrs("fo", None, b"NNNNNNNNNN"),
    ]);
    let newick = "(((on:2,tw:2):1,th:1):4,fo:1);".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[0.125, 0.375, 0.375, 0.125]),
        epsilon = 1e-6
    );
}

#[test]
fn empirical_frequencies_ambig_other() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"VVVVVVVVVV"),
        Record::with_attrs("B", None, b"TTTTVVVTVV"),
    ]);
    let newick = "(A:2,B:2):1.0;".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"SSSSSSSSSSSSSSSSSSSS"),
        Record::with_attrs("B", None, b"WWWWWWWWWWWWWWWWWWWW"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_no_aas() {
    let sequences = Sequences::new(vec![Record::with_attrs("A", None, b"BBBBBBBBB")]);
    let newick = "A:1.0;".to_string();
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree_newick(&newick)).unwrap();
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[3.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0, 3.0 / 10.0]),
        epsilon = 1e-6
    );
}
