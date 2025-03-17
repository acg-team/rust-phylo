use std::{fmt::Debug, fmt::Display, path::PathBuf};

use approx::assert_relative_eq;
use assert_matches::assert_matches;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::frequencies;
use crate::io::{read_sequences_from_file, DataError};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::substitution_models::FreqVector;
use crate::tree::{
    tree_parser::{from_newick, ParsingError},
    NodeIdx::{Internal as I, Leaf as L},
    Tree,
};

use crate::{record_wo_desc as record, tree};

#[cfg(test)]
fn downcast_error<T: Display + Debug + Send + Sync + 'static>(
    result: &Result<PhyloInfo, anyhow::Error>,
) -> &T {
    (result.as_ref().unwrap_err()).downcast_ref::<T>().unwrap()
}

#[test]
fn empirical_frequencies_easy() {
    let sequences = Sequences::new(vec![
        record!("A", b"AAAAA"),
        record!("B", b"CCCCC"),
        record!("C", b"GGGGG"),
        record!("D", b"TTTTT"),
    ]);
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    let freqs = info.freqs();
    assert_eq!(freqs, frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn empirical_frequencies() {
    let sequences = Sequences::new(vec![
        record!("A", b"TT"),
        record!("B", b"CA"),
        record!("C", b"NN"),
        record!("D", b"NN"),
    ]);
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    let freqs = info.freqs();
    assert_eq!(
        freqs,
        frequencies!(&[3.0 / 8.0, 2.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0])
    );
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn setup_info_correct_unaligned() {
    let fldr = PathBuf::from("./data");
    let res_info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(res_info.is_ok());
}

#[test]
fn setup_info_mismatched_ids_missing_tips() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_1.newick"),
    )
    .build();
    let error_msg = downcast_error::<DataError>(&info).to_string();
    assert!(error_msg.contains("tree tip IDs: [\"C\", \"D\"]"));
}

#[test]
fn setup_info_mismatched_ids_missing_sequences() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_3.newick"),
    )
    .build();
    let error_msg = downcast_error::<DataError>(&info).to_string();
    assert!(error_msg.contains("sequence IDs: [\"E\", \"F\"]"));
}

#[test]
fn setup_info_missing_sequence_file() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA_nonexistent.fasta"),
        fldr.join("tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert_matches!(
        info.unwrap_err().to_string().as_str(),
        "Failed to read fasta from \"./data/sequences_DNA_nonexistent.fasta\""
    );
}

#[test]
fn setup_info_empty_sequence_file() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_empty.fasta"),
        fldr.join("tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No sequences provided, aborting."
    );
}

#[test]
fn setup_info_empty_tree_file() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_empty.newick"),
    )
    .build();
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No trees in the tree file, aborting."
    );
}

#[test]
fn setup_info_malformed_tree_file() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_malformed.newick"),
    )
    .build();
    assert!(downcast_error::<ParsingError>(&info)
        .to_string()
        .contains("Malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let fldr = PathBuf::from("./data");
    let res_info = PIB::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_multiple.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(res_info.tree.leaves().len(), 4);
    assert_eq!(res_info.msa.seq_count(), 4);
}

#[test]
fn setup_unaligned() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(info.is_ok());
}

#[test]
fn setup_aligned_msa() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(info.msa.len(), 5);
    info.msa.seqs.iter().for_each(|rec| {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    });
    let sequences = Sequences::new(
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap(),
    );
    let aligned_sequences = info.msa.compile(None, &info.tree).unwrap();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn correct_setup_when_sequences_empty() {
    let fldr = PathBuf::from("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_some_empty.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(info.msa.len(), 1);
    info.msa.seqs.iter().for_each(|rec| {
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    });
    let sequences =
        Sequences::new(read_sequences_from_file(&fldr.join("sequences_some_empty.fasta")).unwrap());
    let aligned_sequences = info.msa.compile(None, &info.tree).unwrap();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn unaligned_setup() {
    let sequences = Sequences::new(vec![
        record!("A0", b"AAAAA"),
        record!("B1", b"A"),
        record!("C2", b"AA"),
        record!("D3", b"A"),
        record!("E4", b"AAA"),
    ]);
    let tree = tree!("((A0:2.0,B1:2.0):1.0,(C2:2.0,(D3:1.0,E4:2.5):3.0):2.0):0.0;");
    let info = PIB::build_from_objects(sequences, tree.clone());
    assert!(info.is_ok());
    let info = info.unwrap();
    println!("{}", info.compile_alignment(None).unwrap());
    assert!(info.msa.len() >= 5);
    assert_eq!(info.msa.seq_count(), 5);
}

#[test]
fn aligned_setup() {
    let sequences = Sequences::new(vec![
        record!("A0", b"AAAAA"),
        record!("B1", b"A----"),
        record!("C2", b"AABCD"),
        record!("D3", b"AAAAA"),
        record!("E4", b"AAATT"),
    ]);
    let tree = tree!("((A0:2.0,B1:2.0):1.0,(C2:2.0,(D3:1.0,E4:2.5):3.0):2.0):0.0;");
    let info = PIB::build_from_objects(sequences, tree);
    assert!(info.is_ok());
}

#[test]
fn check_phyloinfo_creation_newick_msa() {
    let sequences = Sequences::new(vec![
        record!("A", b"CTATATATAC"),
        record!("B", b"ATATATATAA"),
        record!("C", b"TTATATATAT"),
    ]);
    let info = PIB::build_from_objects(sequences, tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;"));
    assert!(info.is_ok());
    assert_eq!(info.unwrap().msa.len(), 10);
}

#[test]
fn check_phyloinfo_creation_tree_no_msa() {
    let sequences = Sequences::new(vec![
        record!("A", b"CTATATAAC"),
        record!("B", b"ATATATATAA"),
        record!("C", b"TTATATATAT"),
    ]);
    assert!(PIB::build_from_objects(sequences, tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;")).is_ok());
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = PIB::build_from_objects(
        Sequences::new(vec![]),
        tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
    );
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let sequences = Sequences::new(vec![
        record!("D", b"CTATATAAC"),
        record!("E", b"ATATATATAA"),
        record!("F", b"TTATATATAT"),
    ]);
    let info = PIB::build_from_objects(sequences, tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;"));
    assert!(info.is_err());
}

#[cfg(test)]
fn make_test_tree() -> Tree {
    let sequences = Sequences::new(vec![
        record!("A", b""),
        record!("B", b""),
        record!("C", b""),
        record!("D", b""),
    ]);
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(4, &L(0), &L(1), 2.0, 2.0);
    tree.add_parent(5, &I(4), &L(2), 1.0, 2.0);
    tree.add_parent(6, &I(5), &L(3), 1.0, 2.0);
    tree.complete = true;
    tree.compute_postorder();
    tree.compute_preorder();
    tree
}

#[test]
fn check_phyloinfo_creation_tree_correct_no_msa() {
    let info = PIB::build_from_objects(
        Sequences::new(vec![
            record!("A", b"AAAAA"),
            record!("B", b"A"),
            record!("C", b"AA"),
            record!("D", b"A"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_ok());
    let info = info.unwrap();
    assert_eq!(info.msa.len(), 5);
    assert_eq!(info.msa.seq_count(), 4);
}

#[test]
fn check_phyloinfo_creation_tree_correct_msa() {
    let info = PIB::build_from_objects(
        Sequences::new(vec![
            record!("A", b"AA"),
            record!("B", b"A-"),
            record!("C", b"AA"),
            record!("D", b"A-"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_ok());
    assert_eq!(info.unwrap().msa.len(), 2);
}

#[test]
fn check_phyloinfo_creation_tree_mismatch_ids() {
    let info = PIB::build_from_objects(
        Sequences::new(vec![
            record!("D", b"CTATATAAC"),
            record!("E", b"ATATATATAA"),
            record!("F", b"TTATATATAT"),
        ]),
        make_test_tree(),
    );
    assert!(info.is_err());
}

#[test]
fn check_empirical_frequencies() {
    let info = PIB::build_from_objects(
        Sequences::new(vec![
            record!("A", b"AAAAC"),
            record!("B", b"TTTCC"),
            record!("C", b"GGGCC"),
            record!("D", b"TTAAA"),
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
        record!("one", b"CCCCCCCC"),
        record!("two", b"AAAAAAAA"),
        record!("three", b"TTTTTTTT"),
        record!("four", b"GGGGGGGG"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_x() {
    let sequences = Sequences::new(vec![
        record!("on", b"XXXXXXXX"),
        record!("tw", b"XXXXXXXX"),
        record!("th", b"NNNNNNNN"),
        record!("fo", b"NNNNNNNN"),
    ]);
    let newick = "((on:2,tw:2):1,(th:1,fo:1):2);".to_string();
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_n() {
    let sequences = Sequences::new(vec![
        record!("on", b"AAAAAAAAAA"),
        record!("tw", b"XXXXXXXXXX"),
        record!("th", b"CCCCCCCCCC"),
        record!("fo", b"NNNNNNNNNN"),
    ]);
    let newick = "(((on:2,tw:2):1,th:1):4,fo:1);".to_string();
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[0.125, 0.375, 0.375, 0.125]),
        epsilon = 1e-6
    );
}

#[test]
fn empirical_frequencies_ambig_other() {
    let sequences = Sequences::new(vec![
        record!("A", b"VVVVVVVVVV"),
        record!("B", b"TTTTVVVTVV"),
    ]);
    let newick = "(A:2,B:2):1.0;".to_string();
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
    let sequences = Sequences::new(vec![
        record!("A", b"SSSSSSSSSSSSSSSSSSSS"),
        record!("B", b"WWWWWWWWWWWWWWWWWWWW"),
    ]);
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_no_aas() {
    let sequences = Sequences::new(vec![record!("A", b"BBBBBBBBB")]);
    let newick = "A:1.0;".to_string();
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[3.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0, 3.0 / 10.0]),
        epsilon = 1e-6
    );
}
