use std::fmt::{Debug, Display};
use std::path::PathBuf;

use approx::assert_relative_eq;
use assert_matches::assert_matches;

use crate::alignment::{Alignment, Sequences};
use crate::io::{read_sequences, DataError};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::substitution_models::FreqVector;
use crate::tree::tree_parser::ParsingError;
use crate::{frequencies, record_wo_desc as record, tree};

#[cfg(test)]
fn downcast_error<T: Display + Debug + Send + Sync + 'static>(
    result: &Result<PhyloInfo, anyhow::Error>,
) -> &T {
    (result.as_ref().unwrap_err()).downcast_ref::<T>().unwrap()
}

#[test]
fn empirical_frequencies_easy() {
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"AAAAA"),
            record!("B", b"CCCCC"),
            record!("C", b"GGGGG"),
            record!("D", b"TTTTT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    let freqs = info.freqs();
    assert_eq!(freqs, frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn empirical_frequencies() {
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"TT"),
            record!("B", b"CA"),
            record!("C", b"NN"),
            record!("D", b"NN"),
        ]),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { tree, msa };
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
        "No sequences found in file"
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
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
    let aligned_sequences = info.compile_alignment(None).unwrap();
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
        Sequences::new(read_sequences(&fldr.join("sequences_some_empty.fasta")).unwrap());
    let aligned_sequences = info.compile_alignment(None).unwrap();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/sequences_empty.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/sequences_protein1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert!(info.is_err());
}

#[test]
fn check_empirical_frequencies() {
    let tree = tree!("((((A:2,B:2):1,C:2):1,D:2):0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"AAAAC"),
            record!("B", b"TTTCC"),
            record!("C", b"GGGCC"),
            record!("D", b"TTAAA"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    let freqs = info.freqs();
    assert_eq!(freqs.clone().sum(), 1.0);
    assert_eq!(freqs, frequencies!(&[5.0, 5.0, 7.0, 3.0]).scale(1.0 / 20.0));
}

#[test]
fn empirical_frequencies_no_ambigs() {
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("one", b"CCCCCCCC"),
            record!("two", b"AAAAAAAA"),
            record!("three", b"TTTTTTTT"),
            record!("four", b"GGGGGGGG"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_x() {
    let tree = tree!("((on:2,tw:2):1,(th:1,fo:1):2);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("on", b"XXXXXXXX"),
            record!("tw", b"XXXXXXXX"),
            record!("th", b"NNNNNNNN"),
            record!("fo", b"NNNNNNNN"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_n() {
    let tree = tree!("(((on:2,tw:2):1,th:1):4,fo:1);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("on", b"AAAAAAAAAA"),
            record!("tw", b"XXXXXXXXXX"),
            record!("th", b"CCCCCCCCCC"),
            record!("fo", b"NNNNNNNNNN"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[0.125, 0.375, 0.375, 0.125]),
        epsilon = 1e-6
    );
}

#[test]
fn empirical_frequencies_ambig_other() {
    let tree = tree!("(A:2,B:2):1.0;");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"VVVVVVVVVV"),
            record!("B", b"TTTTVVVTVV"),
        ]),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo {
        tree: tree.clone(),
        msa,
    };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);

    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"SSSSSSSSSSSSSSSSSSSS"),
            record!("B", b"WWWWWWWWWWWWWWWWWWWW"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_no_aas() {
    let tree = tree!("A:1.0;");
    let msa =
        Alignment::from_aligned(Sequences::new(vec![record!("A", b"BBBBBBBBB")]), &tree).unwrap();

    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[3.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0, 3.0 / 10.0]),
        epsilon = 1e-6
    );
}
