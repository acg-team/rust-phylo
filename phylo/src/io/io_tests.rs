use crate::io::{read_sequences_from_file, write_sequences_to_file};
use bio::io::fasta;
use rstest::*;
use std::io::{Read, Write};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn reading_correct_fasta() {
    let sequences = read_sequences_from_file(PathBuf::from("./data/sequences_DNA1.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for seq in sequences {
        assert_eq!(seq.seq().len(), 5);
    }

    let corr_lengths = vec![1, 2, 2, 4];
    let sequences =
        read_sequences_from_file(PathBuf::from("./data/sequences_DNA2_unaligned.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for (i, seq) in sequences.into_iter().enumerate() {
        assert_eq!(seq.seq().len(), corr_lengths[i]);
    }
}

#[rstest]
#[case::empty_sequence_name("./data/sequences_garbage_empty_name.fasta")]
#[case::garbage_sequence("./data/sequences_garbage_non-ascii.fasta")]
#[case::weird_chars("./data/sequences_garbage_weird_symbols.fasta")]
fn reading_incorrect_fasta(#[case] input: &str) {
    assert!(read_sequences_from_file(PathBuf::from(input)).is_err());
}

#[test]
fn reading_nonexistent_fasta() {
    assert!(read_sequences_from_file(PathBuf::from("./data/sequences_nonexistent.fasta")).is_err());
}

#[test]
fn test_write_sequences_to_file() {
    let sequences = vec![
        fasta::Record::with_attrs("seq1", None, b"ATGC"),
        fasta::Record::with_attrs("seq2", None, b"CGTA"),
    ];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.fasta");
    write_sequences_to_file(&sequences, output_path.clone()).unwrap();
    let mut file_content = String::new();
    std::fs::File::open(output_path)
        .unwrap()
        .read_to_string(&mut file_content)
        .unwrap();
    let expected_output = ">seq1\nATGC\n>seq2\nCGTA\n";
    assert_eq!(file_content, expected_output);
}

#[test]
fn test_write_sequences_to_file_bad_path() {
    let sequences = vec![
        fasta::Record::with_attrs("seq1", None, b"ATGC"),
        fasta::Record::with_attrs("seq2", None, b"CGTA"),
    ];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir
        .path()
        .join("nonexistent_folder")
        .join("output.fasta");
    assert!(write_sequences_to_file(&sequences, output_path.clone()).is_err());
}
