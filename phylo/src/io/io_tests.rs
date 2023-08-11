use crate::io::read_sequences_from_file;
use rstest::*;
use std::path::PathBuf;

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
