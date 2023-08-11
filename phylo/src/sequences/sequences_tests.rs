use crate::io::read_sequences_from_file;
use crate::sequences::{dna_alphabet, get_sequence_type, protein_alphabet, SequenceType};
use bio::alphabets::Alphabet;
use rstest::*;
use std::path::PathBuf;

#[test]
fn alphabets() {
    assert_eq!(
        dna_alphabet(),
        Alphabet::new(b"ACGTRYSWKMBDHVNZXacgtryswkmbdhvnzx-")
    );
    assert_eq!(
        protein_alphabet(),
        Alphabet::new(b"ABCDEFGHIJKLMNPQRSTVWXYZabcdefghijklmnpqrstvwxyz-")
    );
}

#[rstest]
#[case::aligned("./data/sequences_DNA1.fasta")]
#[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
#[case::long("./data/sequences_long.fasta")]
fn dna_type_test(#[case] input: &str) {
    let alphabet = get_sequence_type(&read_sequences_from_file(PathBuf::from(input)).unwrap());
    assert_eq!(alphabet, SequenceType::DNA);
    assert_ne!(alphabet, SequenceType::Protein);
}

#[rstest]
#[case("./data/sequences_protein1.fasta")]
#[case("./data/sequences_protein2.fasta")]
fn protein_type_test(#[case] input: &str) {
    let alphabet = get_sequence_type(&read_sequences_from_file(PathBuf::from(input)).unwrap());
    assert_ne!(alphabet, SequenceType::DNA);
    assert_eq!(alphabet, SequenceType::Protein);
}
