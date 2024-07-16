use assert_matches::assert_matches;
use rstest::*;

use std::path::PathBuf;

use bio::alphabets::Alphabet;

use crate::evolutionary_models::ModelType;
use crate::io::read_sequences_from_file;
use crate::sequences::{dna_alphabet, get_sequence_type, protein_alphabet};

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
    assert_matches!(alphabet, ModelType::DNA(_));
}

#[rstest]
#[case("./data/sequences_protein1.fasta")]
#[case("./data/sequences_protein2.fasta")]
fn protein_type_test(#[case] input: &str) {
    let alphabet = get_sequence_type(&read_sequences_from_file(PathBuf::from(input)).unwrap());
    assert_matches!(alphabet, ModelType::Protein(_));
}
