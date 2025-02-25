use rstest::*;

use std::path::PathBuf;

use crate::alphabets::{detect_alphabet, dna_alphabet, protein_alphabet};
use crate::io::read_sequences_from_file;

#[rstest]
#[case::aligned("./data/sequences_DNA1.fasta")]
#[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
#[case::long("./data/sequences_long.fasta")]
fn dna_type_test(#[case] input: &str) {
    let seqs = read_sequences_from_file(&PathBuf::from(input)).unwrap();
    let alphabet = detect_alphabet(&seqs);
    assert_eq!(alphabet, dna_alphabet());
    assert!(format!("{}", dna_alphabet()).contains("DNA"));
}

#[rstest]
#[case("./data/sequences_protein1.fasta")]
#[case("./data/sequences_protein2.fasta")]
fn protein_type_test(#[case] input: &str) {
    let seqs = read_sequences_from_file(&PathBuf::from(input)).unwrap();
    let alphabet = detect_alphabet(&seqs);
    assert_eq!(alphabet, protein_alphabet());
    assert!(format!("{}", protein_alphabet()).contains("protein"));
}
