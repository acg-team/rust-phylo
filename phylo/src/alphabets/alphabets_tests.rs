use assert_matches::assert_matches;
use rstest::*;

use std::path::PathBuf;

use crate::alignment::Sequences;
use crate::alphabets::sequence_type;
use crate::evolutionary_models::ModelType;
use crate::io::read_sequences_from_file;

#[rstest]
#[case::aligned("./data/sequences_DNA1.fasta")]
#[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
#[case::long("./data/sequences_long.fasta")]
fn dna_type_test(#[case] input: &str) {
    let seqs = read_sequences_from_file(&PathBuf::from(input)).unwrap();
    let alphabet = sequence_type(&Sequences::new(seqs));
    assert_matches!(alphabet, ModelType::DNA(_));
}

#[rstest]
#[case("./data/sequences_protein1.fasta")]
#[case("./data/sequences_protein2.fasta")]
fn protein_type_test(#[case] input: &str) {
    let seqs = read_sequences_from_file(&PathBuf::from(input)).unwrap();
    let alphabet = sequence_type(&Sequences::new(seqs));
    assert_matches!(alphabet, ModelType::Protein(_));
}
