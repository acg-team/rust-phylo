use rstest::*;
use super::*;

#[test]
fn reading_correct_fasta() {
    let sequences = io::read_sequences_from_file("./data/sequences_DNA1.fasta").unwrap();
    assert_eq!(sequences.len(), 4);
    for seq in sequences {
        assert_eq!(seq.seq().len(), 5);
    }

    let corr_lengths = vec![1, 2, 2, 4];
    let sequences = io::read_sequences_from_file("./data/sequences_DNA2_unaligned.fasta").unwrap();
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
    assert!(io::read_sequences_from_file(input).is_err());
}

#[test]
fn reading_nonexistent_fasta() {
    assert!(io::read_sequences_from_file("./data/sequences_nonexistent.fasta").is_err());
}

#[rstest]
#[case::aligned("./data/sequences_DNA1.fasta")]
#[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
#[case::long("./data/sequences_long.fasta")]
fn dna_type_test(#[case] input: &str) {
    let alphabet = sequences::get_sequence_type(&io::read_sequences_from_file(input).unwrap());
    assert_eq!(alphabet, super::sequences::SequenceType::DNA);
    assert_ne!(alphabet, super::sequences::SequenceType::Protein);
}

#[rstest]
#[case("./data/sequences_protein1.fasta")]
fn protein_type_test(#[case] input: &str) {
    let alphabet = sequences::get_sequence_type(&io::read_sequences_from_file(input).unwrap());
    assert_ne!(alphabet, super::sequences::SequenceType::DNA);
    assert_eq!(alphabet, super::sequences::SequenceType::Protein);
}