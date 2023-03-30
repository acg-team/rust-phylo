use approx::relative_eq;
use nalgebra::dmatrix;

use rstest::*;

use super::*;

impl tree::Node {
    fn new(idx: usize, parent: Option<usize>, children: Vec<usize>, blength: f32) -> Self {
        Self {
            idx: idx,
            parent: parent,
            children: children,
            blen: blength,
        }
    }
}

impl PartialEq for tree::Node {
    fn eq(&self, other: &Self) -> bool {
        (self.idx == other.idx)
            && (self.parent == other.parent)
            && (self.children.iter().min() == other.children.iter().min())
            && (self.children.iter().max() == other.children.iter().max())
            && relative_eq!(self.blen, other.blen)
    }
}

#[test]
fn nj_correct() {
    let nj_distances = njmat::NJMat {
        idx: (0..5).collect(),
        distances: dmatrix![
            0.0, 5.0, 9.0, 9.0, 8.0;
            5.0, 0.0, 10.0, 10.0, 9.0;
            9.0, 10.0, 0.0, 8.0, 7.0;
            9.0, 10.0, 8.0, 0.0, 3.0;
            8.0, 9.0, 7.0, 3.0, 0.0],
    };

    let nj_tree = tree::build_nj_tree(nj_distances).unwrap();

    let result = vec![
        tree::Node::new(0, Some(5), Vec::new(), 2.0),
        tree::Node::new(1, Some(5), Vec::new(), 3.0),
        tree::Node::new(2, Some(6), Vec::new(), 4.0),
        tree::Node::new(3, Some(7), Vec::new(), 2.0),
        tree::Node::new(4, Some(7), Vec::new(), 1.0),
        tree::Node::new(5, Some(6), vec![1, 0], 3.0),
        tree::Node::new(6, Some(8), vec![2, 5], 1.0),
        tree::Node::new(7, Some(8), vec![4, 3], 1.0),
        tree::Node::new(8, None, vec![7, 6], 0.0),
    ];

    assert_eq!(nj_tree.root, 8);
    for (i, node) in nj_tree.nodes.into_iter().enumerate() {
        assert_eq!(node, result[i]);
    }
}

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