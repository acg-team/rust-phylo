use bio::io::fasta::Record;
use rstest::*;

use crate::alphabets::{
    dna_alphabet, protein_alphabet, ParsimonySet, AMB_AMINOACIDS, AMB_NUCLEOTIDES, AMINOACIDS,
    NUCLEOTIDES,
};

#[test]
fn parsimony_set_iters() {
    let set = ParsimonySet::from_slice(b"ACGT");
    let from_iter = set.iter().cloned().collect::<Vec<u8>>();
    assert_eq!(from_iter.len(), 4);
    let from_into_iter = set.into_iter().collect::<Vec<u8>>();
    assert_eq!(from_into_iter.len(), 4);

    let set = dna_alphabet().parsimony_set(&b'X').clone();
    let from_iter = set.iter().cloned().collect::<Vec<u8>>();
    assert_eq!(from_iter.len(), 4);
    let from_into_iter = set.into_iter().collect::<Vec<u8>>();
    assert_eq!(from_into_iter.len(), 4);
}

#[test]
fn dna_sets() {
    let record = Record::with_attrs("", None, b"AaCcTtGgXn-");

    let sets = record
        .seq()
        .iter()
        .map(|c| dna_alphabet().parsimony_set(c).clone())
        .collect::<Vec<_>>();
    assert_eq!(sets.len(), 11);
    assert_eq!(sets[0], sets[1]);
    assert_eq!(sets[2], sets[3]);
    assert_eq!(sets[4], sets[5]);
    assert_eq!(sets[6], sets[7]);
    assert_eq!(sets[8], sets[9]);
    assert_eq!(sets[10], ParsimonySet::gap());
    assert_eq!(
        &(&sets[0] | &sets[2]) | &(&sets[4] | &sets[6]),
        sets[8].clone()
    );
}

#[test]
fn protein_sets() {
    let record = Record::with_attrs("", None, b"rRlLeEqQxO-");
    let sets = record
        .seq()
        .iter()
        .map(|c| protein_alphabet().parsimony_set(c).clone())
        .collect::<Vec<_>>();
    assert_eq!(sets.len(), 11);
    assert_eq!(sets[0], sets[1]);
    assert_eq!(sets[2], sets[3]);
    assert_eq!(sets[4], sets[5]);
    assert_eq!(sets[6], sets[7]);
    assert_eq!(sets[8], sets[9]);
    assert_eq!(sets[10], ParsimonySet::gap());
}

#[test]
fn dna_characters() {
    let dna = dna_alphabet();
    assert_eq!(dna.parsimony_set(&b'N'), dna.parsimony_set(&b'X'));
    assert_eq!(
        &(dna.parsimony_set(&b'A') | dna.parsimony_set(&b'C'))
            | &(dna.parsimony_set(&b'G') | dna.parsimony_set(&b'T')),
        *dna.parsimony_set(&b'X')
    );
    assert_eq!(dna.parsimony_set(&b'-'), &ParsimonySet::gap());
    assert!(dna
        .parsimony_set(&b'-')
        .is_disjoint(dna.parsimony_set(&b'X')));
    assert!((dna.parsimony_set(&b'-') & dna.parsimony_set(&b'X')).is_empty());
    assert_eq!(dna.parsimony_set(&b'E'), dna.parsimony_set(&b'E'));
    assert_eq!(dna.parsimony_set(&b'T'), dna.parsimony_set(&b'T'));
    assert!((dna.parsimony_set(&b'V') & dna.parsimony_set(&b'T')).is_empty());
    assert!((dna.parsimony_set(&b'D') & dna.parsimony_set(&b'C')).is_empty());
    assert!((dna.parsimony_set(&b'B') & dna.parsimony_set(&b'A')).is_empty());
    assert!((dna.parsimony_set(&b'H') & dna.parsimony_set(&b'G')).is_empty());
    assert_eq!(
        *dna.parsimony_set(&b'M'),
        dna.parsimony_set(&b'A') | dna.parsimony_set(&b'C')
    );
    assert_eq!(
        *dna.parsimony_set(&b'K'),
        dna.parsimony_set(&b'G') | dna.parsimony_set(&b'T')
    );
    assert_eq!(dna.parsimony_set(&b'-'), &ParsimonySet::gap());
}

#[test]
fn protein_characters() {
    let prot = protein_alphabet();
    assert_eq!(prot.parsimony_set(&b'X'), prot.parsimony_set(&b'O'));
    assert_eq!(prot.parsimony_set(&b'-'), &ParsimonySet::gap());
    assert!(prot
        .parsimony_set(&b'-')
        .is_disjoint(prot.parsimony_set(&b'X')));
    assert!((prot.parsimony_set(&b'-') & prot.parsimony_set(&b'X')).is_empty());
    assert_eq!(prot.parsimony_set(&b'E'), prot.parsimony_set(&b'E'));
    assert_eq!(prot.parsimony_set(&b'N'), prot.parsimony_set(&b'N'));
    assert_eq!(
        *prot.parsimony_set(&b'B'),
        prot.parsimony_set(&b'D') | prot.parsimony_set(&b'N')
    );
    assert_eq!(
        *prot.parsimony_set(&b'Z'),
        prot.parsimony_set(&b'E') | prot.parsimony_set(&b'Q')
    );
    assert_eq!(
        *prot.parsimony_set(&b'J'),
        prot.parsimony_set(&b'I') | prot.parsimony_set(&b'L')
    );
}

#[rstest]
#[case(b"ACGT", "ACGT")]
#[case(b"TGAC", "ACGT")]
#[case(b"VG", "GV")]
#[case(b"QIFGWMNKSEYTVCPRAHLD", "ACDEFGHIKLMNPQRSTVWY")]

fn test_parsimony_set_printing(#[case] input: &[u8], #[case] output: &str) {
    let set = ParsimonySet::from_slice(input);
    assert_eq!(format!("{set}"), format!("[{}]", output));
}

#[test]
fn full_sets() {
    let dna = dna_alphabet();
    let full_set = dna.full_set();
    for char in NUCLEOTIDES.iter() {
        assert!(full_set.contains(char));
    }
    for char in AMB_NUCLEOTIDES.iter() {
        assert!(!(full_set & dna.parsimony_set(char)).is_empty());
    }

    let prot = protein_alphabet();
    let full_set = prot.full_set();
    for char in AMINOACIDS.iter() {
        assert!(full_set.contains(char));
    }
    for char in AMB_AMINOACIDS.iter() {
        assert!(!(full_set & prot.parsimony_set(char)).is_empty());
    }
}
