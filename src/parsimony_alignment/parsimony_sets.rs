use bio::io::fasta;
use phylo::sequences::{charify, SequenceType, AMINOACIDS_STR, NUCLEOTIDES_STR};
use std::collections::HashSet;

pub(crate) type ParsimonySet = HashSet<u8>;

pub(crate) fn make_parsimony_set(chars: impl IntoIterator<Item = u8>) -> ParsimonySet {
    ParsimonySet::from_iter(chars)
}

pub(crate) fn get_dna_set(char: &u8) -> ParsimonySet {
    let nucleotides = charify(NUCLEOTIDES_STR);
    if nucleotides.contains(char) {
        ParsimonySet::from_iter(nucleotides.into_iter().filter(|c| c == char))
    } else {
        match *char {
            b'-' => gap_set(),
            b'V' => ParsimonySet::from_iter(nucleotides.into_iter().filter(|c| *c != b'T')),
            b'D' => ParsimonySet::from_iter(nucleotides.into_iter().filter(|c| *c != b'C')),
            b'B' => ParsimonySet::from_iter(nucleotides.into_iter().filter(|c| *c != b'A')),
            b'H' => ParsimonySet::from_iter(nucleotides.into_iter().filter(|c| *c != b'G')),
            b'M' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'A' || *c == b'C'),
            ),
            b'R' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'A' || *c == b'G'),
            ),
            b'W' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'A' || *c == b'T'),
            ),
            b'S' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'C' || *c == b'G'),
            ),
            b'Y' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'C' || *c == b'T'),
            ),
            b'K' => ParsimonySet::from_iter(
                nucleotides.into_iter().filter(|c| *c == b'G' || *c == b'T'),
            ),
            _ => ParsimonySet::from_iter(nucleotides.into_iter()),
        }
    }
}

pub(crate) fn get_protein_set(char: &u8) -> ParsimonySet {
    let aminoacids = charify(AMINOACIDS_STR);
    if aminoacids.contains(char) {
        ParsimonySet::from_iter(aminoacids.into_iter().filter(|c| c == char))
    } else {
        match *char {
            b'-' => gap_set(),
            b'B' => ParsimonySet::from_iter([b'D', b'N'].into_iter()),
            b'Z' => ParsimonySet::from_iter([b'E', b'Q'].into_iter()),
            b'J' => ParsimonySet::from_iter([b'I', b'L'].into_iter()),
            _ => ParsimonySet::from_iter(aminoacids.into_iter()),
        }
    }
}

pub(crate) fn get_parsimony_sets(
    record: &fasta::Record,
    sequence_type: &SequenceType,
) -> Vec<ParsimonySet> {
    let char_set = match sequence_type {
        SequenceType::DNA => |c| get_dna_set(&c),
        SequenceType::Protein => |c| get_protein_set(&c),
    };
    record
        .seq()
        .to_ascii_uppercase()
        .into_iter()
        .map(char_set)
        .collect()
}

pub(crate) fn gap_set() -> ParsimonySet {
    ParsimonySet::from_iter([b'-'].into_iter())
}

#[cfg(test)]
mod parsimony_sets_tests {
    use crate::parsimony_alignment::parsimony_sets::{
        gap_set, get_dna_set, get_parsimony_sets, get_protein_set,
    };
    use bio::io::fasta::Record;
    use phylo::sequences::SequenceType;

    #[test]
    fn dna_sets() {
        let record = Record::with_attrs("", None, b"AaCcTtGgXn-");
        let sets = get_parsimony_sets(&record, &SequenceType::DNA);
        assert_eq!(sets.len(), 11);
        assert_eq!(sets[0], sets[1]);
        assert_eq!(sets[2], sets[3]);
        assert_eq!(sets[4], sets[5]);
        assert_eq!(sets[6], sets[7]);
        assert_eq!(sets[8], sets[9]);
        assert_eq!(sets[10], gap_set());
        assert_eq!(&(&sets[0] | &sets[2]) | &(&sets[4] | &sets[6]), sets[8]);
    }

    #[test]
    fn protein_sets() {
        let record = Record::with_attrs("", None, b"rRlLeEqQxO-");
        let sets = get_parsimony_sets(&record, &SequenceType::Protein);
        assert_eq!(sets.len(), 11);
        assert_eq!(sets[0], sets[1]);
        assert_eq!(sets[2], sets[3]);
        assert_eq!(sets[4], sets[5]);
        assert_eq!(sets[6], sets[7]);
        assert_eq!(sets[8], sets[9]);
        assert_eq!(sets[10], gap_set());
    }

    #[test]
    fn dna_characters() {
        assert_eq!(get_dna_set(&b'N'), get_dna_set(&b'X'));
        assert_eq!(
            &(&get_dna_set(&b'A') | &get_dna_set(&b'C'))
                | &(&get_dna_set(&b'G') | &get_dna_set(&b'T')),
            get_dna_set(&b'X')
        );
        assert_eq!(get_dna_set(&b'-'), gap_set());
        assert!(get_dna_set(&b'-').is_disjoint(&get_dna_set(&b'X')));
        assert!((&get_dna_set(&b'-') & &get_dna_set(&b'X')).is_empty());
        assert_eq!(get_dna_set(&b'E'), get_dna_set(&b'E'));
        assert_eq!(get_dna_set(&b'T'), get_dna_set(&b'T'));
        assert!((&get_dna_set(&b'V') & &get_dna_set(&b'T')).is_empty());
        assert!((&get_dna_set(&b'D') & &get_dna_set(&b'C')).is_empty());
        assert!((&get_dna_set(&b'B') & &get_dna_set(&b'A')).is_empty());
        assert!((&get_dna_set(&b'H') & &get_dna_set(&b'G')).is_empty());
        assert_eq!(
            get_dna_set(&b'M'),
            &get_dna_set(&b'A') | &get_dna_set(&b'C')
        );
        assert_eq!(
            get_dna_set(&b'K'),
            &get_dna_set(&b'G') | &get_dna_set(&b'T')
        );
        assert_eq!(get_dna_set(&b'-'), gap_set());
    }

    #[test]
    fn protein_characters() {
        assert_eq!(get_protein_set(&b'X'), get_protein_set(&b'O'));
        assert_eq!(get_protein_set(&b'-'), gap_set());
        assert!(get_protein_set(&b'-').is_disjoint(&get_protein_set(&b'X')));
        assert!((&get_protein_set(&b'-') & &get_protein_set(&b'X')).is_empty());
        assert_eq!(get_protein_set(&b'E'), get_protein_set(&b'E'));
        assert_eq!(get_protein_set(&b'N'), get_protein_set(&b'N'));
        assert_eq!(
            get_protein_set(&b'B'),
            &get_protein_set(&b'D') | &get_protein_set(&b'N')
        );
        assert_eq!(
            get_protein_set(&b'Z'),
            &get_protein_set(&b'E') | &get_protein_set(&b'Q')
        );
        assert_eq!(
            get_protein_set(&b'J'),
            &get_protein_set(&b'I') | &get_protein_set(&b'L')
        );
    }
}
