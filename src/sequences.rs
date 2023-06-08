use bio::{alphabets::Alphabet, io::fasta};

#[derive(PartialEq, Debug)]
pub(crate) enum SequenceType {
    DNA,
    Protein,
}

pub(crate) fn charify(chars: &str) -> Vec<u8> {
    chars.chars().map(|c| c as u8).collect()
}

pub(crate) static AMINOACIDS_STR: &str = "ARNDCQEGHILKMFPSTWYV";
pub(crate) static AMB_AMINOACIDS_STR: &str = "BJZX";

pub(crate) static NUCLEOTIDES_STR: &str = "TCAG";
pub(crate) static AMB_NUCLEOTIDES_STR: &str = "RYSWKMBDHVNZX";

pub(crate) static GAP: u8 = b'-';

pub(crate) fn dna_alphabet() -> Alphabet {
    let mut nucleotides = charify(NUCLEOTIDES_STR);
    nucleotides.append(&mut (charify(AMB_NUCLEOTIDES_STR)));
    nucleotides.append(&mut nucleotides.clone().to_ascii_lowercase());
    nucleotides.push(GAP);
    Alphabet::new(nucleotides)
}

#[allow(dead_code)]
pub(crate) fn protein_alphabet() -> Alphabet {
    let mut aminoacids = charify(AMINOACIDS_STR);
    aminoacids.append(&mut (charify(AMB_AMINOACIDS_STR)));
    aminoacids.append(&mut aminoacids.clone().to_ascii_lowercase());
    aminoacids.push(GAP);
    Alphabet::new(aminoacids)
}

pub(crate) fn get_sequence_type(sequences: &Vec<fasta::Record>) -> SequenceType {
    let dna_alphabet = dna_alphabet();
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return SequenceType::Protein;
        }
    }
    SequenceType::DNA
}

#[cfg(test)]
mod sequences_tests {
    use crate::io::read_sequences_from_file;
    use crate::sequences::{dna_alphabet, get_sequence_type, protein_alphabet, SequenceType};
    use bio::alphabets::Alphabet;
    use rstest::*;

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
        let alphabet = get_sequence_type(&read_sequences_from_file(input).unwrap());
        assert_eq!(alphabet, SequenceType::DNA);
        assert_ne!(alphabet, SequenceType::Protein);
    }

    #[rstest]
    #[case("./data/sequences_protein1.fasta")]
    fn protein_type_test(#[case] input: &str) {
        let alphabet = get_sequence_type(&read_sequences_from_file(input).unwrap());
        assert_ne!(alphabet, SequenceType::DNA);
        assert_eq!(alphabet, SequenceType::Protein);
    }
}
