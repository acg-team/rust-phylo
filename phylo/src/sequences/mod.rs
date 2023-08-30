use bio::alphabets::Alphabet;
use bio::io::fasta;

#[derive(PartialEq, Eq, Debug)]
pub enum SequenceType {
    DNA,
    Protein,
}

pub fn charify(chars: &str) -> Vec<u8> {
    chars.chars().map(|c| c as u8).collect()
}

pub static AMINOACIDS_STR: &str = "ARNDCQEGHILKMFPSTWYV";
pub static AMB_AMINOACIDS_STR: &str = "BJZX";

pub static NUCLEOTIDES_STR: &str = "TCAG";
pub static AMB_NUCLEOTIDES_STR: &str = "RYSWKMBDHVNZX";

pub(crate) static GAP: u8 = b'-';

pub fn dna_alphabet() -> Alphabet {
    let mut nucleotides = charify(NUCLEOTIDES_STR);
    nucleotides.append(&mut (charify(AMB_NUCLEOTIDES_STR)));
    nucleotides.append(&mut nucleotides.clone().to_ascii_lowercase());
    nucleotides.push(GAP);
    Alphabet::new(nucleotides)
}

pub fn protein_alphabet() -> Alphabet {
    let mut aminoacids = charify(AMINOACIDS_STR);
    aminoacids.append(&mut (charify(AMB_AMINOACIDS_STR)));
    aminoacids.append(&mut aminoacids.clone().to_ascii_lowercase());
    aminoacids.push(GAP);
    Alphabet::new(aminoacids)
}

pub fn get_sequence_type(sequences: &Vec<fasta::Record>) -> SequenceType {
    let dna_alphabet = dna_alphabet();
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return SequenceType::Protein;
        }
    }
    SequenceType::DNA
}

#[cfg(test)]
mod sequences_tests;