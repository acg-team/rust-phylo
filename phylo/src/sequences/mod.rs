use bio::alphabets::Alphabet;
use bio::io::fasta::Record;

use crate::evolutionary_models::{
    DNAModelType,
    ModelType::{self, *},
    ProteinModelType,
};

pub static AMINOACIDS: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
pub static AMB_AMINOACIDS: &[u8] = b"BJZX";
pub static NUCLEOTIDES: &[u8] = b"TCAG";
pub static AMB_NUCLEOTIDES: &[u8] = b"RYSWKMBDHVNZX";
pub static GAP: u8 = b'-';

pub fn dna_alphabet() -> Alphabet {
    let mut nucleotides = NUCLEOTIDES.to_vec();
    nucleotides.append(&mut AMB_NUCLEOTIDES.to_vec());
    nucleotides.append(&mut nucleotides.clone().to_ascii_lowercase());
    nucleotides.push(GAP);
    Alphabet::new(nucleotides)
}

pub fn protein_alphabet() -> Alphabet {
    let mut aminoacids = AMINOACIDS.to_vec();
    aminoacids.append(&mut AMB_AMINOACIDS.to_vec());
    aminoacids.append(&mut aminoacids.clone().to_ascii_lowercase());
    aminoacids.push(GAP);
    Alphabet::new(aminoacids)
}

pub fn sequence_type(sequences: &[Record]) -> ModelType {
    let dna_alphabet = dna_alphabet();
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return Protein(ProteinModelType::UNDEF);
        }
    }
    DNA(DNAModelType::UNDEF)
}

#[cfg(test)]
mod sequences_tests;
