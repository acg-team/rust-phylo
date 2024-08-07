use lazy_static::lazy_static;

use bio::io::fasta::Record;

use crate::frequencies;
use crate::substitution_models::FreqVector;
use crate::{
    evolutionary_models::{
        DNAModelType,
        ModelType::{self, *},
        ProteinModelType,
    },
    phylo_info::GapHandling,
};

pub static AMINOACIDS: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
pub static AMB_AMINOACIDS: &[u8] = b"BJZX";
pub static NUCLEOTIDES: &[u8] = b"TCAG";
pub static AMB_NUCLEOTIDES: &[u8] = b"RYSWKMBDHVNZX";
pub static GAP: u8 = b'-';

#[derive(Debug, PartialEq)]
pub struct Alphabet {
    symbols: &'static [u8],
    ambiguous: &'static [u8],
    char_sets: &'static [FreqVector],
    index: &'static [usize; 255],
}

impl Alphabet {
    pub fn is_word(&self, word: &[u8]) -> bool {
        word.to_ascii_uppercase()
            .iter()
            .all(|c| self.symbols.contains(c) | self.ambiguous.contains(c) | (*c == GAP))
    }

    pub fn symbols(&self) -> Vec<u8> {
        self.symbols.to_vec()
    }

    pub fn all_symbols(&self) -> Vec<u8> {
        let mut symbols = self.symbols.to_vec();
        symbols.extend_from_slice(self.ambiguous);
        symbols
    }

    pub fn char_encoding(&self, char: u8) -> FreqVector {
        self.char_sets[char.to_ascii_uppercase() as usize].clone()
    }

    pub fn empty_freqs(&self) -> FreqVector {
        FreqVector::zeros(self.char_sets[b'X' as usize].nrows())
    }

    pub fn index(&self) -> &[usize; 255] {
        self.index
    }
}

pub fn sequence_type(sequences: &[Record]) -> ModelType {
    let dna_alphabet = dna_alphabet(&GapHandling::Ambiguous);
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return Protein(ProteinModelType::UNDEF);
        }
    }
    DNA(DNAModelType::UNDEF)
}

pub fn alphabet_from_type(model_type: ModelType, gap_handling: &GapHandling) -> Alphabet {
    match model_type {
        DNA(_) => dna_alphabet(gap_handling),
        Protein(_) => protein_alphabet(gap_handling),
    }
}

fn dna_alphabet(gap_handling: &GapHandling) -> Alphabet {
    match gap_handling {
        GapHandling::Ambiguous => Alphabet {
            symbols: NUCLEOTIDES,
            ambiguous: AMB_NUCLEOTIDES,
            char_sets: &DNA_SETS,
            index: &NUCLEOTIDE_INDEX,
        },
        _ => Alphabet {
            symbols: NUCLEOTIDES,
            ambiguous: AMB_NUCLEOTIDES,
            char_sets: &DNA_GAP_SETS,
            index: &NUCLEOTIDE_INDEX,
        },
    }
}

fn protein_alphabet(gap_handlind: &GapHandling) -> Alphabet {
    match gap_handlind {
        GapHandling::Ambiguous => Alphabet {
            symbols: AMINOACIDS,
            ambiguous: AMB_AMINOACIDS,
            char_sets: &PROTEIN_SETS,
            index: &AMINOACID_INDEX,
        },
        _ => Alphabet {
            symbols: AMINOACIDS,
            ambiguous: AMB_AMINOACIDS,
            char_sets: &PROTEIN_GAP_SETS,
            index: &AMINOACID_INDEX,
        },
    }
}

lazy_static! {
    pub static ref NUCLEOTIDE_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, char) in NUCLEOTIDES.iter().enumerate() {
            index[*char as usize] = i;
            index[(*char).to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 4;
        index
    };
    pub static ref DNA_GAP_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0, 0.0, 0.0, 0.0]));
            } else {
                elem.set_column(0, &generic_dna_sets(char));
            }
        }
        map
    };
    pub static ref DNA_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_dna_sets(char));
        }
        map
    };
}

fn generic_dna_sets(char: u8) -> FreqVector {
    match char.to_ascii_uppercase() {
        b'T' | b't' => frequencies!(&[1.0, 0.0, 0.0, 0.0]),
        b'C' | b'c' => frequencies!(&[0.0, 1.0, 0.0, 0.0]),
        b'A' | b'a' => frequencies!(&[0.0, 0.0, 1.0, 0.0]),
        b'G' | b'g' => frequencies!(&[0.0, 0.0, 0.0, 1.0]),
        b'M' | b'm' => frequencies!(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0]),
        b'R' | b'r' => frequencies!(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0]),
        b'W' | b'w' => frequencies!(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0]),
        b'S' | b's' => frequencies!(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0]),
        b'Y' | b'y' => frequencies!(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
        b'K' | b'k' => frequencies!(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
        b'V' | b'v' => {
            frequencies!(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'D' | b'd' => {
            frequencies!(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'B' | b'b' => {
            frequencies!(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
        }
        b'H' | b'h' => {
            frequencies!(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
        }
        _ => frequencies!(&[1.0 / 4.0; 4]),
    }
}

lazy_static! {
    pub static ref AMINOACID_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, &char) in AMINOACIDS.iter().enumerate() {
            index[char as usize] = i;
            index[char.to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 20;
        index
    };
    pub static ref PROTEIN_GAP_SETS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0; 20]));
            } else {
                elem.set_column(0, &generic_protein_sets(char));
            }
        }
        map
    };
    pub static ref PROTEIN_SETS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_protein_sets(char));
        }
        map
    };
}

fn generic_protein_sets(char: u8) -> FreqVector {
    let index = &AMINOACID_INDEX;
    if AMINOACIDS.contains(&char.to_ascii_uppercase()) {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index[char as usize], 1.0);
        set
    } else if char.to_ascii_uppercase() == b'B' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['D' as usize], 0.5);
        set.fill_row(index['N' as usize], 0.5);
        set
    } else if char.to_ascii_uppercase() == b'Z' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['E' as usize], 0.5);
        set.fill_row(index['Q' as usize], 0.5);
        set
    } else if char.to_ascii_uppercase() == b'J' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['I' as usize], 0.5);
        set.fill_row(index['L' as usize], 0.5);
        set
    } else {
        frequencies!(&[1.0 / 20.0; 20])
    }
}

#[cfg(test)]
mod alphabets_tests;
