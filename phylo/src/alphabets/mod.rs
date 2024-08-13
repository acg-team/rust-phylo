use std::collections::HashSet;

use lazy_static::lazy_static;

use crate::alignment::Sequences;
use crate::evolutionary_models::{
    DNAModelType,
    ModelType::{self, *},
    ProteinModelType,
};
use crate::frequencies;
use crate::phylo_info::GapHandling;
use crate::substitution_models::FreqVector;

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
    valid_symbols: HashSet<u8>,
}

impl Alphabet {
    pub fn is_word(&self, word: &[u8]) -> bool {
        word.to_ascii_uppercase()
            .iter()
            .all(|c| self.valid_symbols.contains(c))
    }

    pub fn symbols(&self) -> &[u8] {
        self.symbols
    }

    pub fn ambiguous(&self) -> &[u8] {
        self.ambiguous
    }

    pub fn char_encoding(&self, char: u8) -> FreqVector {
        self.char_sets[char.to_ascii_uppercase() as usize].clone()
    }

    pub fn empty_freqs(&self) -> FreqVector {
        FreqVector::zeros(self.char_sets[b'X' as usize].nrows())
    }

    pub fn index(&self, char: &u8) -> usize {
        self.index[*char as usize]
    }
}

pub fn sequence_type(sequences: &Sequences) -> ModelType {
    let dna_alphabet = dna_alphabet(&GapHandling::Ambiguous);
    for record in sequences.iter() {
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
    let mut valid_symbols: HashSet<u8> = NUCLEOTIDES.iter().cloned().collect();
    valid_symbols.extend(AMB_NUCLEOTIDES.iter().cloned());
    valid_symbols.insert(GAP);
    match gap_handling {
        GapHandling::Ambiguous => Alphabet {
            symbols: NUCLEOTIDES,
            ambiguous: AMB_NUCLEOTIDES,
            char_sets: &NUCLEOTIDE_SETS,
            index: &NUCLEOTIDE_INDEX,
            valid_symbols,
        },
        _ => Alphabet {
            symbols: NUCLEOTIDES,
            ambiguous: AMB_NUCLEOTIDES,
            char_sets: &NUCLEOTIDE_GAP_SETS,
            index: &NUCLEOTIDE_INDEX,
            valid_symbols,
        },
    }
}

fn protein_alphabet(gap_handlind: &GapHandling) -> Alphabet {
    let mut valid_symbols: HashSet<u8> = AMINOACIDS.iter().cloned().collect();
    valid_symbols.extend(AMB_AMINOACIDS.iter().cloned());
    valid_symbols.insert(GAP);
    match gap_handlind {
        GapHandling::Ambiguous => Alphabet {
            symbols: AMINOACIDS,
            ambiguous: AMB_AMINOACIDS,
            char_sets: &AMINOACID_SETS_AMBIG,
            index: &AMINOACID_INDEX,
            valid_symbols,
        },
        _ => Alphabet {
            symbols: AMINOACIDS,
            ambiguous: AMB_AMINOACIDS,
            char_sets: &AMINOACID_SETS_PROPER,
            index: &AMINOACID_INDEX,
            valid_symbols,
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
    pub static ref NUCLEOTIDE_GAP_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0, 0.0, 0.0, 0.0]));
            } else {
                elem.set_column(0, &generic_nucleotide_sets(char));
            }
        }
        map
    };
    pub static ref NUCLEOTIDE_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_nucleotide_sets(char));
        }
        map
    };
}

fn generic_nucleotide_sets(char: u8) -> FreqVector {
    let char = char.to_ascii_uppercase();
    match char {
        b'T' => frequencies!(&[1.0, 0.0, 0.0, 0.0]),
        b'C' => frequencies!(&[0.0, 1.0, 0.0, 0.0]),
        b'A' => frequencies!(&[0.0, 0.0, 1.0, 0.0]),
        b'G' => frequencies!(&[0.0, 0.0, 0.0, 1.0]),
        b'M' => frequencies!(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0]),
        b'R' => frequencies!(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0]),
        b'W' => frequencies!(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0]),
        b'S' => frequencies!(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0]),
        b'Y' => frequencies!(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
        b'K' => frequencies!(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
        b'V' => {
            frequencies!(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'D' => {
            frequencies!(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'B' => {
            frequencies!(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
        }
        b'H' => {
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
    pub static ref AMINOACID_SETS_PROPER: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0; 20]));
            } else {
                elem.set_column(0, &generic_aminoacid_sets(char));
            }
        }
        map
    };
    pub static ref AMINOACID_SETS_AMBIG: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_aminoacid_sets(char));
        }
        map
    };
}

fn generic_aminoacid_sets(char: u8) -> FreqVector {
    let char = char.to_ascii_uppercase();
    let index = &AMINOACID_INDEX;
    if AMINOACIDS.contains(&char) {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index[char as usize], 1.0);
        return set;
    }
    match char {
        b'B' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'D' as usize], 0.5);
            set.fill_row(index[b'N' as usize], 0.5);
            set
        }
        b'Z' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'E' as usize], 0.5);
            set.fill_row(index[b'Q' as usize], 0.5);
            set
        }
        b'J' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'I' as usize], 0.5);
            set.fill_row(index[b'L' as usize], 0.5);
            set
        }
        _ => {
            frequencies!(&[1.0 / 20.0; 20])
        }
    }
}

#[cfg(test)]
mod alphabets_tests;