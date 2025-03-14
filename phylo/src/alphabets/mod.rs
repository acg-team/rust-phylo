use std::collections::HashSet;
use std::fmt::Display;

use bio::io::fasta::Record;
use itertools::join;
use lazy_static::lazy_static;

use crate::frequencies;
use crate::substitution_models::FreqVector;

pub static AMINOACIDS: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
pub static AMB_AMINOACIDS: &[u8] = b"BJZX";
pub static NUCLEOTIDES: &[u8] = b"TCAG";
pub static AMB_NUCLEOTIDES: &[u8] = b"RYSWKMBDHVNZX";
pub static AMB_CHAR: u8 = b'X';
pub static GAP: u8 = b'-';
pub static POSSIBLE_GAPS: &[u8] = b"_*-";

#[derive(Debug, PartialEq, Clone)]
pub struct Alphabet {
    name: &'static str,
    symbols: &'static [u8],
    ambiguous: &'static [u8],
    index: &'static [usize; 255],
    valid_symbols: &'static HashSet<u8>,
    conditional_probs: &'static [FreqVector],
    parsimony_sets: &'static [HashSet<u8>],
}

impl Display for Alphabet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
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
        self.conditional_probs[char.to_ascii_uppercase() as usize].clone()
    }

    pub fn empty_freqs(&self) -> FreqVector {
        FreqVector::zeros(self.conditional_probs[AMB_CHAR as usize].nrows())
    }

    pub fn index(&self, char: &u8) -> usize {
        self.index[*char as usize]
    }

    pub fn gap_encoding(&self) -> &FreqVector {
        &self.conditional_probs[AMB_CHAR as usize]
    }

    pub fn parsimony_set(&self, char: &u8) -> ParsimonySet {
        self.parsimony_sets[*char as usize].clone()
    }
}

pub fn detect_alphabet(sequences: &[Record]) -> Alphabet {
    let dna_alphabet = dna_alphabet();
    for record in sequences.iter() {
        if !dna_alphabet.is_word(record.seq()) {
            return protein_alphabet();
        }
    }
    dna_alphabet
}

pub(crate) type ParsimonySet = HashSet<u8>;

#[allow(dead_code)]
pub(crate) fn print_parsimony_set(set: &ParsimonySet) -> String {
    let mut chars: Vec<char> = set.iter().map(|&a| a as char).collect();
    chars.sort();
    join(chars, " ")
}

pub(crate) fn dna_alphabet() -> Alphabet {
    Alphabet {
        name: "DNA",
        symbols: NUCLEOTIDES,
        ambiguous: AMB_NUCLEOTIDES,
        index: &NUCLEOTIDE_INDEX,
        valid_symbols: &VALID_NUCLEOTIDES,
        conditional_probs: &NUCL_COND_PROBS,
        parsimony_sets: &NUCL_PARSIMONY_SETS,
    }
}

pub(crate) fn protein_alphabet() -> Alphabet {
    Alphabet {
        name: "protein",
        symbols: AMINOACIDS,
        ambiguous: AMB_AMINOACIDS,
        index: &AMINOACID_INDEX,
        valid_symbols: &VALID_AMINOACIDS,
        conditional_probs: &AA_COND_PROBS,
        parsimony_sets: &AA_PARSIMONY_SETS,
    }
}

pub(crate) fn gap_set() -> ParsimonySet {
    ParsimonySet::from_iter([GAP])
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
    pub static ref VALID_NUCLEOTIDES: HashSet<u8> = {
        NUCLEOTIDES
            .iter()
            .chain(AMB_NUCLEOTIDES.iter().chain([GAP].iter()))
            .cloned()
            .collect()
    };
    pub static ref NUCL_COND_PROBS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &nucl_cond_probs(char));
        }
        map
    };
    pub static ref NUCL_PARSIMONY_SETS: Vec<ParsimonySet> = {
        let mut map: Vec<ParsimonySet> = vec![ParsimonySet::new(); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            *elem = nucl_parsimony_set(&char);
        }
        map
    };
}

fn nucl_cond_probs(char: u8) -> FreqVector {
    let char = char.to_ascii_uppercase();
    match char {
        b'T' => frequencies!(&[1.0, 0.0, 0.0, 0.0]),
        b'C' => frequencies!(&[0.0, 1.0, 0.0, 0.0]),
        b'A' => frequencies!(&[0.0, 0.0, 1.0, 0.0]),
        b'G' => frequencies!(&[0.0, 0.0, 0.0, 1.0]),
        b'M' => frequencies!(&[0.0, 1.0, 1.0, 0.0]),
        b'R' => frequencies!(&[0.0, 0.0, 1.0, 1.0]),
        b'W' => frequencies!(&[1.0, 0.0, 1.0, 0.0]),
        b'S' => frequencies!(&[0.0, 1.0, 0.0, 1.0]),
        b'Y' => frequencies!(&[1.0, 1.0, 0.0, 0.0]),
        b'K' => frequencies!(&[1.0, 0.0, 0.0, 1.0]),
        b'V' => frequencies!(&[0.0, 1.0, 1.0, 1.0]),
        b'D' => frequencies!(&[1.0, 0.0, 1.0, 1.0]),
        b'B' => frequencies!(&[1.0, 1.0, 0.0, 1.0]),
        b'H' => frequencies!(&[1.0, 1.0, 1.0, 0.0]),
        _ => frequencies!(&[1.0; 4]),
    }
}

fn nucl_parsimony_set(char: &u8) -> ParsimonySet {
    let char = char.to_ascii_uppercase();
    if NUCLEOTIDES.contains(&char) {
        return ParsimonySet::from_iter([char]);
    }
    match char {
        b'-' => ParsimonySet::from_iter([GAP]),
        b'M' => ParsimonySet::from_iter([b'C', b'A']),
        b'R' => ParsimonySet::from_iter([b'A', b'G']),
        b'W' => ParsimonySet::from_iter([b'T', b'A']),
        b'S' => ParsimonySet::from_iter([b'C', b'G']),
        b'Y' => ParsimonySet::from_iter([b'T', b'C']),
        b'K' => ParsimonySet::from_iter([b'T', b'G']),
        b'V' => ParsimonySet::from_iter([b'C', b'A', b'G']),
        b'D' => ParsimonySet::from_iter([b'T', b'A', b'G']),
        b'B' => ParsimonySet::from_iter([b'T', b'C', b'G']),
        b'H' => ParsimonySet::from_iter([b'T', b'C', b'A']),
        _ => ParsimonySet::from_iter(NUCLEOTIDES.iter().cloned()),
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
    pub static ref VALID_AMINOACIDS: HashSet<u8> = {
        AMINOACIDS
            .iter()
            .chain(AMB_AMINOACIDS.iter().chain([GAP].iter()))
            .cloned()
            .collect()
    };
    pub static ref AA_COND_PROBS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &aa_cond_probs(char));
        }
        map
    };
    pub static ref AA_PARSIMONY_SETS: Vec<ParsimonySet> = {
        let mut map: Vec<ParsimonySet> = vec![ParsimonySet::new(); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            *elem = aa_parsimony_set(&char);
        }
        map
    };
}

fn aa_cond_probs(char: u8) -> FreqVector {
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
            set.fill_row(index[b'D' as usize], 1.0);
            set.fill_row(index[b'N' as usize], 1.0);
            set
        }
        b'Z' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'E' as usize], 1.0);
            set.fill_row(index[b'Q' as usize], 1.0);
            set
        }
        b'J' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'I' as usize], 1.0);
            set.fill_row(index[b'L' as usize], 1.0);
            set
        }
        _ => {
            frequencies!(&[1.0; 20])
        }
    }
}

fn aa_parsimony_set(char: &u8) -> ParsimonySet {
    let char = char.to_ascii_uppercase();
    if AMINOACIDS.contains(&char) {
        return ParsimonySet::from_iter([char]);
    }
    match char {
        b'-' => ParsimonySet::from_iter([GAP]),
        b'B' => ParsimonySet::from_iter([b'D', b'N']),
        b'Z' => ParsimonySet::from_iter([b'E', b'Q']),
        b'J' => ParsimonySet::from_iter([b'I', b'L']),
        _ => ParsimonySet::from_iter(AMINOACIDS.iter().cloned()),
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
