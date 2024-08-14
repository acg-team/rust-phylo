use crate::{
    alphabets::{detect_alphabet, Alphabet, GAP},
    phylo_info::GapHandling,
};

use bio::io::fasta::Record;

#[derive(Debug, Clone, PartialEq)]
pub struct Sequences {
    pub(crate) s: Vec<Record>,
    pub(crate) aligned: bool,
    pub(crate) msa_len: usize,
    pub(crate) alphabet: Alphabet,
}

impl Sequences {
    /// Creates a new Sequences object from a vector of bio::io::fasta::Record.
    /// The Sequences object is considered aligned if all sequences have the same length.
    /// By default gap handling is set to proper.
    pub fn new(s: Vec<Record>) -> Sequences {
        Self::with_attrs(s, &GapHandling::Proper)
    }

    /// Creates a new Sequences object from a vector of bio::io::fasta::Record.
    /// The Sequences object is considered aligned if all sequences have the same length.
    pub fn with_attrs(s: Vec<Record>, gap_handling: &GapHandling) -> Sequences {
        let len = if s.is_empty() { 0 } else { s[0].seq().len() };
        let alphabet = detect_alphabet(&s, gap_handling);
        if s.iter().filter(|rec| rec.seq().len() != len).count() == 0 {
            Sequences {
                s,
                aligned: true,
                msa_len: len,
                alphabet,
            }
        } else {
            Sequences {
                s,
                aligned: false,
                msa_len: 0,
                alphabet,
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Record> {
        self.s.iter()
    }

    pub fn len(&self) -> usize {
        self.s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    pub fn get(&self, idx: usize) -> &Record {
        &self.s[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut Record {
        &mut self.s[idx]
    }

    pub fn get_by_id(&self, id: &str) -> &Record {
        self.s.iter().find(|r| r.id() == id).unwrap()
    }

    pub fn msa_len(&self) -> usize {
        self.msa_len
    }

    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    pub fn gap_handling(&self) -> &GapHandling {
        &self.alphabet.gap_handling
    }

    /// Removes all gaps from the sequences and returns a new Sequences object.
    pub fn without_gaps(self) -> Sequences {
        let seqs = self
            .s
            .iter()
            .map(|rec| {
                let sequence = rec
                    .seq()
                    .iter()
                    .filter(|&c| c != &GAP)
                    .copied()
                    .collect::<Vec<u8>>();
                Record::with_attrs(rec.id(), rec.desc(), &sequence)
            })
            .collect();
        Sequences {
            s: seqs,
            aligned: false,
            msa_len: 0,
            alphabet: self.alphabet,
        }
    }
}
