use anyhow::bail;
use bio::io::fasta::Record;

use crate::alphabets::{detect_alphabet, Alphabet, GAP};
use crate::Result;

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
    pub fn new(s: Vec<Record>) -> Sequences {
        let alphabet = detect_alphabet(&s);
        Self::with_alphabet(s, alphabet)
    }

    /// Creates a new Sequences object from a vector of bio::io::fasta::Record and a provided alphabet.
    /// The Sequences object is considered aligned if all sequences have the same length.
    pub fn with_alphabet(s: Vec<Record>, alphabet: Alphabet) -> Sequences {
        let msa_len = if s.is_empty() { 0 } else { s[0].seq().len() };
        // Sequences are aligned if all sequences are the same length
        let aligned = s.iter().skip(1).all(|r| r.seq().len() == msa_len);
        Sequences {
            s,
            aligned,
            msa_len,
            alphabet,
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

    pub fn record(&self, idx: usize) -> &Record {
        &self.s[idx]
    }

    pub fn record_mut(&mut self, idx: usize) -> &mut Record {
        &mut self.s[idx]
    }

    pub fn record_by_id(&self, id: &str) -> &Record {
        self.s
            .iter()
            .find(|r| r.id() == id)
            .unwrap_or_else(|| panic!("Sequence with id {} not found", id))
    }

    pub fn try_record_by_id(&self, id: &str) -> Result<&Record> {
        let rec = self.s.iter().find(|r| r.id() == id);
        match rec {
            Some(r) => Ok(r),
            None => bail!("Sequence with id {} not found", id),
        }
    }

    pub fn msa_len(&self) -> usize {
        self.msa_len
    }

    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
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
