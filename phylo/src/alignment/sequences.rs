use std::collections::HashMap;
use std::fmt::Display;

use anyhow::bail;
use bio::io::fasta::Record;
use nalgebra::DMatrix;

use crate::alphabets::{detect_alphabet, Alphabet, GAP};
use crate::Result;

#[derive(Debug, Clone)]
pub struct Sequences {
    pub(crate) s: Vec<Record>,
    pub(crate) aligned: bool,
    pub(crate) alphabet: Alphabet,
}

impl PartialEq for Sequences {
    fn eq(&self, other: &Self) -> bool {
        self.s.len() == other.s.len()
            && self.aligned == other.aligned
            && self.alphabet == other.alphabet
            && {
                let mut self_records = self.s.clone();
                let mut other_records = other.s.clone();
                self_records.sort_by(|a, b| a.id().cmp(b.id()));
                other_records.sort_by(|a, b| a.id().cmp(b.id()));
                self_records == other_records
            }
    }
}

impl Display for Sequences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for record in &self.s {
            write!(f, "{}", record)?;
        }
        Ok(())
    }
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
        let potential_msa_len = if s.is_empty() { 0 } else { s[0].seq().len() };
        // Sequences are aligned if all sequences are the same length
        let aligned = s.iter().skip(1).all(|r| r.seq().len() == potential_msa_len);
        Sequences {
            s,
            aligned,
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

    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    /// Removes all gaps from the sequences and returns a new Sequences object.
    pub fn into_gapless(&self) -> Sequences {
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
            alphabet: self.alphabet.clone(),
        }
    }

    /// Creates a the character encoding for each given ungapped sequence.
    /// Used for the likelihood calculation to avoid having to get the character encoding
    /// from scratch every time the likelihood is optimised.
    pub(crate) fn generate_leaf_encoding(&self) -> HashMap<String, DMatrix<f64>> {
        let alphabet = self.alphabet();
        let mut leaf_encoding = HashMap::with_capacity(self.len());
        for seq in self.iter() {
            if seq.seq().is_empty() {
                leaf_encoding.insert(seq.id().to_string(), DMatrix::zeros(0, 0));
                continue;
            }
            leaf_encoding.insert(
                seq.id().to_string(),
                DMatrix::from_columns(
                    seq.seq()
                        .iter()
                        .map(|&c| alphabet.char_encoding(c))
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            );
        }
        leaf_encoding
    }
}
