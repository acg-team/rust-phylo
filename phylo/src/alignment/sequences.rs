use std::fmt::Display;

use anyhow::bail;
use bio::io::fasta::Record;
use bitvec::vec::BitVec;

use crate::alphabets::{dna_alphabet, protein_alphabet, Alphabet, GAP};
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
            write!(f, "{record}")?;
        }
        Ok(())
    }
}

impl Sequences {
    fn detect_alphabet(sequences: &[Record]) -> Alphabet {
        let dna_alphabet = dna_alphabet();
        for record in sequences.iter() {
            if !dna_alphabet.is_word(record.seq()) {
                return protein_alphabet();
            }
        }
        dna_alphabet
    }

    /// Creates a new Sequences object from a vector of bio::io::fasta::Record.
    /// The Sequences object is considered aligned if all sequences have the same length.
    pub fn new(s: Vec<Record>) -> Sequences {
        let alphabet = Self::detect_alphabet(&s);
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

    /// Returns the number of sequences
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
            .unwrap_or_else(|| panic!("Sequence with id {id} not found"))
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
            alphabet: *self.alphabet(),
        }
    }

    /// Removes all columns that only contain gaps from the sequences.
    pub fn remove_gap_cols(&mut self) {
        assert!(
            self.aligned,
            "Cannot remove gap columns from unaligned sequences"
        );

        let mut gap_cols: BitVec = BitVec::repeat(true, self.s[0].seq().len());
        for rec in &self.s {
            let seq_gaps = rec.seq().iter().map(|&c| c == GAP).collect::<BitVec>();
            gap_cols &= seq_gaps;
        }

        let new_seqs = self.s.iter().map(|rec| {
            let seq: Vec<u8> = rec
                .seq()
                .iter()
                .enumerate()
                .filter(|(i, _)| !gap_cols[*i])
                .map(|(_, c)| *c)
                .collect();
            Record::with_attrs(rec.id(), rec.desc(), &seq)
        });
        self.s = new_seqs.collect();
    }
}

#[cfg(test)]
mod private_tests {
    use rstest::rstest;

    use std::path::PathBuf;

    use crate::io::read_sequences;

    use super::*;

    #[rstest]
    #[case::aligned("./data/sequences_DNA1.fasta")]
    #[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
    #[case::long("./data/sequences_long.fasta")]
    fn dna_type_test(#[case] input: &str) {
        let seqs = read_sequences(&PathBuf::from(input)).unwrap();
        let alphabet = Sequences::detect_alphabet(&seqs);
        assert_eq!(alphabet, dna_alphabet());
        assert!(format!("{alphabet}").contains("DNA"));
    }

    #[rstest]
    #[case("./data/sequences_protein1.fasta")]
    #[case("./data/sequences_protein2.fasta")]
    fn protein_type_test(#[case] input: &str) {
        let seqs = read_sequences(&PathBuf::from(input)).unwrap();
        let alphabet = Sequences::detect_alphabet(&seqs);
        assert_eq!(alphabet, protein_alphabet());
        assert!(format!("{alphabet}").contains("protein"));
    }
}
