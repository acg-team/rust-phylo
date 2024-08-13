use crate::alphabets::GAP;

use bio::io::fasta::Record;

#[derive(Debug, Clone, PartialEq)]
pub struct Sequences {
    pub(crate) s: Vec<Record>,
    pub(crate) aligned: bool,
    pub(crate) msa_len: usize,
}

impl Sequences {
    pub fn new(s: Vec<Record>) -> Sequences {
        let len = if s.is_empty() { 0 } else { s[0].seq().len() };
        if s.iter().filter(|rec| rec.seq().len() != len).count() == 0 {
            Sequences {
                s,
                aligned: true,
                msa_len: len,
            }
        } else {
            Sequences {
                s,
                aligned: false,
                msa_len: 0,
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

    pub fn without_gaps(&self) -> Sequences {
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
        Sequences::new(seqs)
    }
}
