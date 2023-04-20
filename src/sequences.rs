use bio::alignment::distance::*;
use bio::alphabets::Alphabet;
use bio::io::fasta;
use nalgebra::{max, DMatrix};

use crate::tree::NodeIdx;

use super::njmat;

pub(crate) mod parsimony_sets;

#[derive(PartialEq, Debug)]
pub(crate) enum SequenceType {
    DNA,
    Protein,
}

fn charify(chars: &str) -> Vec<u8> {
    chars
        .chars()
        .map(|c| c as u8)
        .collect()
}

static AMINOACIDS_STR: &str = "ARNDCQEGHILKMFPSTWYV";
static AMB_AMINOACIDS_STR: &str = "BJZX";

static NUCLEOTIDES_STR: &str = "TCAG";
static AMB_NUCLEOTIDES_STR: &str = "RYSWKMBDHVNZX";

static GAP: u8 = b'-';

pub(crate) fn dna_alphabet() -> Alphabet {
    let mut nucleotides = charify(NUCLEOTIDES_STR);
    nucleotides.append(&mut (charify(AMB_NUCLEOTIDES_STR)));
    nucleotides.append(&mut nucleotides.clone().to_ascii_lowercase());
    nucleotides.push(GAP);
    Alphabet::new(nucleotides)
}

#[allow(dead_code)]
pub(crate) fn protein_alphabet() -> Alphabet {
    let mut aminoacids = charify(AMINOACIDS_STR);
    aminoacids.append(&mut (charify(AMB_AMINOACIDS_STR)));
    aminoacids.append(&mut aminoacids.clone().to_ascii_lowercase());
    aminoacids.push(GAP);
    Alphabet::new(aminoacids)
}

pub(crate) fn get_sequence_type(sequences: &Vec<fasta::Record>) -> SequenceType {
    let dna_alphabet = dna_alphabet();
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return SequenceType::Protein;
        }
    }
    SequenceType::DNA
}

pub(crate) fn compute_distance_matrix(sequences: &Vec<fasta::Record>) -> njmat::NJMat {
    let nseqs = sequences.len();
    let mut distances = DMatrix::zeros(nseqs, nseqs);
    for i in 0..nseqs {
        for j in (i + 1)..nseqs {
            let lev_dist = levenshtein(sequences[i].seq(), sequences[j].seq()) as f32;
            let proportion_diff = f32::min(
                lev_dist / (max(sequences[i].seq().len(), sequences[j].seq().len()) as f32),
                0.75 - f32::EPSILON,
            );
            let corrected_dist = -3.0 / 4.0 * (1.0 - 4.0 / 3.0 * proportion_diff).ln();
            distances[(i, j)] = corrected_dist;
            distances[(j, i)] = corrected_dist;
        }
    }
    let nj_distances = njmat::NJMat {
        idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
        distances,
    };
    nj_distances
}

pub(crate) fn get_parsimony_sets(record: &fasta::Record, sequence_type: &SequenceType) -> Vec<u32> {
    let set_table = match sequence_type {
        SequenceType::DNA => parsimony_sets::dna_pars_sets(),
        SequenceType::Protein => parsimony_sets::protein_pars_sets(),
    };
    record
        .seq()
        .into_iter()
        .map(|c| set_table[*c as usize])
        .collect()
}

#[cfg(test)]
mod sequences_tests {
    use super::{dna_alphabet, protein_alphabet};
    use bio::alphabets::Alphabet;

    #[test]
    fn alphabets() {
        assert_eq!(
            dna_alphabet(),
            Alphabet::new(b"ACGTRYSWKMBDHVNZXacgtryswkmbdhvnzx-")
        );
        assert_eq!(
            protein_alphabet(),
            Alphabet::new(b"ABCDEFGHIJKLMNPQRSTVWXYZabcdefghijklmnpqrstvwxyz-")
        );
    }
}
