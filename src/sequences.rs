use bio::alignment::distance::*;
use bio::alphabets::Alphabet;
use bio::io::fasta;
use nalgebra::{max, DMatrix};

use super::njmat;

mod dna_pars_sets;

#[allow(dead_code)]
#[derive(PartialEq, Debug)]
pub(crate) enum SequenceType {
    DNA,
    Protein,
}

pub(crate) fn get_sequence_alphabet(sequences: &Vec<fasta::Record>) -> Alphabet {
    let dna_alphabet = dna_alphabet();
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            return protein_alphabet();
        }
    }
    dna_alphabet
}

pub(crate) fn dna_alphabet() -> Alphabet {
    Alphabet::new(b"ACGTRYSWKMBDHVNZXacgtryswkmbdhvnzx-")
}

pub(crate) fn protein_alphabet() -> Alphabet {
    Alphabet::new(b"ABCDEFGHIKLMNPQRSTVWXYZabcdefghiklmnpqrstvwxyz-")
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
    println!("{:?}", distances);
    let nj_distances = njmat::NJMat {
        idx: (0..nseqs).collect(),
        distances,
    };
    nj_distances
}

pub(crate) fn get_parsimony_sets(
    record: &fasta::Record,
    sequence_type: &SequenceType,
) -> Vec<u8> {
    let sequence_length = record.seq().len();
    let set_table = match sequence_type {
        SequenceType::DNA => dna_pars_sets::dna_pars_sets(),
        SequenceType::Protein => [0b11110 as u8; 256],
    };
    let mut parsimony_sets = Vec::<u8>::with_capacity(sequence_length);
    for s in record.seq() {
        parsimony_sets.push(set_table[*s as usize]);
    }
    parsimony_sets
}
