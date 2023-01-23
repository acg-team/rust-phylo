use bio::alignment::distance::*;
use bio::alphabets::{self, Alphabet};
use bio::io::fasta;
use nalgebra::{max, DMatrix};

use super::njmat;

pub(crate) fn get_sequence_type(sequences: &Vec<fasta::Record>) -> Alphabet {
    let mut dna_alphabet = alphabets::dna::n_alphabet();
    dna_alphabet.insert('-' as u8);
    for record in sequences {
        if !dna_alphabet.is_word(record.seq()) {
            let mut protein_alphabet = alphabets::protein::alphabet();
            protein_alphabet.insert('-' as u8);
            return protein_alphabet;
        }
    }
    dna_alphabet
}

pub(crate) fn compute_distance_matrix(sequences: &Vec<fasta::Record>) -> njmat::NJMat {
    let nseqs = sequences.len();
    let mut distances = DMatrix::<f32>::zeros(nseqs, nseqs);
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
