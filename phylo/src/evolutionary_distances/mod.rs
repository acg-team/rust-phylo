use bio::{alignment::distance::levenshtein, io::fasta::Record};
use nalgebra::max;

pub trait EvolutionaryDistance {
    fn dist(&self, a: &Record, b: &Record) -> f64;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
/// Levenshtein distance with Jukes-Cantor correction, meaningful for DNA sequences
pub struct LevenshteinDNACorrected;

impl EvolutionaryDistance for LevenshteinDNACorrected {
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        // Distance formula corrected using the Jukes-Cantor model
        let seq_i = a.seq();
        let seq_j = b.seq();
        // To avoid infinite distance when all characters are different, the maximum
        // proportion of different characters is capped to 3/4=0.75
        let max_proportion = 3.0 / 4.0;
        let dist = levenshtein(seq_i, seq_j) as f64;
        let proportion_diff = f64::min(
            dist / (max(seq_i.len(), seq_j.len()) as f64),
            max_proportion - f64::EPSILON,
        );
        -max_proportion * (1.0 - 1.0 / max_proportion * proportion_diff).ln()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
/// Levenshtein distance with Jukes-Cantor-like correction, meaningful for protein sequences
pub struct LevenshteinProteinCorrected;

impl EvolutionaryDistance for LevenshteinProteinCorrected {
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        // Distance formula corrected using the The Poisson model, equivalent to Jukes-Cantor for proteins
        let seq_i = a.seq();
        let seq_j = b.seq();
        // To avoid infinite distance when all characters are different, the maximum
        // proportion of different characters is capped to 19/20=0.95
        let max_proportion = 19.0 / 20.0;
        let dist = levenshtein(seq_i, seq_j) as f64;
        let proportion_diff = f64::min(
            dist / (max(seq_i.len(), seq_j.len()) as f64),
            max_proportion - f64::EPSILON,
        );
        -max_proportion * (1.0 - 1.0 / max_proportion * proportion_diff).ln()
    }
}
