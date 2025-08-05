use bio::{alignment::distance::levenshtein, io::fasta::Record};
use nalgebra::max;

pub trait EvolutionaryDistance {
    fn dist(&self, a: &Record, b: &Record) -> f64;
}

#[derive(Default)]
pub struct LevenshteinDNACorrected;

// Distance function to be used for default NJ Builder impl
impl EvolutionaryDistance for LevenshteinDNACorrected {
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        let seq_i = a.seq();
        let seq_j = b.seq();
        let dist = levenshtein(seq_i, seq_j) as f64;
        let proportion_diff = f64::min(
            dist / (max(seq_i.len(), seq_j.len()) as f64),
            0.75 - f64::EPSILON,
        );
        // TODO: Doesn't work for proteins! @junniest
        -3.0 / 4.0 * (1.0 - 4.0 / 3.0 * proportion_diff).ln()
    }
}
