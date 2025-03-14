use crate::alphabets::Alphabet;
use crate::parsimony::{GapMultipliers, ParsimonyCosts};

pub(crate) struct ParsimonyCostsSimple {
    mismatch: f64,
    gap_open: f64,
    gap_ext: f64,
    pub(crate) alphabet: Alphabet,
}

impl ParsimonyCostsSimple {
    pub fn new(
        mismatch: f64,
        gap_mult: GapMultipliers,
        alphabet: &Alphabet,
    ) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            mismatch,
            gap_open: gap_mult.open * mismatch,
            gap_ext: gap_mult.ext * mismatch,
            alphabet: alphabet.clone(),
        }
    }
}

impl ParsimonyCosts for ParsimonyCostsSimple {
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    fn r#match(&self, _: f64, char_i: &u8, char_j: &u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.mismatch
        }
    }
    fn gap_ext(&self, _: f64) -> f64 {
        self.gap_ext
    }

    fn gap_open(&self, _: f64) -> f64 {
        self.gap_open
    }

    fn avg(&self, _: f64) -> f64 {
        self.mismatch
    }
}
