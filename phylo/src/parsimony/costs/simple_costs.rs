use crate::parsimony::{GapCost, ParsimonyCosts};

pub(crate) struct SimpleCosts {
    mismatch: f64,
    gap: GapCost,
}

impl SimpleCosts {
    pub fn new(mismatch: f64, gap: GapCost) -> SimpleCosts {
        SimpleCosts {
            mismatch,
            gap: gap * mismatch,
        }
    }
}

impl ParsimonyCosts for SimpleCosts {
    fn r#match(&self, _: f64, char_i: &u8, char_j: &u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.mismatch
        }
    }
    fn gap_ext(&self, _: f64) -> f64 {
        self.gap.ext
    }

    fn gap_open(&self, _: f64) -> f64 {
        self.gap.open
    }

    fn avg(&self, _: f64) -> f64 {
        self.mismatch
    }
}
