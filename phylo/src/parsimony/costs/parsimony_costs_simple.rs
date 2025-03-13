use crate::alphabets::{dna_alphabet, Alphabet};
use crate::parsimony::{BranchParsimonyCosts, CostMatrix, ParsimonyCosts};

pub(crate) struct ParsimonyCostsSimple {
    pub(crate) branch: BranchParsimonyCostsSimple,
    pub(crate) alphabet: Alphabet,
}

#[allow(dead_code)]
impl ParsimonyCostsSimple {
    pub(crate) fn new_default() -> ParsimonyCostsSimple {
        Self::new(1.0, 2.5, 0.5, dna_alphabet())
    }

    pub(crate) fn new(
        mismatch: f64,
        gap_open: f64,
        gap_ext: f64,
        alphabet: Alphabet,
    ) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            branch: BranchParsimonyCostsSimple {
                mismatch,
                gap_open: gap_open * mismatch,
                gap_ext: gap_ext * mismatch,
            },
            alphabet,
        }
    }
}

impl ParsimonyCosts for ParsimonyCostsSimple {
    fn branch_costs(&self, _: f64) -> &dyn BranchParsimonyCosts {
        &self.branch
    }

    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    fn r#match(&self, _: f64, char_i: &u8, char_j: &u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.branch.mismatch
        }
    }
    fn gap_ext(&self, _: f64) -> f64 {
        self.branch.gap_ext
    }

    fn gap_open(&self, _: f64) -> f64 {
        self.branch.gap_open
    }

    fn avg(&self, _: f64) -> f64 {
        self.branch.mismatch
    }
}

pub(crate) struct BranchParsimonyCostsSimple {
    mismatch: f64,
    gap_open: f64,
    gap_ext: f64,
}

impl BranchParsimonyCosts for BranchParsimonyCostsSimple {
    fn r#match(&self, char_i: &u8, char_j: &u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.mismatch
        }
    }
    fn gap_ext(&self) -> f64 {
        self.gap_ext
    }

    fn gap_open(&self) -> f64 {
        self.gap_open
    }

    fn avg(&self) -> f64 {
        self.mismatch
    }

    fn cost_matrix(&self) -> &CostMatrix {
        unimplemented!()
    }
}
