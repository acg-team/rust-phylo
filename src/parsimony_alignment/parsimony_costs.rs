
// type CostMatrix<const N: usize> = SMatrix<f64, N, N>;


pub(crate) trait ParsimonyCosts {
    fn get_branch_costs(&self, branch_length: f64) -> Box<dyn BranchParsimonyCosts>;
}

pub trait BranchParsimonyCosts {
    fn match_cost(&self, i: u8, j: u8) -> f64;
    fn gap_ext_cost(&self) -> f64;
    fn gap_open_cost(&self) -> f64;
}

pub(crate) struct ParsimonyCostsSimple {
    mismatch: f64,
    gap_open: f64,
    gap_ext: f64,
}

impl ParsimonyCostsSimple {
    pub(crate) fn new_default() -> ParsimonyCostsSimple {
        Self::new(1.0, 2.5, 0.5)
    }

    pub(crate) fn new(mismatch: f64, gap_open: f64, gap_ext: f64) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            mismatch,
            gap_open,
            gap_ext,
        }
    }
}

impl ParsimonyCosts for ParsimonyCostsSimple {
    fn get_branch_costs(&self, _: f64) -> Box<dyn BranchParsimonyCosts> {
        Box::new(BranchParsimonyCostsSimple {
            mismatch: self.mismatch,
            gap_open: self.gap_open,
            gap_ext: self.gap_ext,
        })
    }
}

pub(crate) struct BranchParsimonyCostsSimple {
    mismatch: f64,
    gap_open: f64,
    gap_ext: f64,
}

impl BranchParsimonyCosts for BranchParsimonyCostsSimple {
    fn match_cost(&self, char_i: u8, char_j: u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.mismatch
        }
    }
    fn gap_ext_cost(&self) -> f64 {
        self.gap_ext
    }

    fn gap_open_cost(&self) -> f64 {
        self.gap_open
    }
}
