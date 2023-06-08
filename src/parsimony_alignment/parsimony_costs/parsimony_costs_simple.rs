use super::BranchParsimonyCosts;
use super::ParsimonyCosts;

pub(crate) struct ParsimonyCostsSimple {
    costs: BranchParsimonyCostsSimple,
}

impl ParsimonyCostsSimple {
    pub(crate) fn new_default() -> ParsimonyCostsSimple {
        Self::new(1.0, 2.5, 0.5)
    }

    pub(crate) fn new(mismatch: f64, gap_open: f64, gap_ext: f64) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            costs: BranchParsimonyCostsSimple {
                mismatch: mismatch,
                gap_open: gap_open,
                gap_ext: gap_ext,
            },
        }
    }
}

impl ParsimonyCosts for ParsimonyCostsSimple {
    fn get_branch_costs(&self, _: f64) -> Box<&dyn BranchParsimonyCosts> {
        Box::new(&self.costs)
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
