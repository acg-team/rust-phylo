use super::BranchParsimonyCosts;
use super::ParsimonyCosts;

pub(crate) struct ParsimonyCostsSimple {
    costs: BranchParsimonyCostsSimple,
}

#[allow(dead_code)]
impl ParsimonyCostsSimple {
    pub(crate) fn new_default() -> ParsimonyCostsSimple {
        Self::new(1.0, 2.5, 0.5)
    }

    pub(crate) fn new(mismatch: f64, gap_open: f64, gap_ext: f64) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            costs: BranchParsimonyCostsSimple {
                mismatch: mismatch,
                gap_open: gap_open * mismatch,
                gap_ext: gap_ext * mismatch,
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

    fn avg_cost(&self) -> f64 {
        self.mismatch
    }
}

#[cfg(test)]
mod parsimony_costs_simple_test {
    use super::ParsimonyCostsSimple;
    use crate::parsimony_alignment::parsimony_costs::BranchParsimonyCosts;

    #[test]
    fn default_costs() {
        let default_costs = ParsimonyCostsSimple::new_default();
        assert_eq!(default_costs.costs.match_cost(b'A', b'B'), 1.0);
        assert_eq!(default_costs.costs.match_cost(b'B', b'A'), 1.0);
        assert_eq!(default_costs.costs.match_cost(b'N', b'K'), 1.0);
        assert_eq!(default_costs.costs.match_cost(b'C', b'C'), 0.0);
        assert_eq!(default_costs.costs.avg_cost(), 1.0);
        assert_eq!(default_costs.costs.gap_ext_cost(), 0.5);
        assert_eq!(default_costs.costs.gap_open_cost(), 2.5);
    }

    #[test]
    fn simple_costs() {
        let mismatch = 3.0;
        let gap_open = 2.0;
        let gap_ext = 10.5;
        let costs = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext);
        assert_eq!(costs.costs.match_cost(b'A', b'B'), mismatch);
        assert_eq!(costs.costs.match_cost(b'A', b'A'), 0.0);
        assert_eq!(costs.costs.avg_cost(), mismatch);
        assert_eq!(costs.costs.gap_open_cost(), gap_open * mismatch);
        assert_eq!(costs.costs.gap_ext_cost(), gap_ext * mismatch);
        let mismatch = 1.0;
        let gap_open = 10.0;
        let gap_ext = 2.5;
        let costs = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext);
        assert_eq!(costs.costs.match_cost(b'A', b'B'), mismatch);
        assert_eq!(costs.costs.match_cost(b'A', b'A'), 0.0);
        assert_eq!(costs.costs.avg_cost(), mismatch);
        assert_eq!(costs.costs.gap_open_cost(), gap_open);
        assert_eq!(costs.costs.gap_ext_cost(), gap_ext);
    }
}
