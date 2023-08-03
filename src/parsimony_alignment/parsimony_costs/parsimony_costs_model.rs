use std::collections::HashMap;

use crate::{
    cmp_f64, f64_h,
    substitution_models::{
        dna_models, protein_models, DNASubstModel, ProteinSubstModel, SubstitutionModel,
    },
    Result,
};
use log::info;
use nalgebra::{Const, DimMin, SMatrix};
use ordered_float::OrderedFloat;

use super::{BranchParsimonyCosts, ParsimonyCosts};
type CostMatrix<const N: usize> = SMatrix<f64, N, N>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsimonyCostsWModel<const N: usize> {
    times: Vec<f64>,
    costs: HashMap<OrderedFloat<f64>, BranchCostsWModel<N>>,
}

pub(crate) type DNAParsCosts = ParsimonyCostsWModel<4>;
pub(crate) type ProteinParsCosts = ParsimonyCostsWModel<20>;

impl DNAParsCosts {
    pub(crate) fn new(
        model_name: &str,
        model_params: &[f64],
        gap_open_mult: f64,
        gap_ext_mult: f64,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> Result<Self> {
        info!(
            "Setting up the parsimony scoring from the {} substitution model.",
            model_name
        );
        info!(
            "The scoring matrix diagonals will {}be set to zero.",
            if zero_diag { "" } else { "not " }
        );
        info!(
            "The scoring matrix entries will {}be rounded to the closest integer value.",
            if rounded { "" } else { "not " }
        );
        let model = DNASubstModel::new(model_name, model_params)?;
        let costs = generate_costs(
            model,
            times,
            gap_open_mult,
            gap_ext_mult,
            dna_models::nucleotide_index(),
            zero_diag,
            rounded,
        );
        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            model_name, times
        );
        Ok(DNAParsCosts {
            times: sort_times(times),
            costs,
        })
    }
}

impl ProteinParsCosts {
    pub(crate) fn new(
        model_name: &str,
        gap_open_mult: f64,
        gap_ext_mult: f64,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> Result<Self> {
        info!(
            "Setting up the parsimony scoring from the {} substitution model.",
            model_name
        );
        let model = ProteinSubstModel::new(model_name)?;
        let costs = generate_costs(
            model,
            times,
            gap_open_mult,
            gap_ext_mult,
            protein_models::aminoacid_index(),
            zero_diag,
            rounded,
        );
        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            model_name, times
        );
        Ok(ProteinParsCosts {
            times: sort_times(times),
            costs,
        })
    }
}

fn generate_costs<const N: usize>(
    model: SubstitutionModel<N>,
    times: &[f64],
    gap_open_mult: f64,
    gap_ext_mult: f64,
    index: [i32; 255],
    zero_diag: bool,
    rounded: bool,
) -> HashMap<OrderedFloat<f64>, BranchCostsWModel<N>>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let costs = model
        .generate_scorings(times, zero_diag, rounded)
        .into_iter()
        .map(|(key, (branch_costs, avg_cost))| {
            (
                key,
                BranchCostsWModel {
                    index,
                    avg_cost,
                    gap_open: gap_open_mult * avg_cost,
                    gap_ext: gap_ext_mult * avg_cost,
                    costs: branch_costs,
                },
            )
        })
        .collect();
    costs
}

fn sort_times(times: &[f64]) -> Vec<f64> {
    let mut sorted_times = Vec::from(times);
    sorted_times.sort_by(cmp_f64());
    sorted_times
}

impl<const N: usize> ParsimonyCostsWModel<N> {
    fn find_closest_branch_length(&self, target: f64) -> f64 {
        match self
            .times
            .windows(2)
            .filter(|&window| target - window[0] > window[1] - target)
            .last() {
                Some(window) => window[1],
                None => self.times[0],
            }
    }
}

impl<const N: usize> ParsimonyCosts for ParsimonyCostsWModel<N> {
    fn get_branch_costs(&self, branch_length: f64) -> Box<&dyn BranchParsimonyCosts> {
        Box::new(&self.costs[&f64_h::from(self.find_closest_branch_length(branch_length))])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BranchCostsWModel<const N: usize> {
    index: [i32; 255],
    avg_cost: f64,
    gap_open: f64,
    gap_ext: f64,
    costs: CostMatrix<N>,
}

impl<const N: usize> BranchParsimonyCosts for BranchCostsWModel<N> {
    fn match_cost(&self, i: u8, j: u8) -> f64 {
        self.costs[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn gap_ext_cost(&self) -> f64 {
        self.gap_ext
    }

    fn gap_open_cost(&self) -> f64 {
        self.gap_open
    }

    fn avg_cost(&self) -> f64 {
        self.avg_cost
    }
}

#[cfg(test)]
mod parsimony_costs_model_test {
    use crate::{
        f64_h,
        parsimony_alignment::parsimony_costs::{
            parsimony_costs_model::{DNAParsCosts, ProteinParsCosts},
            ParsimonyCosts,
        },
        substitution_models::{protein_models, DNASubstModel, ProteinSubstModel},
    };

    use super::generate_costs;

    #[test]
    fn protein_branch_scoring() {
        let gap_open = 2.5;
        let gap_ext = 0.5;
        let avg_01 = 5.7675;
        let avg_07 = 4.0075;
        let times = [0.1, 0.7];
        let model = ProteinSubstModel::new("wag").unwrap();
        let costs = generate_costs(
            model,
            &times,
            gap_open,
            gap_ext,
            protein_models::aminoacid_index(),
            false,
            true,
        );
        let branch_costs = costs.get(&f64_h::from(0.1)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_01);
        assert_eq!(branch_costs.avg_cost, avg_01);
        assert_eq!(branch_costs.gap_ext, avg_01 * gap_ext);
        assert_eq!(branch_costs.gap_open, avg_01 * gap_open);
        let branch_costs = costs.get(&f64_h::from(0.7)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_07);
        assert_eq!(branch_costs.avg_cost, avg_07);

        let model = ProteinSubstModel::new("wag").unwrap();
        let costs = generate_costs(
            model,
            &times,
            gap_open,
            gap_ext,
            protein_models::aminoacid_index(),
            true,
            true,
        );
        let branch_costs = costs.get(&f64_h::from(0.1)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_01);
        assert_eq!(branch_costs.costs.diagonal().sum(), 0.0);
        let branch_costs = costs.get(&f64_h::from(0.7)).unwrap();
        assert_ne!(branch_costs.costs.mean(), avg_07);
        assert_eq!(branch_costs.costs.diagonal().sum(), 0.0);
    }

    #[test]
    fn protein_scoring() {
        let gap_open = 2.0;
        let gap_ext = 0.1;
        let avg_01 = 5.7675;
        let avg_03 = 4.7475;
        let avg_05 = 4.2825;
        let avg_07 = 4.0075;
        let times = [0.1, 0.3, 0.5, 0.7];
        let model = ProteinParsCosts::new("wag", gap_open, gap_ext, &times, false, true).unwrap();
        let branch_scores = model.get_branch_costs(0.1);
        assert_eq!(branch_scores.avg_cost(), avg_01);
        assert_eq!(branch_scores.gap_ext_cost(), avg_01 * gap_ext);
        assert_eq!(branch_scores.gap_open_cost(), avg_01 * gap_open);
        let branch_scores = model.get_branch_costs(0.3);
        assert_eq!(branch_scores.avg_cost(), avg_03);
        let branch_scores = model.get_branch_costs(0.5);
        assert_eq!(branch_scores.avg_cost(), avg_05);
        let branch_scores = model.get_branch_costs(0.7);
        assert_eq!(branch_scores.avg_cost(), avg_07);
    }

    #[test]
    fn protein_branch_scoring_nearest() {
        let gap_open = 2.0;
        let gap_ext = 0.1;
        let avg_01 = 5.7675;
        let avg_05 = 4.2825;
        let times = [0.1, 0.5];
        let model = ProteinParsCosts::new("wag", gap_open, gap_ext, &times, false, true).unwrap();
        let scores_01 = model.get_branch_costs(0.1);
        assert_eq!(scores_01.avg_cost(), avg_01);
        assert_eq!(scores_01.gap_ext_cost(), avg_01 * gap_ext);
        assert_eq!(scores_01.gap_open_cost(), avg_01 * gap_open);
        let scores_02 = model.get_branch_costs(0.2);
        assert_eq!(scores_01.avg_cost(), scores_02.avg_cost());
        assert_eq!(scores_01.gap_ext_cost(), scores_02.gap_ext_cost());
        assert_eq!(scores_01.gap_open_cost(), scores_02.gap_open_cost());
        let scores_005 = model.get_branch_costs(0.05);
        assert_eq!(scores_005.avg_cost(), avg_01);
        let scores_015 = model.get_branch_costs(0.15);
        assert_eq!(scores_015.avg_cost(), avg_01);
        let scores_045 = model.get_branch_costs(0.45);
        assert_eq!(scores_045.avg_cost(), avg_05);
        let scores_05 = model.get_branch_costs(0.5);
        assert_eq!(scores_05.avg_cost(), avg_05);
        let scores_100 = model.get_branch_costs(100.0);
        assert_eq!(scores_100.avg_cost(), avg_05);
    }

    #[test]
    fn dna_branch_scoring() {
        let gap_open = 2.5;
        let gap_ext = 0.5;
        let avg_01 = 2.25;
        let avg_07 = 1.75;
        let times = [0.1, 0.7];
        let model = DNASubstModel::new("jc69", &Vec::new()).unwrap();
        let costs = generate_costs(
            model,
            &times,
            gap_open,
            gap_ext,
            protein_models::aminoacid_index(),
            false,
            true,
        );
        let branch_costs = costs.get(&f64_h::from(0.1)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_01);
        assert_eq!(branch_costs.avg_cost, avg_01);
        assert_eq!(branch_costs.gap_ext, avg_01 * gap_ext);
        assert_eq!(branch_costs.gap_open, avg_01 * gap_open);
        let branch_costs = costs.get(&f64_h::from(0.7)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_07);
        assert_eq!(branch_costs.avg_cost, avg_07);
        let model = DNASubstModel::new("jc69", &Vec::new()).unwrap();
        let costs = generate_costs(
            model,
            &times,
            gap_open,
            gap_ext,
            protein_models::aminoacid_index(),
            true,
            true,
        );
        let branch_costs = costs.get(&f64_h::from(0.1)).unwrap();
        assert_eq!(branch_costs.costs.mean(), avg_01);
        assert_eq!(branch_costs.costs.diagonal().sum(), 0.0);
        let branch_costs = costs.get(&f64_h::from(0.7)).unwrap();
        assert_ne!(branch_costs.avg_cost, avg_07);
        assert_eq!(branch_costs.costs.diagonal().sum(), 0.0);
    }

    #[test]
    fn dna_branch_scoring_nearest() {
        let gap_open = 3.0;
        let gap_ext = 0.75;
        let avg_01 = 2.25;
        let avg_07 = 1.75;
        let times = [0.1, 0.7];
        let model =
            DNAParsCosts::new("jc69", &Vec::new(), gap_open, gap_ext, &times, false, true).unwrap();
        let scores_01 = model.get_branch_costs(0.1);
        assert_eq!(scores_01.avg_cost(), avg_01);
        assert_eq!(scores_01.gap_ext_cost(), avg_01 * gap_ext);
        assert_eq!(scores_01.gap_open_cost(), avg_01 * gap_open);
        let scores_08 = model.get_branch_costs(0.8);
        assert_eq!(scores_08.avg_cost(), avg_07);
        let scores_05 = model.get_branch_costs(0.5);
        assert_eq!(scores_05.avg_cost(), avg_07);
    }
}
