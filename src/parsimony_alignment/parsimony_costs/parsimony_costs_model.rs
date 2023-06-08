use std::collections::HashMap;

use crate::{
    cmp_f64, f64_h,
    substitution_models::{
        dna_models, protein_models, DNASubstModel, ProteinSubstModel, SubstitutionModel,
    },
    Result,
};
use nalgebra::{Const, DimMin, SMatrix};
use ordered_float::OrderedFloat;

use super::{BranchParsimonyCosts, ParsimonyCosts};
type CostMatrix<const N: usize> = SMatrix<f64, N, N>;

pub(crate) struct ParsimonyCostsWModel<const N: usize> {
    times: Vec<f64>,
    costs: HashMap<OrderedFloat<f64>, BranchCostsWModel<N>>,
}

type DNAParsCosts = ParsimonyCostsWModel<4>;
type ProteinParsCosts = ParsimonyCostsWModel<20>;

impl DNAParsCosts {
    pub(crate) fn new(
        model_name: &str,
        gap_open_mult: f64,
        gap_ext_mult: f64,
        times: &[f64],
    ) -> Result<Self> {
        let model = DNASubstModel::new(model_name)?;
        let costs = generate_costs(
            model,
            times,
            gap_open_mult,
            gap_ext_mult,
            dna_models::nucleotide_index(),
        );
        let mut sorted_times = Vec::from(times);
        sorted_times.sort_by(cmp_f64());
        Ok(DNAParsCosts {
            times: sorted_times,
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
    ) -> Result<Self> {
        let model = ProteinSubstModel::new(model_name)?;
        let costs = generate_costs(
            model,
            times,
            gap_open_mult,
            gap_ext_mult,
            protein_models::aminoacid_index(),
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
) -> HashMap<OrderedFloat<f64>, BranchCostsWModel<N>>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let costs = model
        .generate_scorings(times, true)
        .into_iter()
        .map(|(key, (branch_costs, avg_cost))| {
            (
                key,
                BranchCostsWModel {
                    index,
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
        *self
            .times
            .iter()
            .filter(|&number| *number <= target)
            .last()
            .unwrap_or(&self.times[0])
    }
}

impl<const N: usize> ParsimonyCosts for ParsimonyCostsWModel<N> {
    fn get_branch_costs(&self, branch_length: f64) -> Box<&dyn BranchParsimonyCosts> {
        Box::new(&self.costs[&f64_h::from(self.find_closest_branch_length(branch_length))])
    }
}

pub(crate) struct BranchCostsWModel<const N: usize> {
    index: [i32; 255],
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
}
