use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPCost, PIPModel};
use crate::substitution_models::SubstitutionModel;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub(crate) struct SingleBranchOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) model: &'a PIPModel<SubstModel>,
    pub(crate) phylo_info: &'a PhyloInfo,
    pub(crate) branch: &'a NodeIdx,
}

impl<'a, SubstModel: SubstitutionModel + Clone> CostFunction
    for SingleBranchOptimiser<'a, SubstModel>
where
    SubstModel::ModelType: Clone,
    SubstModel::Params: Clone,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut lc = PIPCost {
            model: self.model,
            info: self.phylo_info.clone(),
        };
        lc.info.tree.set_branch_length(self.branch, *value);
        Ok(-lc.logl().0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct BranchOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) epsilon: f64,
    pub(crate) model: &'a PIPModel<SubstModel>,
    pub(crate) phylo_info: PhyloInfo,
}

impl<'a, SubstModel> BranchOptimiser<'a, SubstModel>
where
    SubstModel: SubstitutionModel + Clone,
    SubstModel::ModelType: Clone,
    SubstModel::Params: Clone,
{
    pub fn new(model: &'a PIPModel<SubstModel>, phylo_info: &PhyloInfo) -> Self {
        BranchOptimiser {
            epsilon: 1e-3,
            model,
            phylo_info: phylo_info.clone(),
        }
    }

    pub fn optimise_parameters(&self) -> Result<(u32, Tree, f64, f64)> {
        info!("Optimising branch lengths.");

        let mut lc = PIPCost {
            model: self.model,
            info: self.phylo_info.clone(),
        };

        let init_logl = lc.logl().0;
        info!("Initial logl: {}.", init_logl);
        let mut opt_logl = init_logl;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;

        let nodes: Vec<NodeIdx> = lc.info.tree.iter().map(|node| node.idx).collect();
        while (prev_logl - opt_logl).abs() > self.epsilon {
            iters += 1;
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for branch in &nodes {
                let (logl, length) = self.optimise_branch(&lc, branch)?;
                opt_logl = logl;
                lc.info.tree.set_branch_length(branch, length);
                debug!(
                    "Optimised branch length {} to value {} with logl {}",
                    branch, length, opt_logl
                );
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, lc.info.tree.clone(), init_logl, opt_logl))
    }

    fn optimise_branch(&self, lc: &PIPCost<SubstModel>, branch: &NodeIdx) -> Result<(f64, f64)> {
        let optimiser = SingleBranchOptimiser {
            model: self.model,
            phylo_info: &lc.info,
            branch,
        };
        let gss = BrentOpt::new(1e-10, 100.0);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(lc.info.tree.blen(branch)))
            .run()?;
        let state = res.state();
        Ok((-state.best_cost, state.best_param.unwrap()))
    }
}
