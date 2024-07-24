use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPLikelihoodCost, PIPModel};
use crate::substitution_models::SubstitutionModel;
use crate::tree::{Node, NodeIdx, Tree};
use crate::Result;

pub(crate) struct SingleBranchOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) model: &'a PIPModel<SubstModel>,
    pub(crate) phylo_info: &'a PhyloInfo,
    pub(crate) branch: NodeIdx,
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
        let mut lc = PIPLikelihoodCost {
            model: self.model,
            info: self.phylo_info.clone(),
        };
        lc.info.tree.set_branch_length(self.branch, *value);
        Ok(-lc.compute_log_likelihood().0)
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

        let mut lc = PIPLikelihoodCost {
            model: self.model,
            info: self.phylo_info.clone(),
        };

        let init_logl = lc.compute_log_likelihood().0;
        info!("Initial logl: {}.", init_logl);
        let mut opt_logl = init_logl;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;

        let nodes = lc
            .info
            .tree
            .leaves
            .clone()
            .into_iter()
            .chain(lc.info.tree.internals.clone())
            .collect::<Vec<Node>>();
        while (prev_logl - opt_logl).abs() > self.epsilon {
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for branch in &nodes {
                let result = self.fun_name(&lc, branch)?;
                opt_logl = result.1;
                lc.info.tree.set_branch_length(branch.idx, result.0);
                debug!(
                    "Optimised branch length {} to value {} with logl {}",
                    branch, result.0, opt_logl
                );
            }
            iters += 1;
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, lc.info.tree.clone(), init_logl, opt_logl))
    }

    fn fun_name(&self, lc: &PIPLikelihoodCost<SubstModel>, branch: &Node) -> Result<(f64, f64)> {
        let optimiser = SingleBranchOptimiser {
            model: self.model,
            phylo_info: &lc.info,
            branch: branch.idx,
        };
        let gss = BrentOpt::new(1e-10, 100.0);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(lc.info.tree.get_branch_length(branch.idx)))
            .run()?;
        let state = res.state();
        Ok((state.best_param.unwrap(), -state.best_cost))
    }
}
