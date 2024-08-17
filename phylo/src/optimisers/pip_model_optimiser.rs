use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::evolutionary_models::{EvoModel, EvoModelParams};
use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPCost, PIPDNAModel, PIPDNAParams, PIPModel};
use crate::substitution_models::dna_models::{DNAParameter, DNASubstModel};
use crate::Result;

pub(crate) struct PIPDNAParamOptimiser<'a> {
    pub(crate) likelihood_cost: &'a PIPCost<'a, DNASubstModel>,
    pub(crate) model: &'a PIPModel<DNASubstModel>,
    pub(crate) phylo_info: &'a PhyloInfo,
    pub(crate) parameter: &'a [DNAParameter],
}

impl CostFunction for PIPDNAParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let mut params = self.model.params().clone();
        for param_name in self.parameter {
            params.set_value(param_name, *value);
        }
        let mut likelihood_cost: PIPCost<DNASubstModel> = self.likelihood_cost.clone();

        let model = PIPDNAModel::create(&params);

        likelihood_cost.model = &model;
        Ok(-likelihood_cost.logl(self.phylo_info).0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct PIPDNAModelOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood_cost: &'a PIPCost<'a, DNASubstModel>,
    pub(crate) info: PhyloInfo,
}

impl<'a> PIPDNAModelOptimiser<'a> {
    pub fn new(cost: &'a PIPCost<'a, DNASubstModel>, phylo_info: &PhyloInfo) -> Self {
        PIPDNAModelOptimiser {
            epsilon: 1e-3,
            likelihood_cost: cost,
            info: phylo_info.clone(),
        }
    }

    pub fn optimise_parameters(&self) -> Result<(u32, PIPDNAParams, f64)> {
        let mut opt_params = self.likelihood_cost.model.params.clone();
        let model_type = opt_params.model_type;
        info!("Optimising PIP with {} parameters.", model_type);

        let param_sets = &PIPDNAParams::parameter_definition(&model_type);

        let mut opt_logl = self.likelihood_cost.logl(&self.info).0;
        info!("Initial logl: {}.", opt_logl);
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;
        let mut model = PIPModel::create(&opt_params);

        while (prev_logl - opt_logl).abs() > self.epsilon {
            iters += 1;
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = PIPDNAParamOptimiser {
                    likelihood_cost: self.likelihood_cost,
                    model: &model,
                    parameter: param_set,
                    phylo_info: &self.info,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(opt_params.value(param_set.first().unwrap()))
                    })
                    .run()?;
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    opt_params.set_value(param_id, value);
                }
                opt_logl = -res.state().best_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, opt_logl
                );
                debug!("New parameters: {}\n", opt_params);
                model = PIPModel::create(&opt_params);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, opt_params, opt_logl))
    }
}
