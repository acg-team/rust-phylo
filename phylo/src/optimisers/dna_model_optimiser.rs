use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::{
    EvoModelParams,
    FrequencyOptimisation::{self, *},
};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{ModelOptimisationResult, ModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{
    DNAParameter, DNASubstModel, DNASubstParams, SubstLikelihoodCost, SubstitutionModel,
};
use crate::Result;

pub type DNAOptimisationResult = ModelOptimisationResult<DNASubstModel>;

pub(crate) struct DNAParamOptimiser<'a> {
    pub(crate) likelihood: &'a SubstLikelihoodCost<'a, DNASubstModel>,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) model: &'a DNASubstModel,
    pub(crate) parameter: &'a [DNAParameter],
}

impl CostFunction for DNAParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut model = self.model.clone();
        for param_name in self.parameter {
            model.set_param(param_name, *value);
        }
        let mut likelihood = self.likelihood.clone();

        likelihood.model = &model;
        Ok(-likelihood.cost(self.info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct DNAModelOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood: &'a SubstLikelihoodCost<'a, DNASubstModel>,
    pub(crate) info: PhyloInfo,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<'a> ModelOptimiser<'a, SubstLikelihoodCost<'a, DNASubstModel>, DNASubstModel>
    for DNAModelOptimiser<'a>
{
    fn new(
        likelihood: &'a SubstLikelihoodCost<'a, DNASubstModel>,
        phylo_info: &PhyloInfo,
        freq_opt: FrequencyOptimisation,
    ) -> Self {
        DNAModelOptimiser {
            epsilon: 1e-3,
            freq_opt,
            likelihood,
            info: phylo_info.clone(),
        }
    }

    fn run(self) -> Result<DNAOptimisationResult> {
        let mut start_params = self.likelihood.model.params.clone();
        let model_type = start_params.model_type;
        info!("Optimising {} parameters.", model_type);
        let param_sets = self.likelihood.model.params.parameter_definition();
        match self.freq_opt {
            Fixed => {}
            Empirical => {
                info!("Seting stationary frequencies to empirical.");
                start_params.set_freqs(self.info.freqs());
            }
            Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                start_params.set_freqs(self.info.freqs());
            }
        }
        self.run_parameter_brent(&start_params, param_sets)
    }
}

impl<'a> DNAModelOptimiser<'a> {
    fn run_parameter_brent(
        &self,
        start_params: &DNASubstParams,
        param_sets: Vec<(&str, Vec<DNAParameter>)>,
    ) -> Result<DNAOptimisationResult> {
        let mut prev_logl = f64::NEG_INFINITY;
        let initial_logl = self.likelihood.cost(&self.info);
        info!("Initial logl: {}.", initial_logl);
        let mut final_logl = initial_logl;
        let mut opt_params = start_params.clone();
        let mut model = DNASubstModel::create(&opt_params);
        let mut iterations = 0;

        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = DNAParamOptimiser {
                    likelihood: self.likelihood,
                    model: &model,
                    info: &self.info,
                    parameter: param_set,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(opt_params.param(param_set.first().unwrap()))
                    })
                    .run()?;
                let logl = -res.state().best_cost;
                if logl < final_logl {
                    continue;
                }
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    opt_params.set_param(param_id, value);
                }
                final_logl = logl;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, final_logl
                );
                debug!("New parameters: {}\n", opt_params);
                model = DNASubstModel::create(&opt_params);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(DNAOptimisationResult {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}
