use std::ops::Div;

use anyhow::bail;
use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::goldensectionsearch::GoldenSectionSearch;
use log::{debug, info};

use crate::substitution_models::{
    dna_models::{make_dna_model, make_pi, DNASubstModel, DNASubstParams, ParamEnum},
    SubstMatrix, SubstitutionLikelihoodCost,
};
use crate::Result;

pub fn gtr(gtr_params: DNASubstParams) -> DNASubstModel {
    info!("Setting up GTR with rates: {}", gtr_params.print_as_gtr());
    let q = gtr_q(&gtr_params);
    make_dna_model(gtr_params, q)
}

pub fn parse_gtr_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
    if model_params.len() != 10 {
        bail!(
            "{} parameters for the GTR model, expected 10, got {}",
            if model_params.len() < 10 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let pi = make_pi(&[
        model_params[0],
        model_params[1],
        model_params[2],
        model_params[3],
    ])?;
    let gtr_params = DNASubstParams {
        pi,
        rtc: model_params[4],
        rta: model_params[5],
        rtg: model_params[6],
        rca: model_params[7],
        rcg: model_params[8],
        rag: model_params[9],
    };

    Ok(gtr_params)
}

fn gtr_q(gtr: &DNASubstParams) -> SubstMatrix {
    let ft = gtr.pi[0];
    let fc = gtr.pi[1];
    let fa = gtr.pi[2];
    let fg = gtr.pi[3];
    let total = (gtr.rtc * fc + gtr.rta * fa + gtr.rtg * fg) * ft
        + (gtr.rtc * ft + gtr.rca * fa + gtr.rcg * fg) * fc
        + (gtr.rta * ft + gtr.rca * fc + gtr.rag * fg) * fa
        + (gtr.rtg * ft + gtr.rcg * fc + gtr.rag * fa) * fg;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -(gtr.rtc * fc + gtr.rta * fa + gtr.rtg * fg),
            gtr.rtc * fc,
            gtr.rta * fa,
            gtr.rtg * fg,
            gtr.rtc * ft,
            -(gtr.rtc * ft + gtr.rca * fa + gtr.rcg * fg),
            gtr.rca * fa,
            gtr.rcg * fg,
            gtr.rta * ft,
            gtr.rca * fc,
            -(gtr.rta * ft + gtr.rca * fc + gtr.rag * fg),
            gtr.rag * fg,
            gtr.rtg * ft,
            gtr.rcg * fc,
            gtr.rag * fa,
            -(gtr.rtg * ft + gtr.rcg * fc + gtr.rag * fa),
        ],
    )
    .div(total)
}

struct GtrParamOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    params: DNASubstParams,
    parameter: ParamEnum,
}

impl CostFunction for GtrParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut gtr_params = self.params.clone();
        set_param(&mut gtr_params, &self.parameter, *value)?;
        let model = gtr(gtr_params);
        Ok(-self.likelihood_cost.compute_log_likelihood(&model).0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub trait DNAModelOptimiser<'a> {
    fn new(likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>) -> Self;
    fn optimise_parameters(&self, params: &DNASubstParams) -> Result<(u32, DNASubstParams, f64)>;
}

pub struct GTRModelOptimiser<'a> {
    epsilon: f64,
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
}

impl<'a> DNAModelOptimiser<'a> for GTRModelOptimiser<'a> {
    fn new(likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>) -> Self {
        GTRModelOptimiser {
            epsilon: 1e-3,
            likelihood_cost,
        }
    }

    fn optimise_parameters(
        &self,
        start_params: &DNASubstParams,
    ) -> Result<(u32, DNASubstParams, f64)> {
        println!("Optimising GTR parameters.");
        let params_to_optimise = [
            ParamEnum::Rca,
            ParamEnum::Rcg,
            ParamEnum::Rta,
            ParamEnum::Rtc,
            ParamEnum::Rtg,
        ];
        let mut opt_logl = self
            .likelihood_cost
            .compute_log_likelihood(&gtr(start_params.clone()))
            .0;
        info!("Initial logl: {}.", opt_logl);
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;
        let mut opt_params = start_params.clone();

        while (prev_logl - opt_logl).abs() > self.epsilon {
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for param_name in params_to_optimise.iter() {
                let optimiser = GtrParamOptimiser {
                    likelihood_cost: self.likelihood_cost,
                    params: opt_params.clone(),
                    parameter: *param_name,
                };
                let gss = GoldenSectionSearch::new(1e-5, 100.0)?.with_tolerance(self.epsilon)?;
                let res = Executor::new(optimiser, gss)
                    .configure(|_| IterState::new().param(opt_params.get_value(param_name)))
                    .run()?;
                let value = res.state().best_param.unwrap();
                opt_params.set_value(param_name, value);
                opt_logl = -res.state().best_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, opt_logl
                );
                debug!("New parameters: {}\n", opt_params.print_as_gtr());
            }
            iters += 1;
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, opt_params, opt_logl))
    }
}

fn set_param(gtr_params: &mut DNASubstParams, param_name: &ParamEnum, value: f64) -> Result<()> {
    match param_name {
        ParamEnum::Rtc => gtr_params.rtc = value,
        ParamEnum::Rta => gtr_params.rta = value,
        ParamEnum::Rtg => gtr_params.rtg = value,
        ParamEnum::Rca => gtr_params.rca = value,
        ParamEnum::Rcg => gtr_params.rcg = value,
        ParamEnum::Rag => gtr_params.rag = value,
        _ => bail!("Cannot optimise frequencies for now."),
    }
    Ok(())
}
