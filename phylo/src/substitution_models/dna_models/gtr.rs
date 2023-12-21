use std::ops::Div;

use anyhow::bail;
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;
use log::info;

use crate::evolutionary_models::EvolutionaryModelInfo;
use crate::substitution_models::{
    dna_models::{make_dna_model, make_pi, DNASubstModel, DNASubstParams},
    SubstMatrix, SubstitutionLikelihoodCost, SubstitutionModelInfo,
};
use crate::Result;

pub fn gtr(model_params: &[f64]) -> Result<DNASubstModel> {
    let gtr_params = parse_gtr_parameters(model_params)?;
    info!("Setting up gtr with rates: {}", gtr_params.print_as_gtr());
    let q = gtr_q(&gtr_params);
    Ok(make_dna_model(gtr_params, q))
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

impl DNASubstModel {
    pub(crate) fn reset_gtr(&mut self, params: &DNASubstParams) {
        self.params = ((*params).clone()).into();
        self.q = gtr_q(params);
    }
}

#[derive(Clone, Copy)]
enum ParamEnum {
    Pit,
    Pic,
    Pia,
    Pig,
    Rtc,
    Rta,
    Rtg,
    Rca,
    Rcg,
    Rag,
}

struct GTRParamOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    base_model: &'a DNASubstModel,
    parameter: ParamEnum,
}

impl CostFunction for GTRParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut params = parse_gtr_parameters(self.base_model.params.as_slice())?;
        match self.parameter {
            ParamEnum::Pit | ParamEnum::Pic | ParamEnum::Pia | ParamEnum::Pig => {
                bail!("Cannot optimise frequencies for now.")
            }
            ParamEnum::Rtc => params.rtc = *param,
            ParamEnum::Rta => params.rta = *param,
            ParamEnum::Rtg => params.rtg = *param,
            ParamEnum::Rca => params.rca = *param,
            ParamEnum::Rcg => params.rcg = *param,
            ParamEnum::Rag => params.rag = *param,
        }
        let mut model = self.base_model.clone();
        model.reset_gtr(&params);
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(-self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct GTRModelOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    base_model: &'a DNASubstModel,
}

impl<'a> GTRModelOptimiser<'a> {
    pub fn new(
        likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
        base_model: &'a DNASubstModel,
    ) -> Self {
        GTRModelOptimiser {
            likelihood_cost,
            base_model,
        }
    }

    pub fn optimise_parameters(&self) -> Result<(u32, DNASubstParams, f64)> {
        let epsilon = 1e-10;
        let params = self.base_model.params.clone();
        let model = gtr(params.as_slice())?;
        let mut logl = f64::NEG_INFINITY;
        let mut new_logl = 0.0;
        let mut gtr_params = parse_gtr_parameters(params.as_slice())?;
        let mut iters = 0;
        let params_to_optimise = [
            ParamEnum::Rag,
            ParamEnum::Rca,
            ParamEnum::Rcg,
            ParamEnum::Rta,
            ParamEnum::Rtc,
            ParamEnum::Rtg,
        ];
        while (logl - new_logl).abs() > epsilon {
            println!("Iteration: {}", iters);
            logl = new_logl;
            for param_name in params_to_optimise.iter() {
                let optimiser = GTRParamOptimiser {
                    likelihood_cost: self.likelihood_cost,
                    base_model: &model,
                    parameter: *param_name,
                };
                let res = Executor::new(optimiser, subst_param_brent()).run()?;
                let value = res.state().best_param.unwrap();
                match param_name {
                    ParamEnum::Pit | ParamEnum::Pic | ParamEnum::Pia | ParamEnum::Pig => {
                        bail!("Cannot optimise frequencies for now.")
                    }
                    ParamEnum::Rtc => gtr_params.rtc = value,
                    ParamEnum::Rta => gtr_params.rta = value,
                    ParamEnum::Rtg => gtr_params.rtg = value,
                    ParamEnum::Rca => gtr_params.rca = value,
                    ParamEnum::Rcg => gtr_params.rcg = value,
                    ParamEnum::Rag => gtr_params.rag = value,
                }
                new_logl = res.state().best_cost;
            }
            iters += 1;
        }
        Ok((iters, gtr_params, -logl))
    }
}

fn subst_param_brent() -> BrentOpt<f64> {
    BrentOpt::new(1e-10, 100000.0)
}

#[cfg(test)]
mod gtr_optimisation_tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;

    use crate::{
        evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo},
        phylo_info::phyloinfo_from_files,
        substitution_models::{
            dna_models::{gtr::GTRModelOptimiser, DNALikelihoodCost, DNASubstModel},
            SubstitutionModelInfo,
        },
    };

    #[test]
    fn check_parameter_optimisation_gtr() {
        // Original params from paml: 0.88892  0.03190  0.00001  0.07102  0.02418
        let info = phyloinfo_from_files(
            PathBuf::from("./data/sim/gtr/gtr.fasta"),
            PathBuf::from("./data/sim/tree.newick"),
        )
        .unwrap();
        let likelihood = DNALikelihoodCost { info: &info };
        let model = DNASubstModel::new(
            "gtr",
            &[
                0.24720, 0.35320, 0.29540, 0.10420, 10000.0, 311.84397, 1.00000, 772.75972,
                415.08690, 10000.0,
            ],
        )
        .unwrap(); // Optimized parameters from PhyML
        let mut tmp_info = SubstitutionModelInfo::new(likelihood.info, &model).unwrap();
        let unopt_logl = likelihood.compute_log_likelihood(&model, &mut tmp_info);
        assert_relative_eq!(unopt_logl, -3474.48083, epsilon = 1.0e-5);
        let (iters, params, logl) = GTRModelOptimiser::new(&likelihood, &model)
            .optimise_parameters()
            .unwrap();
        assert!(logl > unopt_logl);
    }
}
