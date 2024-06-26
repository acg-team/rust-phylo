use std::fmt::Display;

use anyhow::bail;
use log::warn;

use crate::substitution_models::dna_models::DNAModelType;
use crate::substitution_models::FreqVector;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Parameter {
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
    Mu,
    Lambda,
}
use Parameter::*;

#[derive(Clone, Debug, PartialEq)]
pub struct DNASubstParams {
    pub pi: FreqVector,
    pub rtc: f64,
    pub rta: f64,
    pub rtg: f64,
    pub rca: f64,
    pub rcg: f64,
    pub rag: f64,
}

impl DNASubstParams {
    pub fn new(
        pi: FreqVector,
        rtc: f64,
        rta: f64,
        rtg: f64,
        rca: f64,
        rcg: f64,
        rag: f64,
    ) -> Result<Self> {
        if pi.sum() != 1.0 {
            bail!("Frequencies must sum to 1.0.");
        }
        Ok(Self {
            pi,
            rtc,
            rta,
            rtg,
            rca,
            rcg,
            rag,
        })
    }

    pub fn get_value(&self, param_name: &Parameter) -> f64 {
        match param_name {
            Pit => self.pi[0],
            Pic => self.pi[1],
            Pia => self.pi[2],
            Pig => self.pi[3],
            Rtc => self.rtc,
            Rta => self.rta,
            Rtg => self.rtg,
            Rca => self.rca,
            Rcg => self.rcg,
            Rag => self.rag,
            _ => panic!("Invalid parameter name."),
        }
    }

    pub fn set_value(&mut self, param_name: &Parameter, value: f64) {
        match param_name {
            Pit | Pic | Pia | Pig => {
                warn!("Cannot set frequencies individually. Use set_pi() instead.")
            }
            Rtc => self.rtc = value,
            Rta => self.rta = value,
            Rtg => self.rtg = value,
            Rca => self.rca = value,
            Rcg => self.rcg = value,
            Rag => self.rag = value,
            _ => panic!("Invalid parameter name."),
        }
    }

    pub fn set_pi(&mut self, pi: FreqVector) {
        if pi.sum() != 1.0 {
            warn!("Frequencies must sum to 1.0, not setting values");
        } else {
            self.pi = pi;
        }
    }
}

impl Display for DNASubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[pi = {:?}, rtc = {}, rta = {}, rtg = {}, rca = {}, rcg = {}, rag = {}]",
            self.pi.as_slice(),
            self.rtc,
            self.rta,
            self.rtg,
            self.rca,
            self.rcg,
            self.rag
        )
    }
}

impl DNASubstParams {
    pub(crate) fn parameter_definition(
        model_type: DNAModelType,
    ) -> Vec<(&'static str, Vec<Parameter>)> {
        match model_type {
            DNAModelType::JC69 => vec![],
            DNAModelType::K80 => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::HKY => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::TN93 => vec![
                ("alpha1", vec![Rtc]),
                ("alpha2", vec![Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],

            DNAModelType::GTR => vec![
                ("rca", vec![Rca]),
                ("rcg", vec![Rcg]),
                ("rta", vec![Rta]),
                ("rtc", vec![Rtc]),
                ("rtg", vec![Rtg]),
            ],
        }
    }

    pub fn print_as_jc69(&self) -> String {
        debug_assert!(
            self.rtc == 1.0
                && self.rta == 1.0
                && self.rtg == 1.0
                && self.rca == 1.0
                && self.rcg == 1.0
                && self.rag == 1.0
        );
        debug_assert_eq!(self.pi, FreqVector::from_column_slice(&[0.25; 4]));
        format!("[lambda = {}]", self.rtc)
    }

    pub fn print_as_k80(&self) -> String {
        debug_assert!(
            self.rtc == self.rag
                && self.rta == self.rtg
                && self.rta == self.rca
                && self.rta == self.rcg
        );
        debug_assert_eq!(self.pi, FreqVector::from_column_slice(&[0.25; 4]));
        format!("[alpha = {}, beta = {}]", self.rtc, self.rta)
    }

    pub fn print_as_hky(&self) -> String {
        debug_assert!(
            self.rtc == self.rag
                && self.rta == self.rtg
                && self.rta == self.rca
                && self.rta == self.rcg
        );
        format!(
            "[pi = {:?}, alpha = {}, beta = {}]",
            self.pi.as_slice(),
            self.rtc,
            self.rta
        )
    }

    pub fn print_as_tn93(&self) -> String {
        debug_assert!(self.rta == self.rtg && self.rta == self.rca && self.rta == self.rcg);
        format!(
            "[pi = {:?}, alpha1 = {}, alpha2 = {}, beta = {}]",
            self.pi.as_slice(),
            self.rtc,
            self.rag,
            self.rta
        )
    }

    pub fn print_as_gtr(&self) -> String {
        format!("{}", self)
    }
}

impl From<DNASubstParams> for Vec<f64> {
    fn from(val: DNASubstParams) -> Self {
        vec![
            val.pi[0], val.pi[1], val.pi[2], val.pi[3], val.rtc, val.rta, val.rtg, val.rca,
            val.rcg, val.rag,
        ]
    }
}
