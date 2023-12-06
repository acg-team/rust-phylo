use std::fmt::Display;

use crate::substitution_models::FreqVector;

#[derive(Clone, Debug, PartialEq)]
pub struct DNASubstParams {
    pub(crate) pi: FreqVector,
    pub(crate) rtc: f64,
    pub(crate) rta: f64,
    pub(crate) rtg: f64,
    pub(crate) rca: f64,
    pub(crate) rcg: f64,
    pub(crate) rag: f64,
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
    pub(crate) fn print_as_jc69(&self) -> String {
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

    pub(crate) fn print_as_k80(&self) -> String {
        debug_assert!(
            self.rtc == self.rag
                && self.rta == self.rtg
                && self.rta == self.rca
                && self.rta == self.rcg
        );
        debug_assert_eq!(self.pi, FreqVector::from_column_slice(&[0.25; 4]));
        format!("[alpha = {}, beta = {}]", self.rtc, self.rta)
    }

    pub(crate) fn print_as_hky(&self) -> String {
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

    pub(crate) fn print_as_tn93(&self) -> String {
        debug_assert!(self.rta == self.rtg && self.rta == self.rca && self.rta == self.rcg);
        format!(
            "[pi = {:?}, alpha1 = {}, alpha2 = {}, beta = {}]",
            self.pi.as_slice(),
            self.rtc,
            self.rag,
            self.rta
        )
    }

    pub(crate) fn print_as_gtr(&self) -> String {
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
