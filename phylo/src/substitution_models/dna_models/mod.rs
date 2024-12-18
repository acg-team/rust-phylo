use std::cmp::Ordering;
use std::fmt::Display;
use std::iter;

use approx::relative_eq;
use log::warn;

use crate::alphabets::{dna_alphabet, Alphabet, NUCLEOTIDE_INDEX};
use crate::frequencies;
use crate::substitution_models::{FreqVector, QMatrix, SubstMatrix};

const DNA_N: usize = 4;

fn verify_dna_freqs(freqs: &FreqVector) -> bool {
    freqs.len() == DNA_N && relative_eq!(freqs.sum().abs(), 1.0)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JC69 {
    freqs: FreqVector,
    q: SubstMatrix,
    alphabet: Alphabet,
}

impl QMatrix for JC69 {
    fn new(_: &[f64], _: &[f64]) -> Self {
        let r = 1.0 / 3.0;
        let q = SubstMatrix::from_row_slice(
            DNA_N,
            DNA_N,
            &[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0],
        );
        JC69 {
            freqs: frequencies!(&[1.0 / DNA_N as f64; DNA_N]),
            q,
            alphabet: dna_alphabet().clone(),
        }
    }
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, _: FreqVector) {}
    fn set_param(&mut self, _: usize, _: f64) {}
    fn params(&self) -> &[f64] {
        &[]
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn index(&self) -> &[usize; 255] {
        &NUCLEOTIDE_INDEX
    }
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
}

impl Display for JC69 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JC69")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct K80 {
    freqs: FreqVector,
    q: SubstMatrix,
    kappa: Vec<f64>,
    alphabet: Alphabet,
}

impl QMatrix for K80 {
    fn new(_: &[f64], params: &[f64]) -> Self {
        let kappa = match params.len().cmp(&1) {
            Ordering::Less => {
                warn!("Too few values provided for K80, required one value for kappa");
                warn!("Falling back to default value.");
                2.0
            }
            Ordering::Greater => {
                warn!("Too many values provided for K80, required one value for kappa.");
                warn!("Will only use the first value provided.");
                params[0]
            }
            Ordering::Equal => params[0],
        };
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        k80_q(&mut q, kappa);
        K80 {
            freqs: frequencies!(&[1.0 / DNA_N as f64; DNA_N]),
            q,
            kappa: vec![kappa],
            alphabet: dna_alphabet().clone(),
        }
    }
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, _: FreqVector) {}
    fn set_param(&mut self, _: usize, value: f64) {
        self.kappa[0] = value;
        k80_q(&mut self.q, value);
    }
    fn params(&self) -> &[f64] {
        &self.kappa
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn index(&self) -> &[usize; 255] {
        &NUCLEOTIDE_INDEX
    }
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
}

impl Display for K80 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "K80 with [kappa = {:.5}]", self.kappa[0])
    }
}

fn k80_q(q: &mut SubstMatrix, k: f64) {
    let scaler = 1.0 / (k * 0.25 + 0.5);
    q[(0, 0)] = -(k * 0.25 + 0.5);
    q[(0, 1)] = k * 0.25;
    q[(0, 2)] = 0.25;
    q[(0, 3)] = 0.25;

    q[(1, 0)] = k * 0.25;
    q[(1, 1)] = -(k * 0.25 + 0.5);
    q[(1, 2)] = 0.25;
    q[(1, 3)] = 0.25;

    q[(2, 0)] = 0.25;
    q[(2, 1)] = 0.25;
    q[(2, 2)] = -(0.5 + k * 0.25);
    q[(2, 3)] = k * 0.25;

    q[(3, 0)] = 0.25;
    q[(3, 1)] = 0.25;
    q[(3, 2)] = k * 0.25;
    q[(3, 3)] = -(0.5 + k * 0.25);
    q.scale_mut(scaler);
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct HKY {
    freqs: FreqVector,
    q: SubstMatrix,
    kappa: Vec<f64>,
    alphabet: Alphabet,
}

impl QMatrix for HKY {
    fn new(freqs: &[f64], params: &[f64]) -> Self {
        let provided_freqs = frequencies!(freqs);
        let freqs = if verify_dna_freqs(&provided_freqs) {
            provided_freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };
        let kappa = match params.len().cmp(&1) {
            Ordering::Less => {
                warn!("Too few values provided for HKY, required one value for kappa");
                warn!("Falling back to default value.");
                2.0
            }
            Ordering::Greater => {
                warn!("Too many values provided for HKY, required one value for kappa.");
                warn!("Will only use the first value provided.");
                params[0]
            }
            Ordering::Equal => params[0],
        };
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        hky_q(&mut q, &freqs, kappa);
        HKY {
            freqs,
            q,
            kappa: vec![kappa],
            alphabet: dna_alphabet().clone(),
        }
    }
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) {
        self.freqs = if verify_dna_freqs(&freqs) {
            freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };
        hky_q(&mut self.q, &self.freqs, self.kappa[0])
    }
    fn set_param(&mut self, _: usize, value: f64) {
        self.kappa[0] = value;
        hky_q(&mut self.q, &self.freqs, self.kappa[0])
    }
    fn params(&self) -> &[f64] {
        &self.kappa
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn index(&self) -> &[usize; 255] {
        &NUCLEOTIDE_INDEX
    }
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
}

fn hky_q(q: &mut SubstMatrix, pi: &FreqVector, k: f64) {
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let scaler = 1.0
        / ((k * fc + (fa + fg)) * ft
            + (k * ft + (fa + fg)) * fc
            + ((ft + fc) + k * fg) * fa
            + ((ft + fc) + k * fa) * fg);
    q[(0, 0)] = -(k * fc + (fa + fg));
    q[(0, 1)] = k * fc;
    q[(0, 2)] = fa;
    q[(0, 3)] = fg;

    q[(1, 0)] = k * ft;
    q[(1, 1)] = -(k * ft + (fa + fg));
    q[(1, 2)] = fa;
    q[(1, 3)] = fg;

    q[(2, 0)] = ft;
    q[(2, 1)] = fc;
    q[(2, 2)] = -((ft + fc) + k * fg);
    q[(2, 3)] = k * fg;

    q[(3, 0)] = ft;
    q[(3, 1)] = fc;
    q[(3, 2)] = k * fa;
    q[(3, 3)] = -((ft + fc) + k * fa);
    q.scale_mut(scaler);
}

impl Display for HKY {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HKY with [kappa = {:.5}, freqs = {}]",
            self.kappa[0], self.freqs
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct TN93 {
    freqs: FreqVector,
    pub(crate) q: SubstMatrix,
    params: Vec<f64>,
    alphabet: Alphabet,
}

impl QMatrix for TN93 {
    fn new(freqs: &[f64], params: &[f64]) -> Self {
        let provided_freqs = frequencies!(freqs);
        let freqs = if verify_dna_freqs(&provided_freqs) {
            provided_freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };

        let mut params = params.to_vec();
        match params.len().cmp(&3) {
            Ordering::Less => {
                warn!("Too few values provided for TN93, required 3 values.");
                warn!("Falling back to default values.");
                params.extend(iter::repeat(1.0).take(3 - params.len()));
            }
            Ordering::Greater => {
                warn!("Too many values provided for TN93, required three values.");
                warn!("Will only use the first values provided.");
                params.truncate(3);
            }
            Ordering::Equal => {}
        }

        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        tn93_q(&mut q, &freqs, &params);
        TN93 {
            freqs,
            q,
            params,
            alphabet: dna_alphabet().clone(),
        }
    }
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) {
        self.freqs = if verify_dna_freqs(&freqs) {
            freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };
        tn93_q(&mut self.q, &self.freqs, &self.params)
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.params[param] = value;
        tn93_q(&mut self.q, &self.freqs, &self.params)
    }
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn index(&self) -> &[usize; 255] {
        &NUCLEOTIDE_INDEX
    }
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
}

fn tn93_q(q: &mut SubstMatrix, pi: &FreqVector, params: &[f64]) {
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let a1 = params[0];
    let a2 = params[1];
    let b = params[2];

    let scaler = 1.0
        / ((a1 * fc + b * fa + b * fg) * ft
            + (a1 * ft + b * fa + b * fg) * fc
            + (b * ft + b * fc + a2 * fg) * fa
            + (b * ft + b * fc + a2 * fa) * fg);

    q[(0, 0)] = -(a1 * fc + b * fa + b * fg);
    q[(0, 1)] = a1 * fc;
    q[(0, 2)] = b * fa;
    q[(0, 3)] = b * fg;

    q[(1, 0)] = a1 * ft;
    q[(1, 1)] = -(a1 * ft + b * fa + b * fg);
    q[(1, 2)] = b * fa;
    q[(1, 3)] = b * fg;

    q[(2, 0)] = b * ft;
    q[(2, 1)] = b * fc;
    q[(2, 2)] = -(b * ft + b * fc + a2 * fg);
    q[(2, 3)] = a2 * fg;

    q[(3, 0)] = b * ft;
    q[(3, 1)] = b * fc;
    q[(3, 2)] = a2 * fa;
    q[(3, 3)] = -(b * ft + b * fc + a2 * fa);

    q.scale_mut(scaler);
}

impl Display for TN93 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TN93 with [alpha1 = {:.5}, alpha2 = {:.5}, beta = {:.5}, freqs = {}]",
            self.params[0], self.params[1], self.params[2], self.freqs
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct GTR {
    freqs: FreqVector,
    q: SubstMatrix,
    params: Vec<f64>,
    alphabet: Alphabet,
}

impl QMatrix for GTR {
    fn new(freqs: &[f64], params: &[f64]) -> Self {
        let provided_freqs = frequencies!(freqs);
        let freqs = if verify_dna_freqs(&provided_freqs) {
            provided_freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };

        let mut params = params.to_vec();
        if params.len() < 5 {
            warn!("Too few values provided for GTR, required five values.");
            warn!("Falling back to default values.");
            params.extend(iter::repeat(1.0).take(5 - params.len()));
        } else if params.len() > 6 {
            warn!("Too many values provided for GTR, required five values.");
            warn!("Will only use the first values provided.");
            params.truncate(5);
        } else if params.len() == 6 {
            warn!("Allowing all rates to vary for GTR.");
        }

        println!("params.len() = {}", params.len());
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        gtr_q(&mut q, &freqs, &params);
        GTR {
            freqs,
            q,
            params,
            alphabet: dna_alphabet().clone(),
        }
    }
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) {
        self.freqs = if verify_dna_freqs(&freqs) {
            freqs
        } else {
            warn!("Invalid frequencies provided, using equal.");
            frequencies!(&[1.0 / DNA_N as f64; DNA_N])
        };
        gtr_q(&mut self.q, &self.freqs, &self.params)
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.params[param] = value;
        gtr_q(&mut self.q, &self.freqs, &self.params)
    }
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn index(&self) -> &[usize; 255] {
        &NUCLEOTIDE_INDEX
    }
    fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
}

fn gtr_q(q: &mut SubstMatrix, pi: &FreqVector, params: &[f64]) {
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let rtc = params[0];
    let rta = params[1];
    let rtg = params[2];
    let rca = params[3];
    let rcg = params[4];
    let rag = if params.len() == 6 { params[5] } else { 1.0 };

    let scaler = 1.0
        / ((rtc * fc + rta * fa + rtg * fg) * ft
            + (rtc * ft + rca * fa + rcg * fg) * fc
            + (rta * ft + rca * fc + rag * fg) * fa
            + (rtg * ft + rcg * fc + rag * fa) * fg);

    q[(0, 0)] = -(rtc * fc + rta * fa + rtg * fg);
    q[(0, 1)] = rtc * fc;
    q[(0, 2)] = rta * fa;
    q[(0, 3)] = rtg * fg;

    q[(1, 0)] = rtc * ft;
    q[(1, 1)] = -(rtc * ft + rca * fa + rcg * fg);
    q[(1, 2)] = rca * fa;
    q[(1, 3)] = rcg * fg;

    q[(2, 0)] = rta * ft;
    q[(2, 1)] = rca * fc;
    q[(2, 2)] = -(rta * ft + rca * fc + rag * fg);
    q[(2, 3)] = rag * fg;

    q[(3, 0)] = rtg * ft;
    q[(3, 1)] = rcg * fc;
    q[(3, 2)] = rag * fa;
    q[(3, 3)] = -(rtg * ft + rcg * fc + rag * fa);

    q.scale_mut(scaler);
}

impl Display for GTR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GTR with [rtc = {:.5}, rta = {:.5}, rtg = {:.5}, rca = {:.5}, rcg = {:.5}, rag = {:.5}, freqs = {}]",
            self.params[0], self.params[1], self.params[2],self.params[3],self.params[4], if self.params.len() == 6 { self.params[5] } else { 1.0 }, self.freqs
        )
    }
}
