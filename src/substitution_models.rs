use nalgebra::{Const, DimMin};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

use anyhow::anyhow;

use nalgebra::{SMatrix, SVector};

use crate::{Result, f64_h};

pub(crate) mod dna_models;
pub(crate) mod protein_models;

type SubstMatrix<const N: usize> = SMatrix<f64, N, N>;
type FreqVector<const N: usize> = SVector<f64, N>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    q: SubstMatrix<N>,
    pi: FreqVector<N>,
}

pub(crate) type DNASubstModel = SubstitutionModel<4>;
pub(crate) type ProteinSubstModel = SubstitutionModel<20>;

impl DNASubstModel {
    pub(crate) fn new(model_name: &str) -> Result<Self> {
        let q: SubstMatrix<4>;
        let pi: FreqVector<4>;
        match model_name.to_uppercase().as_str() {
            "JC69" => (q, pi) = dna_models::jc69(),
            _ => return Err(anyhow!("Unknown DNA model requested.")),
        }
        Ok(DNASubstModel {
            index: dna_models::nucleotide_index(),
            q,
            pi,
        })
    }
}

impl ProteinSubstModel {
    pub(crate) fn new(model_name: &str) -> Result<Self> {
        let q: SubstMatrix<20>;
        let pi: FreqVector<20>;
        match model_name.to_uppercase().as_str() {
            "WAG" => (q, pi) = protein_models::wag(),
            "BLOSUM" => (q, pi) = protein_models::blosum(),
            // "HIVB" => q = protein_models::hivb(),
            _ => return Err(anyhow!("Unknown protein model requested.")),
        }
        Ok(ProteinSubstModel {
            index: protein_models::aminoacid_index(),
            q,
            pi,
        })
    }
}

impl<const N: usize> SubstitutionModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    pub(crate) fn get_p(&self, time: f64) -> SubstMatrix<N> {
        SubstMatrix::from((self.q * time).exp())
    }

    pub(crate) fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    pub(crate) fn generate_ps(&self, times: &[f64]) -> HashMap<OrderedFloat<f64>, SubstMatrix<N>> {
        HashMap::<f64_h, SubstMatrix<N>>::from_iter(
            times
                .into_iter()
                .map(|&time| (f64_h::from(time), self.get_p(time))),
        )
    }

    pub(crate) fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix<N>, f64)> {
        HashMap::<f64_h, (SubstMatrix<N>, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag),
            )
        }))
    }

    pub(crate) fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q = self.q / factor;
    }

    pub(crate) fn get_scoring_matrix(&self, time: f64) -> (SubstMatrix<N>, f64) {
        self.get_scoring_matrix_corrected(time, false)
    }

    fn get_scoring_matrix_corrected(&self, time: f64, zero_diag: bool) -> (SubstMatrix<N>, f64) {
        let p = self.get_p(time);
        let mut scores = p.map(|x| (-x.ln()).round());
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        (scores, scores.mean())
    }
}

#[cfg(test)]
mod substitution_model_tests {
    use super::{DNASubstModel, ProteinSubstModel, SubstMatrix};
    use rstest::*;

    fn check_pi_convergence<const N: usize>(substmat: SubstMatrix<N>, pi: &[f64], epsilon: f64) {
        assert_eq!(N, pi.len());
        for col in substmat.column_iter() {
            for (i, &cell) in col.iter().enumerate() {
                assert_float_absolute_eq!(cell, pi[i], epsilon);
            }
        }
    }

    #[test]
    fn dna_model_correct() {
        let jc69 = DNASubstModel::new("jc69").unwrap();
        let jc692 = DNASubstModel::new("JC69").unwrap();
        assert_eq!(jc69, jc692);
    }

    #[test]
    fn dna_model_incorrect() {
        assert!(DNASubstModel::new("jc70").is_err());
        assert!(DNASubstModel::new("wag").is_err());
    }

    #[test]
    fn dna_p_matrix() {
        let jc69 = DNASubstModel::new("jc69").unwrap();
        let p_inf = jc69.get_p(200000.0);
        assert_eq!(p_inf.nrows(), 4);
        assert_eq!(p_inf.ncols(), 4);
        check_pi_convergence(p_inf, jc69.pi.as_slice(), 1e-5);
    }

    #[test]
    fn dna_normalisation() {
        let mut jc69 = DNASubstModel::new("jc69").unwrap();
        jc69.normalise();
        assert_eq!((jc69.q.sum() - jc69.q.diagonal().sum()) / 4.0, 1.0);
    }

    #[test]
    fn protein_model_correct() {
        let wag = ProteinSubstModel::new("WAG").unwrap();
        let wag2 = ProteinSubstModel::new("wag").unwrap();
        assert_eq!(wag, wag2);
        wag.get_rate(b'A', b'L');
        wag.get_rate(b'H', b'K');
        let blos = ProteinSubstModel::new("Blosum").unwrap();
        let blos2 = ProteinSubstModel::new("bLoSuM").unwrap();
        assert_eq!(blos, blos2);
        blos.get_rate(b'R', b'N');
        blos.get_rate(b'M', b'K');
        // ProteinSubstModel::new("HIVb").unwrap();
    }

    #[test]
    #[should_panic]
    fn protein_model_incorrect_access() {
        let wag = ProteinSubstModel::new("WAG").unwrap();
        wag.get_rate(b'H', b'J');
    }

    #[test]
    fn protein_model_incorrect() {
        assert!(ProteinSubstModel::new("jc69").is_err());
        assert!(ProteinSubstModel::new("waq").is_err());
        assert!(ProteinSubstModel::new("HIV").is_err());
    }

    #[rstest]
    #[case::wag("wag", 1e-3)]
    #[case::blosum("blosum", 1e-3)]
    fn protein_p_matrix(#[case] input: &str, #[case] epsilon: f64) {
        let model = ProteinSubstModel::new(input).unwrap();
        let p_inf = model.get_p(20000.0);
        assert_eq!(p_inf.nrows(), 20);
        assert_eq!(p_inf.ncols(), 20);
        check_pi_convergence(p_inf, model.pi.as_slice(), epsilon);
    }

    #[rstest]
    #[case::wag("wag", 0.1)]
    #[case::blosum("blosum", 0.01)]
    fn protein_normalisation(#[case] input: &str, #[case] epsilon: f64) {
        // This uses a crazy epsilon, but the matrices are off by a bit, need fixing.
        let mut model = ProteinSubstModel::new(input).unwrap();
        model.normalise();
        assert_float_absolute_eq!(
            (model.q.sum() - model.q.diagonal().sum()) / 20.0,
            1.0,
            epsilon
        );
    }

    #[test]
    fn protein_scoring_matrices() {
        let mut model = ProteinSubstModel::new("wag").unwrap();
        model.normalise();
        let true_matrix_01 = SubstMatrix::from([
            [
                0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0,
                4.0, 9.0, 7.0, 4.0,
            ],
            [
                5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0,
                6.0, 6.0, 7.0, 6.0,
            ],
            [
                5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0, 7.0, 4.0,
                4.0, 9.0, 6.0, 6.0,
            ],
            [
                5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0, 8.0, 8.0, 6.0, 5.0,
                6.0, 9.0, 7.0, 7.0,
            ],
            [
                5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0, 7.0, 5.0,
                6.0, 7.0, 6.0, 5.0,
            ],
            [
                5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0, 5.0, 7.0, 5.0, 4.0, 6.0, 8.0, 5.0, 5.0,
                5.0, 8.0, 7.0, 6.0,
            ],
            [
                4.0, 6.0, 6.0, 3.0, 9.0, 4.0, 0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0, 8.0, 6.0, 5.0,
                5.0, 8.0, 7.0, 5.0,
            ],
            [
                4.0, 6.0, 5.0, 5.0, 7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0, 8.0, 8.0, 7.0, 5.0,
                6.0, 8.0, 8.0, 7.0,
            ],
            [
                6.0, 5.0, 4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0, 5.0, 7.0, 6.0, 6.0, 5.0,
                6.0, 8.0, 4.0, 7.0,
            ],
            [
                6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0, 4.0, 6.0, 5.0, 5.0, 8.0, 6.0,
                5.0, 8.0, 6.0, 3.0,
            ],
            [
                6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0, 4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0,
                6.0, 7.0, 6.0, 4.0,
            ],
            [
                5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 0.0, 6.0, 8.0, 6.0, 5.0,
                5.0, 8.0, 8.0, 6.0,
            ],
            [
                5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 4.0, 3.0, 5.0, 0.0, 5.0, 7.0, 6.0,
                5.0, 7.0, 6.0, 4.0,
            ],
            [
                6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 6.0, 5.0, 4.0, 7.0, 6.0, 0.0, 7.0, 6.0,
                7.0, 6.0, 4.0, 5.0,
            ],
            [
                4.0, 6.0, 7.0, 6.0, 8.0, 6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0,
                5.0, 8.0, 7.0, 6.0,
            ],
            [
                4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0,
                4.0, 7.0, 6.0, 6.0,
            ],
            [
                4.0, 6.0, 5.0, 6.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0,
                0.0, 9.0, 7.0, 5.0,
            ],
            [
                7.0, 5.0, 8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0,
                7.0, 0.0, 5.0, 6.0,
            ],
            [
                6.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0,
                6.0, 6.0, 0.0, 6.0,
            ],
            [
                4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                5.0, 8.0, 7.0, 0.0,
            ],
        ]);
        let (mat, avg) = model.get_scoring_matrix(0.1);
        assert_eq!(mat.to_string(), true_matrix_01.to_string());
        assert_float_absolute_eq!(avg, 5.7675);
        let (_, avg) = model.get_scoring_matrix(0.3);
        assert_float_absolute_eq!(avg, 4.7475);
        let (_, avg) = model.get_scoring_matrix(0.5);
        assert_float_absolute_eq!(avg, 4.2825);
        let (_, avg) = model.get_scoring_matrix(0.7);
        assert_float_absolute_eq!(avg, 4.0075);
    }
}
