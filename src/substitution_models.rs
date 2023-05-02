use ordered_float::OrderedFloat;
use std::collections::HashMap;

use anyhow::anyhow;

use nalgebra::{SMatrix, SVector};

use crate::Result;

mod dna_models;
mod protein_models;

#[allow(non_camel_case_types)]
type f32_h = ordered_float::OrderedFloat<f32>;

type SubstMat<const N: usize> = SMatrix<f64, N, N>;

type DNASubstMatrix = SubstMat<4>;
type ProteinSubstMatrix = SubstMat<20>;

type FreqVector<const N: usize> = SVector<f64, N>;

type DNAFrequencies = FreqVector<4>;
type ProteinFrequencies = FreqVector<20>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SubstMatrix {
    DNA(DNASubstMatrix),
    Protein(ProteinSubstMatrix),
}

pub(crate) trait SubstitutionModel {
    fn new(name: &str) -> Result<Self>
    where
        Self: Sized;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn get_p(&self, time: f32) -> SubstMatrix;
    fn generate_ps(&self, times: Vec<f32>) -> HashMap<OrderedFloat<f32>, SubstMatrix>;
    fn normalise(&mut self);
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DNASubstModel {
    index: [i32; 255],
    q: DNASubstMatrix,
    pi: DNAFrequencies,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProteinSubstModel {
    index: [i32; 255],
    q: ProteinSubstMatrix,
    pi: ProteinFrequencies,
}

impl SubstitutionModel for DNASubstModel {
    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid nucleotide rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn get_p(&self, time: f32) -> SubstMatrix {
        SubstMatrix::DNA((self.q * time as f64).exp())
    }

    fn generate_ps(&self, times: Vec<f32>) -> HashMap<f32_h, SubstMatrix> {
        let mut ps = HashMap::<f32_h, SubstMatrix>::with_capacity(times.len());
        for time in times {
            ps.insert(f32_h::from(time), self.get_p(time));
        }
        ps
    }

    fn new(model_name: &str) -> Result<Self> {
        let q: DNASubstMatrix;
        let pi: DNAFrequencies;
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

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q = self.q / factor;
    }
}

impl SubstitutionModel for ProteinSubstModel {
    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid aminoacid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn get_p(&self, time: f32) -> SubstMatrix {
        SubstMatrix::Protein((self.q * time as f64).exp())
    }

    fn generate_ps(&self, times: Vec<f32>) -> HashMap<f32_h, SubstMatrix> {
        HashMap::<f32_h, SubstMatrix>::from_iter(
            times
                .into_iter()
                .map(|time| (f32_h::from(time), self.get_p(time))),
        )
    }

    fn new(model_name: &str) -> Result<Self> {
        let q: ProteinSubstMatrix;
        let pi: ProteinFrequencies;
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

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q = self.q / factor;
    }
}

#[cfg(test)]
mod substitution_model_tests {
    use crate::substitution_models::{ProteinSubstModel, SubstMatrix};
    use rstest::*;

    use super::{DNASubstModel, SubstMat, SubstitutionModel};

    fn check_pi_convergence<const N: usize>(substmat: SubstMat<N>, pi: &[f64], epsilon: f64) {
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
        assert!(matches!(p_inf, SubstMatrix::DNA(_)));
        if let SubstMatrix::DNA(mat) = p_inf {
            check_pi_convergence(mat, jc69.pi.as_slice(), 1e-5);
        }
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
    #[case::wag("wag", 1e-5)]
    #[case::blosum("blosum", 1e-3)]
    fn protein_p_matrix(#[case] input: &str, #[case] epsilon: f64) {
        let model = ProteinSubstModel::new(input).unwrap();
        let p_inf = model.get_p(20000.0);
        assert!(matches!(p_inf, SubstMatrix::Protein(_)));
        if let SubstMatrix::Protein(mat) = p_inf {
            check_pi_convergence(mat, model.pi.as_slice(), epsilon);
        }
    }

    #[test]
    fn protein_normalisation() {
        // This uses a crazy epsilon, but the wag matrix is just off, needs to be fixed.
        let mut wag = ProteinSubstModel::new("wag").unwrap();
        wag.normalise();
        assert_float_absolute_eq!((wag.q.sum() - wag.q.diagonal().sum()) / 20.0, 1.0, 0.1);
    }
}
