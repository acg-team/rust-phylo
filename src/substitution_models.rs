use ordered_float::OrderedFloat;
use std::collections::HashMap;

use anyhow::anyhow;

use nalgebra::SMatrix;

use crate::Result;

mod dna_models;
mod protein_models;

#[allow(non_camel_case_types)]
type f32_h = ordered_float::OrderedFloat<f32>;

type DNASubstMatrix = SMatrix<f64, 4, 4>;
type ProteinSubstMatrix = SMatrix<f64, 20, 20>;

#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DNASubstModel {
    index: [i32; 255],
    q: DNASubstMatrix,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProteinSubstModel {
    index: [i32; 255],
    q: ProteinSubstMatrix,
}

impl SubstitutionModel for DNASubstModel {
    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0,
            "Invalid nucleotide rate requested."
        );
        assert!(
            self.index[j as usize] >= 0,
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
        match model_name.to_uppercase().as_str() {
            "JC69" => q = dna_models::JC69(),
            _ => return Err(anyhow!("Unknown DNA model requested.")),
        }
        Ok(DNASubstModel {
            index: dna_models::nucleotide_index(),
            q: q,
        })
    }
}

impl SubstitutionModel for ProteinSubstModel {
    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0,
            "Invalid aminoacid rate requested."
        );
        assert!(
            self.index[j as usize] >= 0,
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
        let mut ps = HashMap::<f32_h, SubstMatrix>::with_capacity(times.len());
        for time in times {
            ps.insert(f32_h::from(time), self.get_p(time));
        }
        ps
    }

    fn new(model_name: &str) -> Result<Self> {
        let q: ProteinSubstMatrix;
        match model_name.to_uppercase().as_str() {
            "WAG" => q = protein_models::WAG(),
            "BLOSUM" => q = protein_models::BLOSUM(),
            "HIVB" => q = protein_models::HIVB(),
            _ => return Err(anyhow!("Unknown protein model requested.")),
        }
        Ok(ProteinSubstModel {
            index: protein_models::aminoacid_index(),
            q: q,
        })
    }
}

#[cfg(test)]
mod substitution_model_tests {
    use crate::substitution_models::ProteinSubstModel;

    use super::{DNASubstModel, SubstitutionModel};

    #[test]
    fn dna_correct() {
        let jc69 = DNASubstModel::new("jc69").unwrap();
        let jc692 = DNASubstModel::new("JC69").unwrap();
        assert_eq!(jc69, jc692);
    }

    #[test]
    fn dna_incorrect() {
        assert!(DNASubstModel::new("jc70").is_err());
        assert!(DNASubstModel::new("wag").is_err());
    }

    #[test]
    fn protein_correct() {
        let wag = ProteinSubstModel::new("WAG").unwrap();
        let wag2 = ProteinSubstModel::new("wag").unwrap();
        assert_eq!(wag, wag2);
        ProteinSubstModel::new("Blosum").unwrap();
        ProteinSubstModel::new("HIVb").unwrap();
    }

    #[test]
    fn protein_incorrect() {
        assert!(ProteinSubstModel::new("jc69").is_err());
        assert!(ProteinSubstModel::new("waq").is_err());
        assert!(ProteinSubstModel::new("HIV").is_err());
    }
}
