use std::collections::HashMap;
use ordered_float::OrderedFloat;

use nalgebra::SMatrix;

use crate::sequences::{charify, NUCLEOTIDES_STR};

use crate::Result;

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
    fn new(name: &str) -> Result<Self> where Self: Sized;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn get_p(&self, time: f32) -> SubstMatrix;
    fn generate_ps(&self, times: Vec<f32>) -> HashMap<OrderedFloat<f32>, SubstMatrix>;
}

pub(crate) fn nucleotide_index() -> [i32; 255] {
    let mut index = [-1 as i32; 255];
    for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
        index[char as usize] = i as i32;
        index[char.to_ascii_lowercase() as usize] = i as i32;
    }
    index
}

#[derive(Clone, Debug)]
pub(crate) struct DNASubstModel {
    index: [i32; 255],
    q: DNASubstMatrix,
}

#[derive(Clone, Debug)]
pub(crate) struct ProteinSubstModel {
    index: [i32; 255],
    q: ProteinSubstMatrix,
}

impl SubstitutionModel for ProteinSubstModel {
    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(self.index[i as usize] >= 0, "Invalid aminoacid rate requested.");
        assert!(self.index[j as usize] >= 0, "Invalid aminoacid rate requested.");
        self.q[(self.index[i as usize] as usize, self.index[j as usize] as usize)]
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
        match model_name.to_uppercase().as_str() {
            "WAG" => {
                Ok(ProteinSubstModel {
                    index: protein_models::aminoacid_index(),
                    q: protein_models::protein_matrix(protein_models::WAG),
                })
            }
            "BLOSUM" => {
                Ok(ProteinSubstModel {
                    index: protein_models::aminoacid_index(),
                    q: protein_models::protein_matrix(protein_models::BLOSUM),
                })
            }
            "HIVB" => {
                Ok(ProteinSubstModel {
                    index: protein_models::aminoacid_index(),
                    q: protein_models::protein_matrix(protein_models::HIVB),
                })
            }
            _ => {
                unimplemented!("Unknown protein model requested.");
            }
        }
    }
}
