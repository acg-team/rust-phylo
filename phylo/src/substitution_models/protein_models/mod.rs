use std::cell::RefCell;
use std::vec;

use crate::alphabets::AMINOACID_INDEX;
use crate::evolutionary_models::ProteinModelType;
use crate::substitution_models::{
    FreqVector, SubstMatrix, SubstModel, SubstModelInfo, SubstitutionModel,
};
use crate::Result;

pub(crate) mod protein_generics;
pub(crate) use protein_generics::*;
pub(crate) mod protein_subst_params;
pub(crate) use protein_subst_params::*;

pub type ProteinSubstModel = SubstModel<ProteinSubstParams>;
pub type ProteinSubstModelInfo = SubstModelInfo<ProteinSubstModel>;

impl SubstitutionModel for ProteinSubstModel {
    type ModelType = ProteinModelType;
    const N: usize = 20;

    fn new(model_type: ProteinModelType, _: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = ProteinSubstParams::new(model_type, &[])?;
        let mut model = ProteinSubstModel {
            q: params.q(),
            params,
            tmp: RefCell::new(ProteinSubstModelInfo::empty()),
        };
        model.normalise();
        Ok(model)
    }

    fn designation(&self) -> String {
        format!("Protein model: {}", self.params.model_type)
    }

    fn update(&mut self) {
        self.q = self.params.q();
        self.normalise();
        if !self.tmp.borrow().empty {
            self.tmp.borrow_mut().node_models_valid.fill(false);
        }
    }

    fn normalise(&mut self) {
        self.normalise();
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn model_parameters(&self) -> Vec<f64> {
        vec![]
    }

    fn set_param(&mut self, _param: usize, _value: f64) {}

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.params.set_freqs(freqs);
        self.update();
    }

    fn index(&self) -> &'static [usize; 255] {
        &AMINOACID_INDEX
    }
}
