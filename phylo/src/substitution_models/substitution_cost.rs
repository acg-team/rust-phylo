use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;

use anyhow::bail;
use nalgebra::DMatrix;

use crate::evolutionary_models::EvoModel;
use crate::likelihood::ModelOptCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{SubstMatrix, SubstitutionModel};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::Result;

pub struct SubstitutionCostBuilder<SM>
where
    SM: SubstitutionModel,
{
    model: SM,
    info: PhyloInfo,
}

#[derive(Debug, Clone)]
pub struct SubstitutionCost<SM>
where
    SM: SubstitutionModel,
{
    model: SM,
    info: PhyloInfo,
    tmp: RefCell<SubstModelInfo<SM>>,
}

impl<SM: SubstitutionModel + Clone + EvoModel> ModelOptCostFunction<SM> for SubstitutionCost<SM> {
    fn new(model: SM, info: PhyloInfo) -> Self {
        let tmp = RefCell::new(SubstModelInfo::new(&info, &model).unwrap());
        SubstitutionCost { model, info, tmp }
    }

    fn cost(&self) -> f64 {
        self.logl()
    }

    fn reset_model(&mut self, model: SM) {
        self.model = model;
        self.tmp.borrow_mut().node_models_valid.fill(false);
    }

    fn model(&self) -> &SM {
        &self.model
    }
}

impl<SM: SubstitutionModel + Clone> SubstitutionCost<SM> {
    fn logl(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal(node_idx);
                }
                Leaf(_) => {
                    self.set_leaf(node_idx);
                }
            };
        }
        let tmp_values = self.tmp.borrow();
        debug_assert_eq!(self.info.tree.len(), tmp_values.node_info.len());

        let likelihood = &self
            .model
            .freqs()
            .transpose()
            .mul(&tmp_values.node_info[usize::from(&self.info.tree.root)]);
        drop(tmp_values);

        debug_assert_eq!(likelihood.ncols(), self.info.msa.len());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let childx_info = self.tmp.borrow().node_info[usize::from(&node.children[0])].clone();
        let childy_info = self.tmp.borrow().node_info[usize::from(&node.children[1])].clone();

        let idx = usize::from(node_idx);

        let mut tmp_values = self.tmp.borrow_mut();
        if self.info.tree.dirty[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(node.blen);
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }
        tmp_values.node_info[idx] =
            (&tmp_values.node_models[idx]).mul(childx_info.component_mul(&childy_info));
        tmp_values.node_info_valid[idx] = true;
        if let Some(parent_idx) = node.parent {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        drop(tmp_values);
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        let mut tmp_values = self.tmp.borrow_mut();
        let idx = usize::from(node_idx);

        if self.info.tree.dirty[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(self.info.tree.blen(node_idx));
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }

        // get leaf sequence encoding
        let leaf_seq = tmp_values
            .leaf_sequence_info
            .get(self.info.tree.node_id(node_idx))
            .unwrap();
        tmp_values.node_info[idx] = (&tmp_values.node_models[idx]).mul(leaf_seq);
        if let Some(parent_idx) = self.info.tree.parent(node_idx) {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        tmp_values.node_info_valid[idx] = true;
        drop(tmp_values);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModelInfo<SM: SubstitutionModel> {
    empty: bool,
    phantom: PhantomData<SM>,
    node_info: Vec<DMatrix<f64>>,
    node_info_valid: Vec<bool>,
    node_models: Vec<SubstMatrix>,
    node_models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl<SM: SubstitutionModel> SubstModelInfo<SM> {
    pub fn empty() -> Self {
        SubstModelInfo::<SM> {
            empty: true,
            phantom: PhantomData::<SM>,
            node_info: Vec::new(),
            node_info_valid: Vec::new(),
            node_models: Vec::new(),
            node_models_valid: Vec::new(),
            leaf_sequence_info: HashMap::new(),
        }
    }

    pub fn new(info: &PhyloInfo, _model: &SM) -> Result<Self> {
        let node_count = info.tree.len();
        let msa_length = info.msa.len();

        let mut leaf_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let encoding = if let Some(enc) = info.msa.leaf_encoding.get(&node.id) {
                enc
            } else {
                bail!("Leaf encoding not found for leaf {}", node.id);
            };

            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(SM::N, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    site_info.copy_from(&encoding.column(c));
                } else {
                    site_info.copy_from(info.msa.alphabet().gap_encoding());
                }
            }
            leaf_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo::<SM> {
            empty: false,
            phantom: PhantomData::<SM>,
            node_info: vec![DMatrix::<f64>::zeros(SM::N, msa_length); node_count],
            node_info_valid: vec![false; node_count],
            node_models: vec![SubstMatrix::zeros(SM::N, SM::N); node_count],
            node_models_valid: vec![false; node_count],
            leaf_sequence_info: leaf_info,
        })
    }

    pub fn reset(&mut self) {
        self.empty = true;
        self.node_info.iter_mut().for_each(|x| x.fill(0.0));
        self.node_info_valid.fill(false);
        self.node_models.iter_mut().for_each(|x| x.fill(0.0));
        self.node_models_valid.fill(false);
    }
}
