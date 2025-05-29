use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::iter::{self};
use std::marker::PhantomData;
use std::ops::Range;

use anyhow::bail;
use fixedbitset::FixedBitSet;
use indices::{DisjointIndices, DisjointRanges, NodeCacheIndexer, NodeCacheSlicer};
use itertools::Itertools;
use lazy_static::lazy_static;
use log::warn;
use nalgebra::{
    DMatrix, DMatrixView, DMatrixViewMut, DVectorView, DVectorViewMut, MatrixViewMutXx3,
    MatrixViewXx3,
};

use crate::alignment::Mapping;
use crate::alphabets::{Alphabet, GAP};
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, QMatrix, QMatrixMaker, SubstMatrix};
use crate::tree::{
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

// (2.0 * PI).ln() / 2.0;
pub static SHIFT: f64 = 0.9189385332046727;

lazy_static! {
    pub static ref MINLOGPROB: f64 = (f64::MIN_POSITIVE).ln();
}

fn log_factorial_shifted(n: usize) -> f64 {
    // An approximation using Stirling's formula, minus constant log(sqrt(2*PI)).
    // Has an absolute error of 3e-4 for n=2, 3e-6 for n=10, 3e-9 for n=100.
    let n = n as f64;
    n * n.ln() - n + n.ln() / 2.0 + 1.0 / (12.0 * n) + SHIFT
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModel<Q: QMatrix> {
    pub(crate) subst_q: Q,
    q: SubstMatrix,
    freqs: FreqVector,
    params: Vec<f64>,
}

fn pip_q(q: &mut SubstMatrix, subst_q: &SubstMatrix, mu: f64) {
    let n = subst_q.ncols();
    q.view_mut((0, 0), (n, n)).copy_from(subst_q);
    q.fill_column(n, mu);
    for i in 0..(n + 1) {
        q[(i, i)] -= mu;
    }
}

impl<Q: QMatrix> PIPModel<Q> {
    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }
}

impl<Q: QMatrix + QMatrixMaker> PIPModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Self {
        let mut params = params.to_vec();
        if params.len() < 2 {
            warn!("Too few values provided for PIP, 2 values required, lambda and mu.");
            warn!("Falling back to default values.");
            params.extend(iter::repeat_n(1.5, 2 - params.len()));
        }
        let mu = params[1];

        let subst_q = Q::create(frequencies, &params[2..]);
        let n = subst_q.n();
        let freqs = subst_q.freqs().clone().insert_row(n, 0.0);
        let mut q = SubstMatrix::zeros(n + 1, n + 1);
        pip_q(&mut q, subst_q.q(), mu);
        PIPModel {
            subst_q,
            q,
            freqs,
            params: params.to_vec(),
        }
    }
}

impl<Q: QMatrix> EvoModel for PIPModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        let mut to = SubstMatrix::zeros(self.q().nrows(), self.q().ncols());
        self.p_to(time, &mut to.as_view_mut());
        to
    }
    fn p_to(&self, time: f64, to: &mut DMatrixViewMut<f64>) {
        to.copy_from(self.q());
        *to *= time;
        // TODO: nalgebra seems to require the matrix to be owned
        // for exp
        to.copy_from(&to.clone_owned().exp());
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        let n = self.subst_q.n();
        if i == GAP {
            self.q[(n, 0)]
        } else if j == GAP {
            self.q[(0, n)]
        } else {
            self.subst_q.rate(i, j)
        }
    }

    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }

    // This assumes correct dimensions to minimise runtime checks
    fn set_freqs(&mut self, pi: FreqVector) {
        debug_assert!(self.freqs.nrows() - 1 == pi.nrows() || self.freqs.nrows() == pi.nrows());
        self.freqs = pi.clone().insert_row(pi.nrows(), 0.0);
        self.subst_q.set_freqs(pi);
        pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, param: usize, value: f64) {
        match param {
            0 => {
                self.params[0] = value;
            }
            1 => {
                self.params[1] = value;
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
            _ => {
                self.params[param] = value;
                self.subst_q.set_param(param - 2, value);
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
        }
    }

    fn n(&self) -> usize {
        self.subst_q.n() + 1
    }

    fn alphabet(&self) -> &Alphabet {
        self.subst_q.alphabet()
    }
}

impl<Q: QMatrix + Display> Display for PIPModel<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PIP with [lambda = {:.5}, mu = {:.5}]\n and {}",
            self.params[0], self.params[1], self.subst_q
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct C0Scalars {
    f1: f64,
    pnu: f64,
}

trait SubSlicer<T>: AsRef<[T]> {
    fn slice_idx_with_len_in(&self, range: Range<usize>, idx: usize, len: usize) -> &[T] {
        let slice_in_range = idx * len..(idx + 1) * len;
        &self.as_ref()[range][slice_in_range]
    }
}
trait SubSlicerMut<T>: AsMut<[T]> {
    fn slice_mut_idx_with_len_in(
        &mut self,
        range: Range<usize>,
        idx: usize,
        len: usize,
    ) -> &mut [T] {
        let slice_in_range = idx * len..(idx + 1) * len;
        &mut self.as_mut()[range][slice_in_range]
    }
}

impl<T> SubSlicer<T> for Box<[T]> {}
impl<T> SubSlicer<T> for [T] {}
impl<T> SubSlicerMut<T> for Box<[T]> {}
impl<T> SubSlicerMut<T> for [T] {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PIPModelCacheBufDimensions {
    n: usize,
    msa_length: usize,
    node_count: usize,
}

impl PIPModelCacheBufDimensions {
    pub fn new(n: usize, msa_length: usize, node_count: usize) -> Self {
        Self {
            n,
            msa_length,
            node_count,
        }
    }

    const fn models_len(&self) -> usize {
        self.n * self.n
    }
    const fn models_range(&self) -> Range<usize> {
        let offset = 0;
        offset..(offset + self.node_count * self.n * self.n)
    }

    const fn surv_ins_weights_range(&self) -> Range<usize> {
        let offset = self.models_range().end;
        offset..(offset + self.node_count)
    }

    const fn ftilde_len(&self) -> usize {
        self.n * self.msa_length
    }
    const fn ftilde_range(&self) -> Range<usize> {
        let offset = self.surv_ins_weights_range().end;
        offset..(offset + self.node_count * self.ftilde_len())
    }

    const fn anc_len(&self) -> usize {
        3 * self.msa_length
    }
    const fn anc_range(&self) -> Range<usize> {
        let offset = self.ftilde_range().end;
        offset..(offset + self.node_count * self.anc_len())
    }

    const fn pnu_len(&self) -> usize {
        self.msa_length
    }
    const fn pnu_range(&self) -> Range<usize> {
        let offset = self.anc_range().end;
        offset..(offset + self.node_count * self.pnu_len())
    }

    const fn c0_range(&self) -> Range<usize> {
        let offset = self.pnu_range().end;
        offset..(offset + self.node_count * 2)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModelCacheBuf {
    buf: Box<[f64]>,
    dimensions: PIPModelCacheBufDimensions,
    valid: FixedBitSet,
    models_valid: FixedBitSet,
}

#[derive(Debug)]
pub struct PIPModelCacheEntryViewMut<'a> {
    surv_ins_weight: &'a mut f64,
    anc: MatrixViewMutXx3<'a, f64>,
    ftilde: DMatrixViewMut<'a, f64>,
    pnu: DVectorViewMut<'a, f64>,
    c0_f1: &'a mut f64,
    c0_pnu: &'a mut f64,
    model: DMatrixViewMut<'a, f64>,
}

impl PIPModelCacheBuf {
    fn new(n: usize, msa_length: usize, node_count: usize) -> Self {
        let dimensions = PIPModelCacheBufDimensions::new(n, msa_length, node_count);
        let cum_dynamic_vector_and_matrix_size_in_f64 = [
            dimensions.models_range().len(),
            dimensions.surv_ins_weights_range().len(),
            dimensions.ftilde_range().len(),
            dimensions.anc_range().len(),
            dimensions.pnu_range().len(),
            dimensions.c0_range().len(),
        ]
        .iter()
        .sum::<usize>();
        Self {
            buf: vec![0.0; cum_dynamic_vector_and_matrix_size_in_f64].into_boxed_slice(),
            dimensions,
            valid: FixedBitSet::with_capacity(node_count),
            models_valid: FixedBitSet::with_capacity(node_count),
        }
    }

    fn idx_len_to_range(idx: usize, len: usize) -> Range<usize> {
        idx * len..(idx + 1) * len
    }

    fn entry_mut(&mut self, idx: usize) -> PIPModelCacheEntryViewMut {
        let [models_slice, surv_ins_weights_slice, ftildes_slice, ancs_slice, pnus_slice, c0s_slice] =
            self.buf
                .get_disjoint_mut([
                    self.dimensions.models_range(),
                    self.dimensions.surv_ins_weights_range(),
                    self.dimensions.ftilde_range(),
                    self.dimensions.anc_range(),
                    self.dimensions.pnu_range(),
                    self.dimensions.c0_range(),
                ])
                .expect("regions dont overlap");
        let model = {
            let item_len = self.dimensions.models_len();
            DMatrixViewMut::from_slice(
                &mut models_slice[Self::idx_len_to_range(idx, item_len)],
                self.dimensions.n,
                self.dimensions.n,
            )
        };
        let surv_ins_weight = &mut surv_ins_weights_slice[idx];

        let ftilde = {
            let item_len = self.dimensions.ftilde_len();

            DMatrixViewMut::from_slice(
                &mut ftildes_slice[Self::idx_len_to_range(idx, item_len)],
                self.dimensions.n,
                self.dimensions.msa_length,
            )
        };
        let anc = {
            let item_len = self.dimensions.anc_len();

            MatrixViewMutXx3::from_slice(
                &mut ancs_slice[Self::idx_len_to_range(idx, item_len)],
                self.dimensions.msa_length,
            )
        };

        let pnu = {
            let item_len = self.dimensions.pnu_len();

            DVectorViewMut::from_slice(
                &mut pnus_slice[Self::idx_len_to_range(idx, item_len)],
                self.dimensions.msa_length,
            )
        };

        let c0 = &mut bytemuck::cast_slice_mut::<f64, C0Scalars>(c0s_slice)[idx];

        PIPModelCacheEntryViewMut {
            model,
            surv_ins_weight,
            ftilde,
            anc,
            pnu,
            c0_f1: &mut c0.f1,
            c0_pnu: &mut c0.pnu,
        }
    }

    fn entries_mut(&mut self, indices: DisjointIndices) -> [PIPModelCacheEntryViewMut; 3] {
        let [models_slice, surv_ins_weights_slice, ftildes_slice, ancs_slice, pnus_slice, c0s_slice] =
            self.buf
                .get_disjoint_mut([
                    self.dimensions.models_range(),
                    self.dimensions.surv_ins_weights_range(),
                    self.dimensions.ftilde_range(),
                    self.dimensions.anc_range(),
                    self.dimensions.pnu_range(),
                    self.dimensions.c0_range(),
                ])
                .expect("regions dont overlap");
        let models = {
            let item_len = self.dimensions.models_len();
            let disjoint_ranges = DisjointRanges::new(indices, item_len);
            let sub_slices = models_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.n)
            })
        };
        let surv_ins_weights = surv_ins_weights_slice.left_right_this_mut(indices);

        let ftildes = {
            let item_len = self.dimensions.ftilde_len();

            let disjoint_ranges = DisjointRanges::new(indices, item_len);
            let sub_slices = ftildes_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.msa_length)
            })
        };
        let ancs = {
            let item_len = self.dimensions.anc_len();

            let disjoint_ranges = DisjointRanges::new(indices, item_len);
            let sub_slices = ancs_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                MatrixViewMutXx3::from_slice(sub_slice, self.dimensions.msa_length)
            })
        };

        let pnus = {
            let item_len = self.dimensions.pnu_len();

            let disjoint_ranges = DisjointRanges::new(indices, item_len);
            let sub_slices = pnus_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices
                .map(|sub_slice| DVectorViewMut::from_slice(sub_slice, self.dimensions.msa_length))
        };

        let c0s = bytemuck::cast_slice_mut::<f64, C0Scalars>(c0s_slice)
            .get_disjoint_mut(indices.left_right_this())
            .unwrap();
        itertools::izip!(models, surv_ins_weights, ftildes, ancs, pnus, c0s)
            .map(
                |(model, surv_ins_weight, ftilde, anc, pnu, c0)| PIPModelCacheEntryViewMut {
                    surv_ins_weight,
                    anc,
                    ftilde,
                    pnu,
                    c0_f1: &mut c0.f1,
                    c0_pnu: &mut c0.pnu,
                    model,
                },
            )
            .collect_array::<3>()
            .expect("all input cache view arrays are of length 3")
    }

    fn indices_models_mut(&mut self, indices: DisjointIndices) -> [DMatrixViewMut<f64>; 3] {
        let range = self.dimensions.models_range();
        let item_len = self.dimensions.models_len();

        let disjoint_ranges = DisjointRanges::new(indices, item_len);
        let sub_slices = self.buf[range].slice_left_right_this_mut(disjoint_ranges);

        sub_slices.map(|sub_slice| {
            DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.n)
        })
    }
    fn models_mut(&mut self, idx: usize) -> DMatrixViewMut<f64> {
        let range = self.dimensions.models_range();
        let item_len = self.dimensions.models_len();

        DMatrixViewMut::from_slice(
            self.buf.slice_mut_idx_with_len_in(range, idx, item_len),
            self.dimensions.n,
            self.dimensions.n,
        )
    }

    fn surv_ins_weights_mut(&mut self) -> &mut [f64] {
        let range = self.dimensions.surv_ins_weights_range();
        &mut self.buf[range]
    }
    fn surv_ins_weights(&self) -> &[f64] {
        let range = self.dimensions.surv_ins_weights_range();
        &self.buf[range]
    }

    fn indices_ftilde_mut(&mut self, indices: DisjointIndices) -> [DMatrixViewMut<f64>; 3] {
        let range = self.dimensions.ftilde_range();
        let item_len = self.dimensions.ftilde_len();

        let disjoint_ranges = DisjointRanges::new(indices, item_len);
        let sub_slices = self.buf[range].slice_left_right_this_mut(disjoint_ranges);

        sub_slices.map(|sub_slice| {
            DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.msa_length)
        })
    }
    fn ftilde_mut(&mut self, idx: usize) -> DMatrixViewMut<f64> {
        let range = self.dimensions.ftilde_range();
        let item_len = self.dimensions.ftilde_len();

        DMatrixViewMut::from_slice(
            self.buf.slice_mut_idx_with_len_in(range, idx, item_len),
            self.dimensions.n,
            self.dimensions.msa_length,
        )
    }
    fn ftilde(&self, idx: usize) -> DMatrixView<f64> {
        let range = self.dimensions.ftilde_range();
        let item_len = self.dimensions.ftilde_len();

        DMatrixView::from_slice(
            self.buf.slice_idx_with_len_in(range, idx, item_len),
            self.dimensions.n,
            self.dimensions.msa_length,
        )
    }

    fn indices_anc_mut(&mut self, indices: DisjointIndices) -> [MatrixViewMutXx3<f64>; 3] {
        let range = self.dimensions.anc_range();
        let item_len = self.dimensions.anc_len();

        let disjoint_ranges = DisjointRanges::new(indices, item_len);
        let sub_slices = self.buf[range].slice_left_right_this_mut(disjoint_ranges);

        sub_slices
            .map(|sub_slice| MatrixViewMutXx3::from_slice(sub_slice, self.dimensions.msa_length))
    }
    fn anc_mut(&mut self, idx: usize) -> MatrixViewMutXx3<f64> {
        let range = self.dimensions.anc_range();
        let item_len = self.dimensions.anc_len();

        MatrixViewMutXx3::from_slice(
            self.buf.slice_mut_idx_with_len_in(range, idx, item_len),
            self.dimensions.msa_length,
        )
    }
    fn anc(&self, idx: usize) -> MatrixViewXx3<f64> {
        let range = self.dimensions.anc_range();
        let item_len = self.dimensions.anc_len();

        MatrixViewXx3::from_slice(
            self.buf.slice_idx_with_len_in(range, idx, item_len),
            self.dimensions.msa_length,
        )
    }

    fn indices_pnu_mut(&mut self, indices: DisjointIndices) -> [DVectorViewMut<f64>; 3] {
        let range = self.dimensions.pnu_range();
        let item_len = self.dimensions.pnu_len();

        let disjoint_ranges = DisjointRanges::new(indices, item_len);
        let sub_slices = self.buf[range].slice_left_right_this_mut(disjoint_ranges);

        sub_slices
            .map(|sub_slice| DVectorViewMut::from_slice(sub_slice, self.dimensions.msa_length))
    }
    fn pnu_mut(&mut self, idx: usize) -> DVectorViewMut<f64> {
        let range = self.dimensions.pnu_range();
        let item_len = self.dimensions.pnu_len();

        DVectorViewMut::from_slice(
            self.buf.slice_mut_idx_with_len_in(range, idx, item_len),
            self.dimensions.msa_length,
        )
    }
    fn pnu(&self, index: usize) -> DVectorView<f64> {
        let range = self.dimensions.pnu_range();
        let item_len = self.dimensions.pnu_len();

        DVectorView::from_slice(
            &self.buf[range][index * item_len..(index + 1) * item_len],
            self.dimensions.msa_length,
        )
    }

    fn c0(&self) -> &[C0Scalars] {
        let range = self.dimensions.c0_range();
        bytemuck::cast_slice(&self.buf[range])
    }
    fn c0_mut(&mut self) -> &mut [C0Scalars] {
        let range = self.dimensions.c0_range();
        bytemuck::cast_slice_mut(&mut self.buf[range])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    cache: PIPModelCacheBuf,
    matrix_buf: DMatrix<f64>,
}

impl<Q: QMatrix> PIPModelInfo<Q> {
    pub fn new(info: &PhyloInfo, model: &PIPModel<Q>) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        let mut cache_entries = PIPModelCacheBuf::new(n, msa_length, node_count);
        for node in info.tree.leaves() {
            let seq = info.msa.seqs.record_by_id(&node.id).seq().to_vec();

            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_seq_w_gaps = &mut cache_entries.ftilde_mut(usize::from(node.idx));

            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = info
                        .msa
                        .alphabet()
                        .char_encoding(seq[c])
                        .clone()
                        .insert_row(n - 1, 0.0);

                    site_info.copy_from(&encoding);
                } else {
                    site_info.fill_row(n - 1, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
        }
        Ok(PIPModelInfo {
            phantom: PhantomData,
            cache: cache_entries,
            matrix_buf: DMatrix::zeros(n, msa_length),
        })
    }
}

pub struct PIPCostBuilder<Q: QMatrix> {
    model: PIPModel<Q>,
    info: PhyloInfo,
}

impl<Q: QMatrix> PIPCostBuilder<Q> {
    pub fn new(model: PIPModel<Q>, info: PhyloInfo) -> Self {
        PIPCostBuilder { model, info }
    }

    pub fn build(self) -> Result<PIPCost<Q>> {
        if self.info.msa.alphabet() != self.model.alphabet() {
            bail!("Alphabet mismatch between model and alignment.");
        }

        let tmp = RefCell::new(PIPModelInfo::new(&self.info, &self.model).unwrap());
        Ok(PIPCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PIPCost<Q: QMatrix> {
    pub(crate) model: PIPModel<Q>,
    pub(crate) info: PhyloInfo,
    tmp: RefCell<PIPModelInfo<Q>>,
}

impl<Q: QMatrix> TreeSearchCost for PIPCost<Q> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]) {
        self.info.tree = tree;
        if dirty_nodes.is_empty() {
            self.tmp.borrow_mut().cache.valid.clear();
            self.tmp.borrow_mut().cache.models_valid.clear();
            return;
        }
        for node_idx in dirty_nodes {
            self.tmp
                .borrow_mut()
                .cache
                .valid
                .remove(usize::from(node_idx));
            self.tmp
                .borrow_mut()
                .cache
                .models_valid
                .remove(usize::from(node_idx));
        }
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl<Q: QMatrix> ModelSearchCost for PIPCost<Q> {
    fn cost(&self) -> f64 {
        self.logl()
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().cache.models_valid.clear();
    }

    fn params(&self) -> &[f64] {
        self.model.params()
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.model.set_freqs(freqs);
        self.tmp.borrow_mut().cache.models_valid.clear();
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.model.freqs()
    }
}

impl<Q: QMatrix + Display> Display for PIPCost<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

struct CacheLeafViewMut<'a, 'b: 'a> {
    surv_ins_weights: &'a mut f64,
    anc: &'a mut MatrixViewMutXx3<'b, f64>,
    ftilde: &'a DMatrixView<'b, f64>,
    pnu: &'a mut DVectorViewMut<'b, f64>,
    c0_f1: &'a mut f64,
    c0_pnu: &'a mut f64,
}

struct CacheFtildeViewMut<'a, 'b: 'a> {
    ftilde: &'a mut DMatrixViewMut<'b, f64>,
    f: &'a mut DVectorViewMut<'b, f64>,
}
struct CacheFtildeChildView<'a> {
    ftilde: DMatrixView<'a, f64>,
    models: DMatrixView<'a, f64>,
}

struct CachePnuViewMut<'a, 'b: 'a> {
    surv_ins_weights: f64,
    anc: MatrixViewXx3<'a, f64>,
    pnu: &'a mut DVectorViewMut<'b, f64>,
}
struct CachePnuChildView<'a> {
    pnu: DVectorView<'a, f64>,
}

struct CacheC0ViewMut<'a> {
    surv_ins_weights: f64,
    c0_f1: &'a mut f64,
    c0_pnu: &'a mut f64,
}
struct CacheC0ChildView {
    c0_f1: f64,
    c0_pnu: f64,
}

impl<Q: QMatrix> PIPCost<Q> {
    fn logl(&self) -> f64 {
        let mut tmp = self.tmp.borrow_mut();
        let PIPModelInfo {
            cache, matrix_buf, ..
        } = &mut *tmp;

        let root_idx = usize::from(self.tree().root);
        let msa_length = self.info.msa.len();

        let postorder_tree = self.tree().postorder();
        let mut invalid_leaf_nodes: Vec<usize> = Vec::with_capacity(self.info.msa.seq_count());
        let mut internal_nodes: Vec<DisjointIndices> =
            Vec::with_capacity(self.info.msa.seq_count() - 1);

        for tree_node_idx in postorder_tree.iter() {
            let number_node_idx = usize::from(tree_node_idx);
            if self.tree().dirty[number_node_idx] || !cache.models_valid[number_node_idx] {
                self.set_model(
                    self.tree().nodes[number_node_idx].blen,
                    &mut cache.models_mut(number_node_idx),
                );
                cache.valid.remove(number_node_idx);
            }
            let children = self.tree().children(tree_node_idx);
            match *tree_node_idx {
                Leaf(leaf_idx) if !cache.valid[leaf_idx] => invalid_leaf_nodes.push(leaf_idx),
                Leaf(_) => {}
                Int(internal_idx) => internal_nodes.push(DisjointIndices::new(
                    [children[0].into(), children[1].into()],
                    internal_idx,
                )),
            };
        }
        cache.models_valid.insert_range(..);

        for leaf_node_idx in invalid_leaf_nodes {
            let this_blen = self.tree().nodes[leaf_node_idx].blen;
            let mut leaf_cache = cache.entry_mut(leaf_node_idx);
            self.set_leaf(
                this_blen,
                CacheLeafViewMut {
                    ftilde: &leaf_cache.ftilde.as_view(),
                    pnu: &mut leaf_cache.pnu,
                    anc: &mut leaf_cache.anc,
                    surv_ins_weights: leaf_cache.surv_ins_weight,
                    c0_f1: leaf_cache.c0_f1,
                    c0_pnu: leaf_cache.c0_pnu,
                },
                self.info.msa.leaf_map(&NodeIdx::Leaf(leaf_node_idx)),
            );
            let parent_idx = self.tree().nodes[leaf_node_idx]
                .parent
                .expect("all internal nodes have a parent");
            cache.valid.remove(usize::from(parent_idx));
            // valid gets set all at once in the end
        }

        for internal_node in internal_nodes.iter() {
            if !cache.valid[internal_node.this()] {
                let this_blen = self.tree().nodes[internal_node.this()].blen;

                if internal_node.this() == root_idx {
                    cache.surv_ins_weights_mut()[internal_node.this()] =
                        self.model.lambda() / self.model.mu();
                    cache.anc_mut(internal_node.this()).fill_column(0, 1.0);
                } else {
                    cache.surv_ins_weights_mut()[internal_node.this()] =
                        Self::survival_insertion_weight(
                            self.model.lambda(),
                            self.model.mu(),
                            this_blen,
                        );
                    let parent_idx = self.tree().nodes[internal_node.this()]
                        .parent
                        .expect("all internal nodes have a parent");
                    cache.valid.remove(usize::from(parent_idx));
                }
            }
        }
        internal_nodes.retain(|node_idx| !cache.valid[node_idx.this()]);
        let invalid_internal_nodes = internal_nodes;

        for internal_node in invalid_internal_nodes.iter().copied() {
            let [children_cache @ .., mut this_cache] = cache.entries_mut(internal_node);

            self.set_ftilde(
                CacheFtildeViewMut {
                    f: &mut this_cache.pnu, // on purpose, f doen't get used for anything else but assign to pnu
                    ftilde: &mut this_cache.ftilde,
                },
                children_cache.map(|child| CacheFtildeChildView {
                    ftilde: child.ftilde.into(),
                    models: child.model.into(),
                }),
                matrix_buf,
            );
        }
        for internal_node in invalid_internal_nodes.iter().copied() {
            let [left_anc, right_anc, mut this_anc] = cache.indices_anc_mut(internal_node);

            self.set_ancestors(&mut this_anc, [&left_anc.into(), &right_anc.into()]);
        }
        for internal_node in invalid_internal_nodes.iter().copied() {
            let [children_cache @ .., mut this_cache] = cache.entries_mut(internal_node);

            self.set_pnu(
                CachePnuViewMut {
                    pnu: &mut this_cache.pnu,
                    surv_ins_weights: *this_cache.surv_ins_weight,
                    anc: this_cache.anc.into(),
                },
                children_cache.map(|child| CachePnuChildView {
                    pnu: child.pnu.into(),
                }),
            );
        }

        for internal_node in invalid_internal_nodes.iter().copied() {
            let this_surv_ins_weights = cache.surv_ins_weights()[internal_node.this()];
            let [children_c0 @ .., this_c0] = cache.c0_mut().left_right_this_mut(internal_node);

            self.set_c0(
                internal_node
                    .left_right()
                    .map(|child_idx| self.tree().nodes[child_idx].blen),
                CacheC0ViewMut {
                    surv_ins_weights: this_surv_ins_weights,
                    c0_f1: &mut this_c0.f1,
                    c0_pnu: &mut this_c0.pnu,
                },
                children_c0.map(|child| CacheC0ChildView {
                    c0_f1: child.f1,
                    c0_pnu: child.pnu,
                }),
            );
        }
        cache.valid.insert_range(..);

        // In certain scenarios (e.g. a completely unrelated sequence, see data/p105.msa.fa)
        // individual column probabilities become too close to 0.0 (become subnormal)
        // and the log likelihood becomes -Inf. This is mathematically reasonable, but during branch
        // length optimisation BrentOpt cannot handle it and proposes NaN branch lengths.
        // This is a workaround that sets the probability to the smallest posible positive float,
        // which is equivalent to restricting the log likelihood to f64::MIN.
        cache
            .pnu(root_idx)
            .map(|x| {
                if x == 0.0 || x.is_subnormal() {
                    *MINLOGPROB
                } else {
                    x.ln()
                }
            })
            .sum()
            + cache.c0()[root_idx].pnu
            - log_factorial_shifted(msa_length)
    }

    fn set_leaf(
        &self,
        node_blen: f64,
        CacheLeafViewMut {
            pnu,
            c0_f1,
            c0_pnu,
            surv_ins_weights,
            ftilde,
            anc,
        }: CacheLeafViewMut,
        leaf_map: &Mapping,
    ) {
        *surv_ins_weights =
            Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node_blen);
        for (i, c) in leaf_map.iter().enumerate() {
            if c.is_some() {
                anc[(i, 0)] = 1.0;
            }
        }

        let f = pnu;
        ftilde.tr_mul_to(self.model.freqs(), f);
        f.component_mul_assign(&anc.column(0));

        let pnu = f;

        *pnu *= *surv_ins_weights;
        *c0_f1 = -1.0;
        *c0_pnu = -*surv_ins_weights;
    }

    /// Dependent:
    /// - tmp.pnu of both children
    /// - tmp.anc of this node
    /// - tmp.f(actually tmp.pnu) of this node
    /// - tmp.surv_ins_weights of this node
    ///
    /// Modifies:
    /// - tmp.pnu of this node
    fn set_pnu(&self, this: CachePnuViewMut, [left, right]: [CachePnuChildView; 2]) {
        this.pnu.component_mul_assign(&this.anc.column(0));
        // the multiplication `*this.pnu *= this.surv_ins_weights` is bundled
        // into the cmpy function below
        // pnu = 1 * a `compmul` b + surv_ins_weights * pnu
        this.pnu
            .cmpy(1., &this.anc.column(1), &left.pnu, this.surv_ins_weights);
        // pnu = 1 * a `compmul` b + 1. * pnu
        this.pnu.cmpy(1., &this.anc.column(2), &right.pnu, 1.);
    }

    /// Dependent:
    /// - tmp.models for both children
    /// - tmp.ftilde for both children
    ///
    /// Modifies:
    /// - tmp.ftilde for this node
    /// - tmp.f for this node
    fn set_ftilde(
        &self,
        this: CacheFtildeViewMut,
        [left, right]: [CacheFtildeChildView; 2],
        ftilde_buf: &mut DMatrix<f64>,
    ) {
        left.models.mul_to(&left.ftilde, this.ftilde);
        right.models.mul_to(&right.ftilde, ftilde_buf);
        this.ftilde.component_mul_assign(ftilde_buf);

        this.ftilde.tr_mul_to(self.model.freqs(), this.f);
    }

    fn set_model(&self, node_blen: f64, models: &mut DMatrixViewMut<f64>) {
        self.model.p_to(node_blen, models);
    }

    /// Dependent:
    /// - tmp.anc of both children
    ///
    /// Modifies:
    /// - tmp.anc of this node
    fn set_ancestors(
        &self,
        this_anc: &mut MatrixViewMutXx3<f64>,
        [left_anc, right_anc]: [&MatrixViewXx3<f64>; 2],
    ) {
        this_anc.set_column(1, &left_anc.column(0));
        this_anc.set_column(2, &right_anc.column(0));
        left_anc
            .column(0)
            .add_to(&right_anc.column(0), &mut this_anc.column_mut(0));

        for i in 0..this_anc.nrows() {
            debug_assert!((0.0..=2.0).contains(&this_anc[(i, 0)]));
            if this_anc[(i, 0)] == 2.0 {
                this_anc.fill_row(i, 0.0);
                this_anc[(i, 0)] = 1.0;
            }
        }
    }

    /// Dependent:
    /// - blen of both children
    /// - tmp.c0_f1 of both children
    /// - tmp.c0_pnu of both children
    ///
    /// Modifies:
    /// - tmp.c0_f1 of this node
    /// - tmp.c0_pnu of this node
    fn set_c0(
        &self,
        [left_blen, right_blen]: [f64; 2],
        this: CacheC0ViewMut,
        [left, right]: [CacheC0ChildView; 2],
    ) {
        let mu = self.model.mu();
        *this.c0_f1 = (1.0 + (-mu * left_blen).exp() * left.c0_f1)
            * (1.0 + (-mu * right_blen).exp() * right.c0_f1)
            - 1.0;

        *this.c0_pnu = this.surv_ins_weights * *this.c0_f1 + left.c0_pnu + right.c0_pnu;
    }

    fn survival_insertion_weight(lambda: f64, mu: f64, b: f64) -> f64 {
        // A function equal to old
        // nu * insertion_probability(tree_length, b, mu) * survival_probablitily(mu, b)
        lambda / mu * (1.0 - (-b * mu).exp())
    }
}

mod indices {
    // to disallow direct member access that violates invariants
    mod disjoint_indices {
        #[derive(Clone, Copy)]
        pub struct DisjointIndices {
            left_right_this: [usize; 3],
        }

        impl DisjointIndices {
            pub fn new([left, right]: [usize; 2], this: usize) -> Self {
                assert_ne!(left, right);
                assert_ne!(left, this);
                assert_ne!(right, this);
                Self {
                    left_right_this: [left, right, this],
                }
            }
            pub fn left_right_this(&self) -> [usize; 3] {
                self.left_right_this
            }
            pub fn left_right(&self) -> [usize; 2] {
                [self.left_right_this[0], self.left_right_this[1]]
            }
            pub fn this(&self) -> usize {
                self.left_right_this[2]
            }
        }
    }
    mod disjoint_ranges {
        use std::ops::Range;

        use super::DisjointIndices;

        #[derive(Clone)]
        pub struct DisjointRanges {
            left_right_this: [Range<usize>; 3],
        }

        impl DisjointRanges {
            const LEFT: usize = 0;
            const RIGHT: usize = 1;
            const THIS: usize = 2;
            pub fn new(indices: DisjointIndices, len: usize) -> Self {
                Self {
                    left_right_this: indices
                        .left_right_this()
                        .map(|idx| (idx * len)..((idx + 1) * len)),
                }
            }
            pub fn left_right_this(&self) -> [Range<usize>; 3] {
                self.left_right_this.clone()
            }
            pub fn left_right(&self) -> [Range<usize>; 2] {
                [
                    self.left_right_this[Self::LEFT].clone(),
                    self.left_right_this[Self::RIGHT].clone(),
                ]
            }
            pub fn this(&self) -> Range<usize> {
                self.left_right_this[Self::THIS].clone()
            }
        }
    }

    pub(super) use disjoint_indices::DisjointIndices;
    pub(super) use disjoint_ranges::DisjointRanges;

    pub(super) trait NodeCacheIndexer<T>: AsMut<[T]> {
        fn left_right_this_mut(&mut self, indices: DisjointIndices) -> [&mut T; 3] {
            // NOTE: could check len and then use get_disjoint_mut_unchecked if unsafe
            // becomes necessary
            self.as_mut()
                .get_disjoint_mut(indices.left_right_this())
                .expect("indices should be distinct")
        }
        fn this(&mut self, indices: DisjointIndices) -> &T {
            &self.as_mut()[indices.this()]
        }
        fn this_mut(&mut self, indices: DisjointIndices) -> &mut T {
            &mut self.as_mut()[indices.this()]
        }
        fn left_right_mut(&mut self, indices: DisjointIndices) -> [&mut T; 2] {
            // NOTE: could check len and then use get_disjoint_mut_unchecked if unsafe
            // becomes necessary
            self.as_mut()
                .get_disjoint_mut(indices.left_right())
                .expect("indices should be distinct")
        }
    }
    impl<T> NodeCacheIndexer<T> for Box<[T]> {}
    impl<T> NodeCacheIndexer<T> for [T] {}
    pub(super) trait NodeCacheSlicer<T>: AsMut<[T]> {
        fn slice_left_right_this_mut(&mut self, indices: DisjointRanges) -> [&mut [T]; 3] {
            // NOTE: could check len and then use get_disjoint_mut_unchecked if unsafe
            // becomes necessary
            self.as_mut()
                .get_disjoint_mut(indices.left_right_this())
                .expect("indices should be distinct")
        }
        fn this(&mut self, indices: DisjointRanges) -> &[T] {
            &self.as_mut()[indices.this()]
        }
        fn this_mut(&mut self, indices: DisjointRanges) -> &mut [T] {
            &mut self.as_mut()[indices.this()]
        }
        fn left_right_mut(&mut self, indices: DisjointRanges) -> [&mut [T]; 2] {
            // NOTE: could check len and then use get_disjoint_mut_unchecked if unsafe
            // becomes necessary
            self.as_mut()
                .get_disjoint_mut(indices.left_right())
                .expect("indices should be distinct")
        }
    }
    impl<T> NodeCacheSlicer<T> for [T] {}
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
