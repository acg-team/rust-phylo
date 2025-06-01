use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::iter::{self};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
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
        (self.q() * time).exp()
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
        let aligned_len = PIPModelCacheBufDimensions::pad(len);
        let slice_in_range = idx * aligned_len..(idx * aligned_len) + len;
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
        let aligned_len = PIPModelCacheBufDimensions::pad(len);
        let slice_in_range = idx * aligned_len..(idx * aligned_len) + len;
        &mut self.as_mut()[range][slice_in_range]
    }
}

impl<T> SubSlicer<T> for [T] {}
impl<T> SubSlicerMut<T> for [T] {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PIPModelCacheBufDimensions {
    // equals (4[dna] or 20[aa]) + 1[gap]
    n: usize,
    msa_length: usize,
    node_count: usize,
}

impl PIPModelCacheBufDimensions {
    pub const ALIGNMENT_IN_U8: usize = 16;
    const ALIGNMENT_IN_F64: usize = Self::ALIGNMENT_IN_U8 / size_of::<f64>();
    const _CHECK: () = {
        assert!(Self::ALIGNMENT_IN_U8 % 8 == 0);
    };
    pub fn new(n: usize, msa_length: usize, node_count: usize) -> Self {
        let dimensions = Self {
            n,
            msa_length,
            node_count,
        };

        #[cfg(debug_assertions)]
        for ranges in dimensions.ordered().windows(2) {
            let [prev, current] = ranges else { panic!() };
            debug_assert_eq!(prev.end, current.start);
        }

        #[cfg(debug_assertions)]
        {
            let cum_dynamic_vector_and_matrix_size_in_f64 = dimensions
                .ordered()
                .map(|range| range.len())
                .iter()
                .sum::<usize>();
            assert_eq!(
                cum_dynamic_vector_and_matrix_size_in_f64,
                dimensions.ordered().last().unwrap().end
            );
        }

        dimensions
    }

    pub const fn ordered(&self) -> [Range<usize>; 6] {
        [
            self.models_range(),
            self.surv_ins_weights_range(),
            self.ftilde_range(),
            self.anc_range(),
            self.pnu_range(),
            self.c0_range(),
        ]
    }
    pub const fn total_len_f64_padded(&self) -> usize {
        let ordered = self.ordered();
        Self::pad(ordered.last().unwrap().end)
    }

    const fn pad(len: usize) -> usize {
        (len + (Self::ALIGNMENT_IN_F64 - 1)) & !(Self::ALIGNMENT_IN_F64 - 1)
    }

    const fn models_len(&self) -> usize {
        self.n * self.n
    }
    const fn models_range(&self) -> Range<usize> {
        let offset = 0;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + self.node_count * Self::pad(self.models_len())
    }

    const fn surv_ins_weights_len(&self) -> usize {
        self.node_count
    }
    const fn surv_ins_weights_range(&self) -> Range<usize> {
        let offset = self.models_range().end;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + Self::pad(self.surv_ins_weights_len())
    }

    const fn ftilde_len(&self) -> usize {
        self.n * self.msa_length
    }
    const fn ftilde_range(&self) -> Range<usize> {
        let offset = self.surv_ins_weights_range().end;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + self.node_count * Self::pad(self.ftilde_len())
    }

    const fn anc_len(&self) -> usize {
        3 * self.msa_length
    }
    const fn anc_range(&self) -> Range<usize> {
        let offset = self.ftilde_range().end;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + self.node_count * Self::pad(self.anc_len())
    }

    const fn pnu_len(&self) -> usize {
        self.msa_length
    }
    const fn pnu_range(&self) -> Range<usize> {
        let offset = self.anc_range().end;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + self.node_count * Self::pad(self.pnu_len())
    }

    const fn c0_len(&self) -> usize {
        self.node_count * 2
    }
    const fn c0_range(&self) -> Range<usize> {
        let offset = self.pnu_range().end;
        debug_assert!(offset * size_of::<f64>() % Self::ALIGNMENT_IN_U8 == 0);
        offset..offset + Self::pad(self.c0_len())
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn msa_length(&self) -> usize {
        self.msa_length
    }

    pub const fn node_count(&self) -> usize {
        self.node_count
    }
}

#[derive(Debug, PartialEq)]
pub struct PIPModelCacheBuf {
    // TODO: would need borrowing version of pipcost to do cleanly
    buf: &'static mut [f64],
    dimensions: PIPModelCacheBufDimensions,
    valid: FixedBitSet,
    models_valid: FixedBitSet,
    is_owned: bool,
}

impl Clone for PIPModelCacheBuf {
    fn clone(&self) -> Self {
        let new_cache_buf = Box::new_uninit_slice(self.buf.len());
        let new_cache_ref = Box::leak(new_cache_buf);
        // let new_cache_buf = BoxSlice::alloc_slice_uninit(NonZero::new(self.buf.len()).unwrap());
        // // safety: we manually dealloc the cache in Drop
        // let new_cache_ref = unsafe { new_cache_buf.leak() };
        new_cache_ref.copy_from_slice(unsafe {
            std::mem::transmute::<&[f64], &[MaybeUninit<f64>]>(self.buf)
        });

        let new_cache_ref =
            unsafe { std::mem::transmute::<&mut [MaybeUninit<f64>], &mut [f64]>(new_cache_ref) };
        Self {
            buf: new_cache_ref,
            dimensions: self.dimensions,
            valid: self.valid.clone(),
            models_valid: self.models_valid.clone(),
            is_owned: true,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.dimensions = source.dimensions;
        self.buf.copy_from_slice(source.buf);
        self.valid.clone_from(&source.valid);
        self.models_valid.clone_from(&source.models_valid);
    }
}

impl Drop for PIPModelCacheBuf {
    fn drop(&mut self) {
        if self.is_owned {
            // owned buffers are allocated with BoxSlice
            drop(unsafe { Box::from_raw(self.buf) });
        }
    }
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
    pub fn new_owned(dimensions: PIPModelCacheBufDimensions) -> Self {
        let storage = vec![0.0; dimensions.total_len_f64_padded()].into_boxed_slice();
        let storage_ref = Box::leak(storage);
        // let storage = BoxSlice::alloc_slice(
        //     0.0,
        //     NonZero::new(dimensions.total_len_f64_padded()).unwrap(),
        // );
        // let storage_ref = unsafe { storage.leak() };

        let mut cache = Self::new(dimensions, storage_ref);
        cache.is_owned = true;

        cache
    }
    pub fn new(dimensions: PIPModelCacheBufDimensions, storage: &'static mut [f64]) -> Self {
        assert_eq!(storage.len(), dimensions.total_len_f64_padded());
        Self {
            is_owned: false,
            buf: storage,
            dimensions,
            valid: FixedBitSet::with_capacity(dimensions.node_count),
            models_valid: FixedBitSet::with_capacity(dimensions.node_count),
        }
    }

    pub fn clone_in(&self, buf: &'static mut [MaybeUninit<f64>]) -> Self {
        buf.copy_from_slice(unsafe {
            std::mem::transmute::<&[f64], &[MaybeUninit<f64>]>(self.buf)
        });
        let buf = unsafe {
            std::mem::transmute::<&'static mut [MaybeUninit<f64>], &'static mut [f64]>(buf)
        };
        Self {
            buf,
            dimensions: self.dimensions,
            valid: self.valid.clone(),
            models_valid: self.models_valid.clone(),
            is_owned: false,
        }
    }

    fn idx_len_to_range(idx: usize, len: usize) -> Range<usize> {
        let aligned_len = PIPModelCacheBufDimensions::pad(len);
        idx * aligned_len..(idx * aligned_len) + len
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
            let disjoint_ranges = DisjointRanges::new_aligned(indices, item_len);
            let sub_slices = models_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.n)
            })
        };
        let surv_ins_weights = surv_ins_weights_slice.left_right_this_mut(indices);

        let ftildes = {
            let item_len = self.dimensions.ftilde_len();

            let disjoint_ranges = DisjointRanges::new_aligned(indices, item_len);
            let sub_slices = ftildes_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                DMatrixViewMut::from_slice(sub_slice, self.dimensions.n, self.dimensions.msa_length)
            })
        };
        let ancs = {
            let item_len = self.dimensions.anc_len();

            let disjoint_ranges = DisjointRanges::new_aligned(indices, item_len);
            let sub_slices = ancs_slice.slice_left_right_this_mut(disjoint_ranges);

            sub_slices.map(|sub_slice| {
                MatrixViewMutXx3::from_slice(sub_slice, self.dimensions.msa_length)
            })
        };

        let pnus = {
            let item_len = self.dimensions.pnu_len();

            let disjoint_ranges = DisjointRanges::new_aligned(indices, item_len);
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

    fn models_mut(&mut self, idx: usize) -> DMatrixViewMut<f64> {
        let range = self.dimensions.models_range();
        let item_len = self.dimensions.models_len();

        DMatrixViewMut::from_slice(
            self.buf.slice_mut_idx_with_len_in(range, idx, item_len),
            self.dimensions.n,
            self.dimensions.n,
        )
    }

    #[allow(dead_code)]
    fn surv_ins_weights(&self) -> &[f64] {
        let range = self.dimensions.surv_ins_weights_range();
        &self.buf[range][..self.dimensions.surv_ins_weights_len()]
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

    #[allow(dead_code)]
    fn anc(&self, idx: usize) -> MatrixViewXx3<f64> {
        let range = self.dimensions.anc_range();
        let item_len = self.dimensions.anc_len();

        MatrixViewXx3::from_slice(
            self.buf.slice_idx_with_len_in(range, idx, item_len),
            self.dimensions.msa_length,
        )
    }

    fn pnu(&self, idx: usize) -> DVectorView<f64> {
        let range = self.dimensions.pnu_range();
        let item_len = self.dimensions.pnu_len();

        DVectorView::from_slice(
            self.buf.slice_idx_with_len_in(range, idx, item_len),
            self.dimensions.msa_length,
        )
    }

    fn c0(&self) -> &[C0Scalars] {
        let range = self.dimensions.c0_range();
        bytemuck::cast_slice(&self.buf[range][..self.dimensions.c0_len()])
    }

    pub const fn dimensions(&self) -> PIPModelCacheBufDimensions {
        self.dimensions
    }
}

#[derive(Debug, PartialEq)]
pub struct PIPModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    cache: PIPModelCacheBuf,
    matrix_buf: DMatrix<f64>,
}

impl<Q: QMatrix> Clone for PIPModelInfo<Q> {
    fn clone(&self) -> Self {
        Self {
            phantom: self.phantom,
            cache: self.cache.clone(),
            matrix_buf: self.matrix_buf.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.phantom = source.phantom;
        self.cache.clone_from(&source.cache);
        // NOTE: contents of matrix buf are irrelevant
    }
}

impl<Q: QMatrix> PIPModelInfo<Q> {
    pub const fn dimensions(&self) -> PIPModelCacheBufDimensions {
        self.cache.dimensions()
    }
    pub fn new(info: &PhyloInfo, model: &PIPModel<Q>) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        let dimensions = PIPModelCacheBufDimensions::new(n, msa_length, node_count);
        let mut cache_entries = PIPModelCacheBuf::new_owned(dimensions);
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

#[derive(Debug)]
pub struct PIPCost<Q: QMatrix> {
    pub(crate) model: PIPModel<Q>,
    pub(crate) info: PhyloInfo,
    tmp: RefCell<PIPModelInfo<Q>>,
}

impl<Q: QMatrix> Clone for PIPCost<Q> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            info: self.info.clone(),
            tmp: self.tmp.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.model.clone_from(&source.model);
        self.info.clone_from(&source.info);
        self.tmp.borrow_mut().clone_from(&*source.tmp.borrow());
    }
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
    pub fn cache_dimensions(&self) -> PIPModelCacheBufDimensions {
        self.tmp.borrow().dimensions()
    }
    pub fn clone_with_cache_in(&self, buf: &'static mut [MaybeUninit<f64>]) -> Self {
        debug_assert_eq!(buf.len(), self.cache_dimensions().total_len_f64_padded());
        Self {
            model: self.model.clone(),
            info: self.info.clone(),
            tmp: RefCell::new(PIPModelInfo {
                phantom: PhantomData,
                cache: self.tmp.borrow().cache.clone_in(buf),
                matrix_buf: self.tmp.borrow().matrix_buf.clone(),
            }),
        }
    }
    fn logl(&self) -> f64 {
        let mut tmp = self.tmp.borrow_mut();
        let PIPModelInfo {
            cache, matrix_buf, ..
        } = &mut *tmp;

        let root_idx = usize::from(self.tree().root);
        let msa_length = self.info.msa.len();

        for node_idx in self.tree().postorder() {
            let number_node_idx = usize::from(node_idx);
            if self.tree().dirty[number_node_idx] || !cache.models_valid[number_node_idx] {
                self.set_model(
                    self.tree().nodes[number_node_idx].blen,
                    &mut cache.models_mut(number_node_idx),
                );
                cache.models_valid.insert(number_node_idx);
                cache.valid.remove(number_node_idx);
            }

            let this_blen = self.tree().nodes[number_node_idx].blen;

            if !cache.valid[number_node_idx] {
                match node_idx {
                    Int(_) => {
                        let children = &self.tree().nodes[number_node_idx].children;
                        let children = [children[0], children[1]].map(usize::from);
                        let indices =
                            DisjointIndices::new([children[0], children[1]], number_node_idx);
                        let children_blen =
                            children.map(|child_idx| self.tree().nodes[child_idx].blen);

                        let mut this_cache = cache.entry_mut(indices.this());

                        if root_idx == number_node_idx {
                            *this_cache.surv_ins_weight = self.model.lambda() / self.model.mu();
                            this_cache.anc.fill_column(0, 1.0);
                        } else {
                            *this_cache.surv_ins_weight = Self::survival_insertion_weight(
                                self.model.lambda(),
                                self.model.mu(),
                                this_blen,
                            );
                            let parent_idx = self.tree().nodes[number_node_idx]
                                .parent
                                .expect("all internal nodes have a parent");
                            cache.valid.remove(usize::from(parent_idx));
                        }
                        self.set_internal_common(
                            cache.entries_mut(indices),
                            matrix_buf,
                            children_blen,
                        );
                    }
                    Leaf(_) => {
                        let mut leaf_cache = cache.entry_mut(number_node_idx);
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
                            self.info.msa.leaf_map(node_idx),
                        );
                        let parent_idx = self.tree().nodes[number_node_idx]
                            .parent
                            .expect("all leaf nodes have a parent");
                        cache.valid.remove(usize::from(parent_idx));
                    }
                };
                cache.valid.insert(number_node_idx)
            }
        }

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

    fn set_internal_common(
        &self,
        [left_cache, right_cache, mut this_cache]: [PIPModelCacheEntryViewMut; 3],
        matrix_buf: &mut DMatrix<f64>,
        children_blen: [f64; 2],
    ) {
        self.set_ftilde(
            CacheFtildeViewMut {
                f: &mut this_cache.pnu, // on purpose, f doen't get used for anything else but assign to pnu
                ftilde: &mut this_cache.ftilde,
            },
            itertools::izip!(
                [left_cache.ftilde, right_cache.ftilde],
                [left_cache.model, right_cache.model]
            )
            .map(|(ftilde, model)| CacheFtildeChildView {
                ftilde: ftilde.into(),
                models: model.into(),
            })
            .collect_array::<2>()
            .unwrap(),
            matrix_buf,
        );
        self.set_ancestors(
            &mut this_cache.anc,
            [&left_cache.anc.as_view(), &right_cache.anc.as_view()],
        );
        self.set_pnu(
            CachePnuViewMut {
                pnu: &mut this_cache.pnu,
                surv_ins_weights: *this_cache.surv_ins_weight,
                anc: this_cache.anc.as_view(),
            },
            [left_cache.pnu, right_cache.pnu].map(|pnu| CachePnuChildView { pnu: pnu.into() }),
        );
        self.set_c0(
            children_blen,
            CacheC0ViewMut {
                surv_ins_weights: *this_cache.surv_ins_weight,
                c0_f1: this_cache.c0_f1,
                c0_pnu: this_cache.c0_pnu,
            },
            itertools::izip!(
                [left_cache.c0_f1, right_cache.c0_f1],
                [left_cache.c0_pnu, right_cache.c0_pnu]
            )
            .map(|(c0_f1, c0_pnu)| CacheC0ChildView {
                c0_f1: *c0_f1,
                c0_pnu: *c0_pnu,
            })
            .collect_array::<2>()
            .unwrap(),
        );
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
        let f = (self.model.q() * node_blen).exp();
        // f.fill(node_blen);
        models.copy_from(&f);
        // self.model.p_to(node_blen, models);
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
            pub fn this(&self) -> usize {
                self.left_right_this[2]
            }
        }
    }
    mod disjoint_ranges {
        use std::ops::Range;

        use crate::pip_model::PIPModelCacheBufDimensions;

        use super::DisjointIndices;

        #[derive(Clone)]
        pub struct DisjointRanges {
            left_right_this: [Range<usize>; 3],
        }

        #[allow(dead_code)]
        impl DisjointRanges {
            const LEFT: usize = 0;
            const RIGHT: usize = 1;
            const THIS: usize = 2;
            pub fn new_aligned(indices: DisjointIndices, len: usize) -> Self {
                let aligned_len = PIPModelCacheBufDimensions::pad(len);
                Self {
                    left_right_this: indices
                        .left_right_this()
                        // only the start of the matrix is aligned
                        .map(|idx| (idx * aligned_len)..((idx * aligned_len) + len)),
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
    }
    impl<T> NodeCacheIndexer<T> for [T] {}
    pub(super) trait NodeCacheSlicer<T>: AsMut<[T]> {
        fn slice_left_right_this_mut(&mut self, indices: DisjointRanges) -> [&mut [T]; 3] {
            // NOTE: could check len and then use get_disjoint_mut_unchecked if unsafe
            // becomes necessary
            self.as_mut()
                .get_disjoint_mut(indices.left_right_this())
                .expect("indices should be distinct")
        }
    }
    impl<T> NodeCacheSlicer<T> for [T] {}
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
