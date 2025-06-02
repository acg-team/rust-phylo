mod unused_experiments {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    pub struct PIPModelCacheSOA {
        surv_ins_weights: Box<[f64]>,
        anc: Box<[MatrixXx3<f64>]>,
        ftilde: Box<[DMatrix<f64>]>,
        pnu: Box<[DVector<f64>]>,
        c0_f1: Box<[f64]>,
        c0_pnu: Box<[f64]>,
        models: Box<[SubstMatrix]>,
        valid: FixedBitSet,
        models_valid: FixedBitSet,
        length: usize,
    }

    impl PIPModelCacheSOA {
        fn make_array<C: Clone>(item: C, count: usize) -> Box<[C]> {
            repeat_n(item, count).collect()
        }
        fn make_default(n: usize, msa_length: usize, node_count: usize) -> Self {
            Self {
                length: node_count,
                ftilde: Self::make_array(DMatrix::<f64>::zeros(n, msa_length), node_count),
                surv_ins_weights: Self::make_array(0.0, node_count),
                pnu: Self::make_array(DVector::<f64>::zeros(msa_length), node_count),
                c0_f1: Self::make_array(0.0, node_count),
                c0_pnu: Self::make_array(0.0, node_count),
                anc: Self::make_array(MatrixXx3::<f64>::zeros(msa_length), node_count),
                models: Self::make_array(SubstMatrix::zeros(n, n), node_count),
                valid: FixedBitSet::with_capacity(node_count),
                models_valid: FixedBitSet::with_capacity(node_count),
            }
        }
    }

    #[derive(Debug)]
    pub struct PIPModelCacheContinuous {
        cache: PIPModelCacheSOARefs<'static>,
        alloc: NonNull<bumpalo::Bump>,
        valid: FixedBitSet,
        models_valid: FixedBitSet,
    }

    impl PIPModelCacheContinuous {
        pub fn new(
            n: usize,
            msa_length: usize,
            node_count: usize,
            alloc: Box<bumpalo::Bump>,
        ) -> Self {
            let mut alloc = NonNull::from(Box::leak(alloc));

            // safety:
            // - alloc is a non-null pointer initialized to a valid heap object one line above
            let cache = PIPModelCacheSOARefs::make_default(n, msa_length, node_count, unsafe {
                alloc.as_mut()
            });

            Self {
                cache,
                alloc,
                valid: FixedBitSet::with_capacity(node_count),
                models_valid: FixedBitSet::with_capacity(node_count),
            }
        }
    }

    impl Drop for PIPModelCacheContinuous {
        fn drop(&mut self) {
            // safety:
            // alloc is never changed after inialization so still points to a valid
            // heap object of the correct layout and type
            let owned_alloc = unsafe { Box::from_raw(self.alloc.as_ptr()) };
            drop(owned_alloc);
            // NOTE: no need to drop the cache, it only contains references into the allocator
        }
    }

    #[derive(Debug)]
    pub struct PIPModelCacheSOARefs<'a> {
        surv_ins_weights: &'a mut [f64],
        anc: &'a mut [MatrixViewMutXx3<'a, f64>],
        ftilde: &'a mut [DMatrixViewMut<'a, f64>],
        pnu: &'a mut [DVectorViewMut<'a, f64>],
        c0_f1: &'a mut [f64],
        c0_pnu: &'a mut [f64],
        models: &'a mut [DMatrixViewMut<'a, f64>],
    }

    impl<'a> PIPModelCacheSOARefs<'a> {
        fn make_default(
            n: usize,
            msa_length: usize,
            node_count: usize,
            buf: &'a mut bumpalo::Bump,
        ) -> PIPModelCacheSOARefs<'a> {
            let cum_size_of_array_items = size_of::<DMatrix<f64>>()
                + size_of::<f64>()
                + size_of::<DVector<f64>>()
                + 2 * size_of::<f64>()
                + size_of::<MatrixXx3<f64>>()
                + size_of::<SubstMatrix>();
            let cum_dynamic_vector_and_matrix_size =
                [n * msa_length, msa_length, 3 * msa_length, n * n]
                    .iter()
                    .sum::<usize>()
                    * size_of::<f64>();

            let total_heap_size =
                node_count * (cum_size_of_array_items + cum_dynamic_vector_and_matrix_size);
            assert_eq!(buf.chunk_capacity(), total_heap_size);
            buf.set_allocation_limit(Some(0));

            let ftilde = buf.alloc_slice_fill_with(node_count, |_idx| {
                DMatrixViewMut::from_slice(
                    buf.alloc_slice_fill_copy(n * msa_length, 0.0f64),
                    n,
                    msa_length,
                )
            });
            let pnu = buf.alloc_slice_fill_with(node_count, |_idx| {
                DVectorViewMut::from_slice(
                    buf.alloc_slice_fill_copy(msa_length, 0.0f64),
                    msa_length,
                )
            });
            let anc = buf.alloc_slice_fill_with(node_count, |_idx| {
                MatrixViewMutXx3::from_slice(
                    buf.alloc_slice_fill_copy(3 * msa_length, 0.0f64),
                    msa_length,
                )
            });
            let models = buf.alloc_slice_fill_with(node_count, |_idx| {
                DMatrixViewMut::from_slice(buf.alloc_slice_fill_copy(n * n, 0.0f64), n, n)
            });

            Self {
                ftilde,
                surv_ins_weights: buf.alloc_slice_fill_copy(node_count, 0.0f64),
                pnu,
                c0_f1: buf.alloc_slice_fill_copy(node_count, 0.0f64),
                c0_pnu: buf.alloc_slice_fill_copy(node_count, 0.0f64),
                anc,
                models,
            }
        }
    }
}
