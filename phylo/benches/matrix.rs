use std::{
    hint::black_box,
    num::NonZero,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, Criterion};
use phylo::{
    pip_model::{PIPModelCacheBuf, PIPModelCacheBufDimensions},
    util::mem::boxed::BoxSlice,
};
mod helpers;
use helpers::{
    SequencePaths, DNA_EASY_17X2292, DNA_EASY_33X4455, DNA_EASY_46X16250, DNA_EASY_5X1000,
    DNA_EASY_8X1252, DNA_MEDIUM_128X688,
};

const DNA_N: usize = 5;

fn run_for_sizes(
    dimensions: &[PIPModelCacheBufDimensions],
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, data: PIPModelCacheBuf| {
        bench_group.bench_function(id, |bench| {
            bench.iter(|| data.clone());
        });
    };
    for n_items in (0..100 * 1024).step_by(10).map(|n| n * 1024).skip(1) {
        let chunk = black_box(vec![0.0; n_items].into_boxed_slice());
        let now = Instant::now();
        let _chunk2 = black_box(chunk.clone());
        let took = now.elapsed();
        drop(_chunk2);

        let chunk_huge = BoxSlice::alloc_slice(0.0, NonZero::try_from(n_items).unwrap());
        let now_huge = Instant::now();
        let _chunk_huge2 = black_box(chunk_huge.clone());
        let took_huge = now_huge.elapsed();
        drop(_chunk_huge2);

        let now_huge_manual = Instant::now();
        let _chunk_huge_manual2 = black_box(chunk_huge.clone_manual());
        let took_huge_manual = now_huge_manual.elapsed();
        drop(_chunk_huge_manual2);

        let chunk_huge_transparent =
            BoxSlice::alloc_slice_transparent_hugepages(0.0, NonZero::try_from(n_items).unwrap());
        let now_huge_transparent = Instant::now();
        let _chunk_huge_transparent2 = black_box(chunk_huge_transparent.clone());
        let took_huge_transparent = now_huge_transparent.elapsed();
        drop(_chunk_huge_transparent2);

        let n_chunks = n_items.div_ceil(10 * 1024); // 10kb chunk
        let chunk_multiple = black_box(
            vec![black_box(vec![0.0; 10 * 1024].into_boxed_slice()); n_chunks].into_boxed_slice(),
        );
        let now_multiple = Instant::now();
        let _chunk_multiple2 = black_box(chunk_multiple.clone());
        let took_multiple = now_multiple.elapsed();
        drop(_chunk_multiple2);

        let size_kb = (size_of::<f64>() * n_items) / 1024;
        println!(
            "size(kb): {size_kb:0>6} clone takes(us) {:0>6} vs huge {:0>6} vs huge-manual {:0>6} vs huge-transparent {:0>6} vs multiple {:0>6}",
            took.as_micros(),
            took_huge.as_micros(),
            took_huge_manual.as_micros(),
            took_huge_transparent.as_micros(),
            took_multiple.as_micros(),
        );
    }

    // for dimension in dimensions {
    //     bench(
    //         &format!(
    //             "n={}_msa_len={}_node_count={}-TOTAL={}",
    //             dimension.n(),
    //             dimension.msa_length(),
    //             dimension.node_count(),
    //             dimension
    //                 .ordered()
    //                 .iter()
    //                 .map(std::ops::Range::len)
    //                 .sum::<usize>()
    //         ),
    //         black_box(PIPModelCacheBuf::new(
    //             black_box(dimension.n()),
    //             black_box(dimension.msa_length()),
    //             black_box(dimension.node_count()),
    //         )),
    //     );
    // }
    bench_group.finish();
}

fn pip_cost_dna_easy(criterion: &mut Criterion) {
    run_for_sizes(
        &[
            (1_000, 5),
            (1_252, 8),
            (2_292, 17),
            (4_455, 33),
            (10_000, 40),
            (15_000, 40),
            (16_250, 46),
            (688, 128),
        ]
        .map(|(msa_len, n_seqs)| PIPModelCacheBufDimensions::new(DNA_N, msa_len, 2 * n_seqs - 1)),
        "PIP CACHE BUF CLONE",
        criterion,
    );
}

criterion_group! {
name = matrix_dna;
config = helpers::setup_suite().measurement_time(Duration::from_secs(10));
targets = pip_cost_dna_easy,
}
criterion_main!(matrix_dna);
