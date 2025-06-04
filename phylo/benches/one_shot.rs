use std::fmt::Display;
use std::result::Result::Ok;
use std::time::Instant;

use anyhow::Result;

use log::info;

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::{ModelSearchCost, TreeSearchCost};
use phylo::optimisers::{ModelOptimiser, TopologyOptimiser, TopologyOptimiserPredicate};
use phylo::substitution_models::{QMatrix, JC69};
use phylo::tree::Tree;
mod helpers;
use crate::helpers::{
    HIV_1, N100_M500, N10_M1000, N10_M10000, N10_M100000, N10_M20000, N10_M200000, N10_M300000,
    N10_M500, N10_M5000, N10_M50000, N200_M500, N20_M500, N300_M500, N30_M500, N400_M500, N40_M500,
    N50_M500, N60_M500, N70_M500, N80_M500, N90_M500,
};
use helpers::{black_box_raw_pip_cost_with_config, SequencePaths};
use phylo::pip_model::PIPCost;

/// NOTE: This is essentially a snapshot of the JATI binary at the time of creation
/// ANY changes must be manually tracked to accurately measure real-world usage.
///
/// TODO: expose this as part of the rust-phylo library
fn run_optimisation(
    cost: &PIPCost<impl QMatrix>,
    freq_opt: FrequencyOptimisation,
    max_iterations: usize,
    epsilon: f64,
) -> Result<(f64, Tree)> {
    // needs to be intialized first for now because it holds the master cache
    let mut topo_opt = TopologyOptimiser::new_with_pred_inplace(
        cost,
        TopologyOptimiserPredicate::gt_epsilon(1e-3),
    );
    let mut prev_cost = f64::NEG_INFINITY;
    let mut final_cost = TreeSearchCost::cost(topo_opt.base_cost_fn());

    let mut iterations = 0;
    while final_cost - prev_cost > epsilon && iterations < max_iterations {
        iterations += 1;
        info!("Iteration: {}", iterations);

        prev_cost = final_cost;
        ModelOptimiser::new(topo_opt.base_cost_fn_mut(), freq_opt).run()?;

        // resulting cost_fn is in base_cost_fn
        let o = topo_opt.run_mut().unwrap();
        final_cost = o.final_cost;
    }
    Ok((final_cost, topo_opt.base_cost_fn().tree().clone()))
}

#[test]
fn test_one_shot_increasing_length() {
    let paths = SequencePaths::from([
        ("10X1000", N10_M1000),
        ("10X5000", N10_M5000),
        ("10X10000", N10_M10000),
        ("10X20000", N10_M20000),
        ("10X50000", N10_M50000),
        ("10X100000", N10_M100000),
        ("10X200000", N10_M200000),
    ]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}

#[test]
fn test_one_shot_increasing_taxa() {
    let paths = SequencePaths::from([
        ("10X500", N10_M500),
        ("20X500", N20_M500),
        ("30X500", N30_M500),
        ("40X500", N40_M500),
        ("50X500", N50_M500),
    ]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}

#[test]
fn test_one_shot_increasing_taxa_large() {
    let paths = SequencePaths::from([
        ("50X500", N50_M500),
        ("100X500", N100_M500),
        ("200X500", N200_M500),
    ]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}

#[test]
fn test_one_shot_increasing_taxa_vlarge() {
    let paths = SequencePaths::from([("300X500", N300_M500)]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}

#[test]
fn test_one_shot_increasing_taxa_xlarge() {
    let paths = SequencePaths::from([("400X500", N400_M500)]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}

#[test]
fn test_one_shot_real_world() {
    let paths = SequencePaths::from([("250x4177", HIV_1)]);
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<JC69>(path);
        let start = Instant::now();
        run_optimisation(&data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon)
            .expect("Should have run");
        let duration = start.elapsed();
        println!("{}", key);
        println!("{:#?}", duration);
    }
}
