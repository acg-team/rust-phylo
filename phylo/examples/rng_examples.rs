use rand::distr::weighted::WeightedIndex;
use rand::rngs::{SmallRng, StdRng};
use std::iter::repeat_n;

use phylo::random::{DefaultGenerator, RandomGenerator};

fn main() {
    // 1. Using the default RNG (StdRng-based)
    let seed = 42;
    let mut rng = DefaultGenerator::new(seed);
    println!("1. RNG (StdRng) with seed {} - reproducible:", rng.seed());
    for i in 0..3 {
        println!("  {i}: {:.6}", rng.random::<f64>());
    }
    // Reset and show reproducibility
    rng.reseed(seed);
    println!("After resetting to seed {}:", rng.seed());
    for i in 0..3 {
        println!("  {i}: {:.6}", rng.random::<f64>());
    }
    println!();

    // 2. Generating from default RNG:
    println!("2. Generating from default RNG:");
    let mut rng = DefaultGenerator::default();
    println!("  Random f64: {:.6}", rng.random::<f64>());
    println!("  Random u32: {}", rng.random::<u32>());
    println!("  Random i32: {}", rng.random::<i32>());
    println!("  Probability [0,1): {:.6}", rng.random::<f64>());
    println!("  Range 1-100: {}", rng.random_range(1..=100));
    println!("  Range 0.0-10.0: {:.3}", rng.random_range(0.0..10.0));
    println!("  Boolean (50%): {}", rng.random_bool(0.5));
    println!("  Boolean (25%): {}", rng.random_bool(0.25));
    let dist = WeightedIndex::new(repeat_n(1.0, 15)).unwrap();
    let sample = rng.sample(&dist);
    println!("  Sample from a uniform weighted distribution: {}", sample);

    println!();

    // 3. Creating a custom StdRng instance
    let seed = 123;
    println!("3. Custom StdRng instance with seed {seed} - reproducible:");
    let mut custom_std_rng = RandomGenerator::<StdRng>::new(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", custom_std_rng.random::<f64>());
    }
    // Reset and show reproducibility
    custom_std_rng.reseed(seed);
    println!("After resetting to seed {seed}:");
    for i in 0..3 {
        println!("  {i}: {:.6}", custom_std_rng.random::<f64>());
    }
    println!();

    // 4. Generating from the custom StdRng instance
    println!("4. Generating from custom RNG:");
    println!("  Random u32: {}", custom_std_rng.random::<u32>());
    println!("  Random i32: {}", custom_std_rng.random::<i32>());
    println!("  Probability [0,1): {:.6}", custom_std_rng.random::<f64>());
    println!("  Range 1-100: {}", custom_std_rng.random_range(1..=100));
    println!(
        "  Range 0.0-10.0: {:.3}",
        custom_std_rng.random_range(0.0..10.0)
    );
    println!("  Boolean (50%): {}", custom_std_rng.random_bool(0.5));
    println!("  Boolean (25%): {}", custom_std_rng.random_bool(0.25));
    println!();

    // 5. Using a different seed for another StdRng instance
    let seed = 456;

    let mut another_std_rng: RandomGenerator<StdRng> = RandomGenerator::new(seed);
    println!(
        "5. Another StdRng instance with different seed {}:",
        another_std_rng.seed()
    );
    for i in 0..3 {
        println!("  {i}: {:.6}", another_std_rng.random::<f64>());
    }
    println!();

    // 6. Creating a custom RNG instance with a different generator
    println!("6. Custom RNG instance with SmallRng with seed {seed}:");
    let mut secure_rng: RandomGenerator<SmallRng> = RandomGenerator::new(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", secure_rng.random::<f64>());
    }
    println!();
}
