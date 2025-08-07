use rand::rngs::{SmallRng, StdRng};

use phylo::random::{
    gen, gen_probability, gen_range, global_rng, init_rng, GlobalRng, RandomGenerator,
};

fn main() {
    // 1. Using the global RNG (StdRng-based)
    let seed = 42;
    println!("1. Global RNG (StdRng) with seed {seed} - reproducible:");
    init_rng(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", gen::<f64>());
    }
    // Reset and show reproducibility
    init_rng(seed);
    println!("After resetting to seed {seed}:");
    for i in 0..3 {
        println!("  {i}: {:.6}", gen::<f64>());
    }
    println!();

    // 2. Generating from global RNG:
    println!("2. Generating from global RNG:");
    println!("  Random f64: {:.6}", global_rng().gen::<f64>());
    println!("  Random u32: {}", global_rng().gen::<u32>());
    println!("  Random i32: {}", global_rng().gen::<i32>());
    println!("  Probability [0,1): {:.6}", global_rng().gen_f64());
    println!("  Range 1-100: {}", global_rng().gen_range(1..=100));
    println!("  Range 0.0-10.0: {:.3}", global_rng().gen_range(0.0..10.0));
    println!("  Boolean (50%): {}", global_rng().gen_bool(0.5));
    println!("  Boolean (25%): {}", global_rng().gen_bool(0.25));
    println!();

    // 3. Generating using global RNG functions
    println!("3. Generating using global RNG functions:");
    println!("  Probability [0,1): {:.6}", gen_probability());
    println!("  Random u32: {}", gen::<u32>());
    println!("  Random i32: {}", gen::<i32>());
    println!("  Range 1-10: {}", gen_range(1..=10));
    println!("  Range -5.0 to 5.0: {:.3}", gen_range(-5.0..5.0));
    println!();

    // 4. Creating a custom StdRng instance
    let seed = 123;
    println!("4. Custom StdRng instance with seed {seed} - reproducible:");
    let custom_std_rng: GlobalRng = RandomGenerator::new(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", custom_std_rng.gen::<f64>());
    }
    // Reset and show reproducibility
    custom_std_rng.reseed(seed);
    println!("After resetting to seed {seed}:");
    for i in 0..3 {
        println!("  {i}: {:.6}", custom_std_rng.gen::<f64>());
    }
    println!();

    // 5. Generating from the custom StdRng instance
    println!("5. Generating from custom RNG:");
    println!("  Random u32: {}", custom_std_rng.gen::<u32>());
    println!("  Random i32: {}", custom_std_rng.gen::<i32>());
    println!("  Probability [0,1): {:.6}", custom_std_rng.gen_f64());
    println!("  Range 1-100: {}", custom_std_rng.gen_range(1..=100));
    println!(
        "  Range 0.0-10.0: {:.3}",
        custom_std_rng.gen_range(0.0..10.0)
    );
    println!("  Boolean (50%): {}", custom_std_rng.gen_bool(0.5));
    println!("  Boolean (25%): {}", custom_std_rng.gen_bool(0.25));
    println!();

    // 6. Using a different seed for another StdRng instance
    let seed = 456;
    println!("6. Another StdRng instance with different seed {seed}:");
    let another_std_rng: RandomGenerator<StdRng> = RandomGenerator::new(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", another_std_rng.gen::<f64>());
    }
    println!();

    // 7. Creating a custom RNG instance with a different generator
    println!("7. Custom RNG instance with SmallRng with seed {seed}:");
    let secure_rng: RandomGenerator<SmallRng> = RandomGenerator::new(seed);
    for i in 0..3 {
        println!("  {i}: {:.6}", secure_rng.gen::<f64>());
    }
    println!();
}
