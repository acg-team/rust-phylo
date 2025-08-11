use std::sync::Mutex;

use ntimestamp::Timestamp;
use rand::distributions::{
    uniform::{SampleRange, SampleUniform},
    Distribution, Standard,
};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Trait for random number generation
pub trait RandomSource {
    /// Generate a random value of type T.
    fn gen<T>(&self) -> T
    where
        T: 'static,
        Standard: Distribution<T>;

    /// Generate a random value in the specified range.
    fn gen_range<T, Range>(&self, range: Range) -> T
    where
        T: 'static + SampleUniform,
        Range: SampleRange<T>;

    /// Generate a random bool with probability p.
    fn gen_bool(&self, p: f64) -> bool;

    /// Generate a random uniform probability in the range [0.0, 1.0).
    fn gen_probability(&self) -> f64;

    /// Shuffle a slice in place.
    fn shuffle<T>(&mut self, slice: &mut [T]);

    /// Reseed the RNG with a new seed.
    fn reseed(&self, seed: u64);
}

/// A generic random number generator wrapper that can work with different RNGs.
///
/// This provides a thread-safe, reusable interface for different RNG backends.
/// The RNG must implement `Rng + SeedableRng + Send`.
///
/// # Examples
///
/// ```rust
/// use rand::rngs::StdRng;
///
/// use phylo::random::{RandomGenerator, RandomSource};
///
/// // Create a custom RNG instance
/// let custom_rng: RandomGenerator<StdRng> = RandomGenerator::new(123);
/// let custom_value: f64 = custom_rng.gen();
/// ```
pub struct RandomGenerator<R>
where
    R: Rng + SeedableRng + Send,
{
    rng: Mutex<R>,
}

impl<R> RandomGenerator<R>
where
    R: Rng + SeedableRng + Send,
{
    /// Create a new RandomGenerator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(R::seed_from_u64(seed)),
        }
    }
}

/// Type alias for the default RNG implementation.
/// Currently uses StdRng for good performance and reproducibility.
pub type DefaultGenerator = RandomGenerator<StdRng>;

impl Default for DefaultGenerator {
    fn default() -> Self {
        let seed = Timestamp::now().as_u64(); // Use current timestamp as seed
        Self::new(seed)
    }
}

impl<R> RandomSource for RandomGenerator<R>
where
    R: Rng + SeedableRng + Send,
{
    /// Generate a random value of type T.
    fn gen<T>(&self) -> T
    where
        T: 'static,
        Standard: Distribution<T>,
    {
        let mut rng = self.rng.lock().unwrap();
        rng.gen::<T>()
    }

    /// Generate a random value in the specified range.
    fn gen_range<T, Range>(&self, range: Range) -> T
    where
        T: 'static + SampleUniform,
        Range: SampleRange<T>,
    {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_range(range)
    }

    /// Generate a random bool with probability p.
    fn gen_bool(&self, p: f64) -> bool {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_bool(p)
    }

    /// Generate a random uniform probability in the range [0.0, 1.0).
    fn gen_probability(&self) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_range(0.0..1.0)
    }

    /// Shuffle a slice in place.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let mut rng = self.rng.lock().unwrap();
        slice.shuffle(&mut *rng);
    }

    /// Reseed the RNG with a new seed.
    fn reseed(&self, seed: u64) {
        let mut rng = self.rng.lock().unwrap();
        *rng = R::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_rng_reproducibility() {
        // Test that creating new instances with the same seed produces the same sequence
        let rng1 = DefaultGenerator::new(42);
        let val1: f64 = rng1.gen::<f64>();
        let val2: u32 = rng1.gen::<u32>();

        let rng2 = DefaultGenerator::new(42);
        let val1_repeat: f64 = rng2.gen::<f64>();
        let val2_repeat: u32 = rng2.gen::<u32>();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
    }

    #[test]
    fn test_reseed() {
        // Test that reseeding works correctly
        let rng = DefaultGenerator::new(42);
        let val1: f64 = rng.gen::<f64>();

        rng.reseed(42);
        let val1_repeat: f64 = rng.gen::<f64>();

        assert_eq!(val1, val1_repeat);
    }

    #[test]
    fn test_global_rng_functions() {
        let rng = DefaultGenerator::new(123);

        // Test different random generation functions
        let _random_f64: f64 = rng.gen::<f64>();
        let _random_probability = rng.gen_probability();
        let _random_range = rng.gen_range(1..10);
        let _random_bool = rng.gen_bool(0.5);

        // Just ensure they don't panic and return reasonable values
        assert!((0.0..1.0).contains(&rng.gen_probability()));
        assert!((1..10).contains(&rng.gen_range(1..10)));
    }

    #[test]
    fn test_different_seeds_produce_different_values() {
        let rng = DefaultGenerator::new(1);
        // init_rng(1);
        let val1: f64 = rng.gen();

        rng.reseed(2);
        let val2: f64 = rng.gen();

        assert_ne!(val1, val2);
    }

    #[test]
    fn test_shuffle() {
        let mut rng = DefaultGenerator::new(42);
        let mut vec = vec![1, 2, 3, 4, 5];
        let original_vec = vec.clone();
        rng.shuffle(&mut vec);
        assert_ne!(vec, original_vec);
        // Check that all elements are still present
        assert!(vec.iter().all(|x| original_vec.contains(x)));
    }
}
