use std::sync::Mutex;

use ntimestamp::Timestamp;
use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::distributions::{Distribution, Standard};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub mod fake_random;
pub use fake_random::*;

/// Trait for random number generation
pub trait RandomSource {
    /// Get the current seed of the RNG.
    fn seed(&self) -> u64;

    /// Generate a random value of type T.
    fn gen<T>(&self) -> T
    where
        T: 'static,
        Standard: Distribution<T>;

    /// Generate a random bool with probability p.
    fn gen_bool(&self, p: f64) -> bool;

    /// Generate a random uniform probability in the range [0.0, 1.0).
    fn gen_probability(&self) -> f64;

    /// Shuffle a slice in place.
    fn shuffle<T>(&self, slice: &mut [T]);

    /// Reseed the RNG with a new seed.
    fn reseed(&self, seed: u64);

    /// Sample from a weighted distribution.
    fn sample<D, T>(&self, dist: &D) -> T
    where
        T: 'static,
        D: Distribution<T>;
}

pub struct SeededRng<R>
where
    R: Rng + SeedableRng + Send,
{
    pub seed: u64,
    pub rng: R,
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
    r: Mutex<SeededRng<R>>,
}

impl<R> RandomGenerator<R>
where
    R: Rng + SeedableRng + Send,
{
    /// Create a new RandomGenerator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            r: Mutex::new(SeededRng {
                seed,
                rng: R::seed_from_u64(seed),
            }),
        }
    }
}

/// Type alias for the default RNG implementation.
/// Currently uses StdRng which is not platform-independent, and should be replaced with
/// a platform-independent RNG.
/// TODO: Replace with a platform-independent RNG implementation (at the moment `rand_pcg` and `rand_chacha`
/// cause dependency clashes).
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
    /// Get the current seed of the RNG.
    fn seed(&self) -> u64 {
        let rng = self.r.lock().unwrap();
        rng.seed
    }

    /// Generate a random value of type T.
    fn gen<T>(&self) -> T
    where
        T: 'static,
        Standard: Distribution<T>,
    {
        let mut r = self.r.lock().unwrap();
        r.rng.gen::<T>()
    }

    /// Generate a random bool with probability p.
    fn gen_bool(&self, p: f64) -> bool {
        let mut r = self.r.lock().unwrap();
        r.rng.gen_bool(p)
    }

    /// Generate a random uniform probability in the range [0.0, 1.0).
    fn gen_probability(&self) -> f64 {
        let mut r = self.r.lock().unwrap();
        r.rng.gen_range(0.0..1.0)
    }

    /// Shuffle a slice in place.
    fn shuffle<T>(&self, slice: &mut [T]) {
        let mut r = self.r.lock().unwrap();
        slice.shuffle(&mut r.rng);
    }

    /// Reseed the RNG with a new seed.
    fn reseed(&self, seed: u64) {
        let mut r = self.r.lock().unwrap();
        *r = SeededRng {
            seed,
            rng: R::seed_from_u64(seed),
        };
    }

    /// Sample from a weighted distribution.
    fn sample<D, T>(&self, dist: &D) -> T
    where
        T: 'static,
        D: Distribution<T>,
    {
        let mut r = self.r.lock().unwrap();
        r.rng.sample(dist)
    }
}

impl<R> RandomGenerator<R>
where
    R: Rng + SeedableRng + Send,
{
    /// Generate a random value in the specified range.
    pub fn gen_range<T, Range>(&self, range: Range) -> T
    where
        T: 'static + SampleUniform,
        Range: SampleRange<T>,
    {
        let mut r = self.r.lock().unwrap();
        r.rng.gen_range(range)
    }
}

#[cfg_attr(coverage, coverage(off))]
#[cfg(test)]
mod tests {
    use itertools::repeat_n;
    use rand::distributions::WeightedIndex;

    use super::*;

    #[test]
    fn rng_reproducibility() {
        // Test that creating new instances with the same seed produces the same sequence
        let rng1 = DefaultGenerator::new(42);
        assert_eq!(rng1.seed(), 42);
        let val1: f64 = rng1.gen();
        let val2: u32 = rng1.gen();

        let rng2 = DefaultGenerator::new(42);
        assert_eq!(rng2.seed(), 42);
        let val1_repeat: f64 = rng2.gen();
        let val2_repeat: u32 = rng2.gen();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
    }

    #[test]
    fn rng_different_types() {
        // Test that most common types can be generated from the same RNG instance and the
        // generator does not panic.
        let rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let _val: usize = rng.gen();
        let _val: u64 = rng.gen();
        let _val: u32 = rng.gen();
        let _val: u16 = rng.gen();
        let _val: u8 = rng.gen();
        let _val: isize = rng.gen();
        let _val: i128 = rng.gen();
        let _val: i64 = rng.gen();
        let _val: i32 = rng.gen();
        let _val: i16 = rng.gen();
        let _val: i8 = rng.gen();
        let _val: char = rng.gen();
        let _val: bool = rng.gen();
        let val: f64 = rng.gen();
        assert!(val.is_finite() && !val.is_nan());
        let val: f32 = rng.gen();
        assert!(val.is_finite() && !val.is_nan());
    }

    #[test]
    fn reseed() {
        // Test that reseeding to the same value produces the same sample
        let rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let val1: f64 = rng.gen();

        rng.reseed(42);
        assert_eq!(rng.seed(), 42);
        let val1_repeat: f64 = rng.gen();

        assert_eq!(val1, val1_repeat);
    }

    #[test]
    fn rng_functions() {
        let rng = DefaultGenerator::new(123);
        assert_eq!(rng.seed(), 123);
        let _random_f64: f64 = rng.gen::<f64>();
        let _random_probability = rng.gen_probability();
        let _random_bool = rng.gen_bool(0.5);
        assert!((0.0..1.0).contains(&rng.gen_probability()));
    }

    #[test]
    fn rng_range() {
        let rng = DefaultGenerator::new(123);
        assert_eq!(rng.seed(), 123);
        for _ in 0..10 {
            let val: u32 = rng.gen_range(1..100);
            assert!((1..100).contains(&val));
        }
    }

    #[test]
    fn different_seeds_produce_different_values() {
        // This is not guaranteed to produce different values because there are no guarantees on the Rng
        // implementation, but it is extremely unlikely that it will fail.
        let rng = DefaultGenerator::new(1);
        assert_eq!(rng.seed(), 1);
        let val1: f64 = rng.gen();
        rng.reseed(2);
        assert_eq!(rng.seed(), 2);
        let val1_repeat: f64 = rng.gen();
        assert_ne!(val1, val1_repeat);
    }

    #[test]
    fn different_timestamps_produce_different_values() {
        // DefaultGenerator uses the current timestamp as a seed, so different instances should have different seeds
        // per definition of ntimestamp, but not guaranteed to produce different values, as in the previous test.
        let timestamp = Timestamp::now().as_u64();
        let rng = DefaultGenerator::default();
        assert!(rng.seed() > timestamp);
        let val1: usize = rng.gen();
        let val2: f64 = rng.gen();
        let rng2 = DefaultGenerator::default();
        assert!(rng2.seed() > rng.seed());
        assert!(rng2.seed() > timestamp);
        let val1_repeat: usize = rng2.gen();
        let val2_repeat: f64 = rng2.gen();
        assert_ne!(val1, val1_repeat);
        assert_ne!(val2, val2_repeat);
    }

    #[test]
    fn shuffle() {
        let rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let mut vec = vec![1, 2, 3, 4, 5];
        let original_vec = vec.clone();
        rng.shuffle(&mut vec);
        assert_ne!(vec, original_vec);
        // Check that all elements are still present
        assert!(vec.iter().all(|x| original_vec.contains(x)));
    }

    #[test]
    fn sample_weighted_index() {
        let rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample < 3);
        let dist = WeightedIndex::new(repeat_n(1.0, 15)).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample < 15);
    }
}
