use std::sync::Mutex;
use std::sync::OnceLock;

use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Global random number generator for the crate.
///
/// This provides a thread-safe, globally accessible RNG that is seeded once
/// at the start of execution. This ensures reproducible randomness across
/// the entire crate when using the same seed.
pub struct GlobalRng {
    rng: Mutex<StdRng>,
}

impl GlobalRng {
    /// Create a new GlobalRng with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Generate a random value of type T.
    pub fn gen<T>(&self) -> T
    where
        Standard: Distribution<T>,
    {
        let mut rng = self.rng.lock().unwrap();
        rng.gen()
    }

    /// Generate a random value in the given range.
    pub fn gen_range<T, R>(&self, range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_range(range)
    }

    /// Generate a random f64 in the range [0, 1).
    pub fn gen_f64(&self) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        rng.gen()
    }

    /// Generate a random boolean.
    pub fn gen_bool(&self, p: f64) -> bool {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_bool(p)
    }

    /// Reseed the global RNG with a new seed.
    ///
    /// This is useful for testing or when you want to start a new
    /// reproducible sequence.
    pub fn reseed(&self, seed: u64) {
        let mut rng = self.rng.lock().unwrap();
        *rng = StdRng::seed_from_u64(seed);
    }
}

/// Global instance of the RNG.
static GLOBAL_RNG: OnceLock<GlobalRng> = OnceLock::new();

/// Initialize the global RNG with a seed.
///
/// This should be called once at the start of your program.
/// If not called explicitly, the RNG will be initialized with a default seed
/// when first accessed.
///
/// # Examples
/// ```
/// phylo::random::init_rng(42);
/// let random_value: f64 = phylo::random::random();
/// ```
pub fn init_rng(seed: u64) {
    let _ = GLOBAL_RNG.set(GlobalRng::new(seed));
}

/// Get a reference to the global RNG.
///
/// If the RNG hasn't been initialized with `init_rng()`, it will be
/// initialized with a default seed of 0.
pub fn global_rng() -> &'static GlobalRng {
    GLOBAL_RNG.get_or_init(|| GlobalRng::new(0))
}

/// Generate a random value of type T using the global RNG.
///
/// # Examples
/// ```
/// let random_u32: u32 = phylo::random::gen();
/// let random_f64: f64 = phylo::random::gen();
/// ```
pub fn gen<T>() -> T
where
    Standard: Distribution<T>,
{
    global_rng().gen()
}

/// Generate a random value in the given range using the global RNG.
///
/// # Examples
/// ```
/// let random_int = phylo::random::gen_range(1..10);
/// let random_float = phylo::random::gen_range(0.0..1.0);
/// ```
pub fn gen_range<T, R>(range: R) -> T
where
    T: rand::distributions::uniform::SampleUniform,
    R: rand::distributions::uniform::SampleRange<T>,
{
    global_rng().gen_range(range)
}

/// Generate a random f64 in the range [0, 1) using the global RNG.
///
/// # Examples
/// ```
/// let probability = phylo::random::gen_probability();
/// ```
pub fn gen_probability() -> f64 {
    global_rng().gen_f64()
}

/// Generate a random boolean with the given probability using the global RNG.
///
/// # Examples
/// ```
/// let coin_flip = phylo::random::gen_bool(0.5);
/// ```
pub fn gen_bool(p: f64) -> bool {
    global_rng().gen_bool(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_rng_reproducibility() {
        // Test that creating new instances with the same seed produces the same sequence
        let rng1 = GlobalRng::new(42);
        let val1: f64 = rng1.gen();
        let val2: u32 = rng1.gen();

        let rng2 = GlobalRng::new(42);
        let val1_repeat: f64 = rng2.gen();
        let val2_repeat: u32 = rng2.gen();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
    }

    #[test]
    fn test_reseed() {
        // Test that reseeding works correctly
        let rng = GlobalRng::new(42);
        let val1: f64 = rng.gen();

        rng.reseed(42);
        let val1_repeat: f64 = rng.gen();

        assert_eq!(val1, val1_repeat);
    }

    #[test]
    fn test_global_rng_functions() {
        init_rng(123);

        // Test different random generation functions
        let _random_f64: f64 = gen();
        let _random_probability = gen_probability();
        let _random_range = gen_range(1..10);
        let _random_bool = gen_bool(0.5);

        // Just ensure they don't panic and return reasonable values
        assert!((0.0..1.0).contains(&gen_probability()));
        assert!((1..10).contains(&gen_range(1..10)));
    }

    #[test]
    fn test_different_seeds_produce_different_values() {
        init_rng(1);
        let val1: f64 = gen();

        global_rng().reseed(2);
        let val2: f64 = gen();

        assert_ne!(val1, val2);
    }
}
