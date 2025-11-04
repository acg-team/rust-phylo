use rand::{RngCore, SeedableRng};

/// A fake random number generator for deterministic testing.
/// Can return pre-configured values for unsigned (usize, u64, u32, u16, u8) and
/// signed (isize, i64, i32, i16, i8) integer types, and defaults to 0, 0.0, or false for other types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FakeRng {
    u64_values: Vec<u64>,
    u64_index: usize,
}

impl RngCore for FakeRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64_value() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64_value()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // Fill the byte array with values from the saved u64 sequence
        for chunk in dest.chunks_mut(8) {
            let value = self.next_u64_value();
            let bytes = value.to_le_bytes();
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte = bytes[i];
            }
        }
    }
}

// Implement SeedableRng but the seed is ignored since this is a fake RNG
impl SeedableRng for FakeRng {
    type Seed = [u8; 8];

    fn from_seed(_: Self::Seed) -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }

    fn seed_from_u64(_: u64) -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }
}

unsafe impl Send for FakeRng {}

impl FakeRng {
    /// Create a new FakeRng with an empty u64 value sequence, will return 0 for every int value
    pub fn new() -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }

    /// Create a FakeRng with pre-configured u64 values
    pub fn from_u64_values(values: Vec<u64>) -> Self {
        Self {
            u64_values: values,
            u64_index: 0,
        }
    }

    /// Get the next u64 value, default is 0
    fn next_u64_value(&mut self) -> u64 {
        if self.u64_values.is_empty() {
            0
        } else {
            let value = self.u64_values[self.u64_index % self.u64_values.len()];
            self.u64_index += 1;
            value
        }
    }
}

impl Default for FakeRng {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use rand::{distr::weighted::WeightedIndex, SeedableRng};

    use crate::random::{FakeGenerator, RandomGenerator};

    use super::FakeRng;

    #[test]
    fn fake_rng_defaults() {
        // Test FakeRng defaults
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::default());
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.random();
        assert_eq!(val, 0);
        let val: f64 = fake_rng.random();
        assert_eq!(val, 0.0);
        let val: bool = fake_rng.random();
        assert!(!val);
    }

    #[test]
    fn fake_rng_from_seed() {
        // Test new FakeGenerator from seed (seed makes no difference)
        let fake1 = RandomGenerator::from_rng(FakeRng::seed_from_u64(42));
        let fake2 = RandomGenerator::from_rng(FakeRng::from_seed([42; 8]));
        let fake3 = RandomGenerator::from_rng(FakeRng::new());
        let fake4 = FakeGenerator::default();
        assert_eq!(fake1.seed(), 0);
        assert_eq!(fake2.seed(), 0);
        assert_eq!(fake3.seed(), 0);
        assert_eq!(fake4.seed(), 0);
        assert_eq!(fake1, fake2);
        assert_eq!(fake1, fake3);
        assert_eq!(fake1, fake4);
    }

    #[test]
    fn fake_generator_defaults() {
        // Test FakeGenerator defaults
        let mut fake_rng = FakeGenerator::default();
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.random();
        assert_eq!(val, 0);
        let val: f64 = fake_rng.random();
        assert_eq!(val, 0.0);
        let val: bool = fake_rng.random();
        assert!(!val);
    }

    #[test]
    fn fake_rng_with_u64_values() {
        // Test FakeGenerator with pre-configured values
        let values = (15..25).collect::<Vec<u64>>();
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        assert_eq!(fake_rng.seed(), 0);
        for i in 0..10 {
            let val: u64 = fake_rng.random();
            assert_eq!(val, values[i % values.len()]);
        }
        let val: f64 = fake_rng.random();
        assert_eq!(val, 0.0); // Default for f64
    }

    #[test]
    fn fake_rng_with_diff_types() {
        // Test FakeGenerator with different value types
        let values = vec![5, 6, 7, 8, 9, 14, 15, 16];
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.random();
        assert_eq!(val, values[0]);
        let val: u32 = fake_rng.random();
        assert_eq!(val, values[1] as u32);
        let val: u16 = fake_rng.random();
        assert_eq!(val, values[2] as u16);
        let val: u8 = fake_rng.random();
        assert_eq!(val, values[3] as u8);
        let val: i64 = fake_rng.random();
        assert_eq!(val, values[4] as i64);
        let val: i32 = fake_rng.random();
        assert_eq!(val, values[5] as i32);
        let val: i16 = fake_rng.random();
        assert_eq!(val, values[6] as i16);
        let val: i8 = fake_rng.random();
        assert_eq!(val, values[7] as i8);
        let val: f64 = fake_rng.random();
        assert_eq!(val, 0.0); // Default for f64
        let val: f32 = fake_rng.random();
        assert_eq!(val, 0.0); // Default for f32
        let val: bool = fake_rng.random();
        assert!(!val); // Default for bool
    }

    #[test]
    fn fake_rng_reproducibility() {
        // Test that creating new instances with the same values produces the same sequence
        let values = vec![5, 7, 8, 10, 12];
        let mut rng1 = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        let val1: u64 = rng1.random();
        let val2: u32 = rng1.random();

        let mut rng2 = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        let val1_repeat: u64 = rng2.random();
        let val2_repeat: u32 = rng2.random();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
        assert_eq!(val1, values[0]);
        assert_eq!(val2, values[1] as u32);
    }

    #[test]
    fn fake_reseed_empty() {
        // Test that reseeding does not do anything to an empty FakeGenerator
        let mut fake_rng = FakeGenerator::default();
        assert_eq!(fake_rng.seed(), 0);
        let val1: u64 = fake_rng.random();

        fake_rng.reseed(42);
        assert_eq!(fake_rng.seed(), 42);
        let val1_repeat: u64 = fake_rng.random();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val1, 0); // Default value after reseed
    }

    #[test]
    fn fake_shuffle() {
        let mut rng = FakeGenerator::default();
        let mut vec = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut vec);
        assert_eq!(vec, vec![5, 1, 2, 3, 4]);
    }

    #[test]
    fn fake_sample_default() {
        let mut rng = FakeGenerator::default();
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(rng.sample(&dist), 0);
    }

    #[test]
    fn fake_sample() {
        let mut rng = FakeGenerator::default();
        let dist = WeightedIndex::new([0.0, 3.0, 2.0, 1.0, 2.0]).unwrap();
        assert_eq!(rng.sample(&dist), 1);
        assert_eq!(rng.sample(&dist), 1);
        assert_eq!(rng.sample(&dist), 1);
    }
}
