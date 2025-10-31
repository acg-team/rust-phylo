use std::any::{Any, TypeId};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::random::RandomSource;

/// A fake random number generator for deterministic testing.
/// Returns predictable values from pre-configured sequences.
pub struct FakeGenerator {
    u64_values: Vec<u64>,
    f64_values: Vec<f64>,
    bool_values: Vec<bool>,
    u64_index: usize,
    f64_index: usize,
    bool_index: usize,
    seed: u64,
}

macro_rules! fakegen_downcast {
    ($val:expr) => {{
        let boxed: Box<dyn Any> = Box::new($val);
        *boxed.downcast::<T>().unwrap()
    }};
}

impl FakeGenerator {
    /// Create a new FakeGenerator with empty value sequences,
    /// which will default to 0, 0.0, or false if no values are provided
    pub fn new() -> Self {
        Self {
            u64_values: Vec::new(),
            f64_values: Vec::new(),
            bool_values: Vec::new(),
            u64_index: 0,
            f64_index: 0,
            bool_index: 0,
            seed: 0,
        }
    }

    /// Create a FakeGenerator with pre-configured u64 values
    pub fn from_u64_values(values: Vec<u64>) -> Self {
        Self {
            u64_values: values,
            f64_values: Vec::new(),
            bool_values: Vec::new(),
            u64_index: 0,
            f64_index: 0,
            bool_index: 0,
            seed: 0,
        }
    }

    /// Create a FakeGenerator with pre-configured f64 values
    pub fn from_f64_values(values: Vec<f64>) -> Self {
        Self {
            u64_values: Vec::new(),
            f64_values: values,
            bool_values: Vec::new(),
            u64_index: 0,
            f64_index: 0,
            bool_index: 0,
            seed: 0,
        }
    }

    /// Add more u64 values to the sequence
    pub fn add_u64_values(&mut self, values: Vec<u64>) {
        self.u64_values.extend(values);
    }

    /// Add more f64 values to the sequence
    pub fn add_f64_values(&mut self, values: Vec<f64>) {
        self.f64_values.extend(values);
    }

    /// Add more bool values to the sequence
    pub fn add_bool_values(&mut self, values: Vec<bool>) {
        self.bool_values.extend(values);
    }

    /// Get the next u64 value, default is 0
    fn next_u64(&mut self) -> u64 {
        if self.u64_values.is_empty() {
            0
        } else {
            let value = self.u64_values[self.u64_index % self.u64_values.len()];
            self.u64_index += 1;
            value
        }
    }

    /// Get the next f64 value, default is 0.0
    fn next_f64(&mut self) -> f64 {
        if self.f64_values.is_empty() {
            0.0
        } else {
            let value = self.f64_values[self.f64_index % self.f64_values.len()];
            self.f64_index += 1;
            value
        }
    }

    /// Get the next bool value, default is false
    fn next_bool(&mut self) -> bool {
        if self.bool_values.is_empty() {
            false
        } else {
            let value = self.bool_values[self.bool_index % self.bool_values.len()];
            self.bool_index += 1;
            value
        }
    }

    fn next_value<T>(&mut self) -> T
    where
        T: 'static,
    {
        // Safe implementation using downcasting
        let type_id = TypeId::of::<T>();
        // Create the appropriate value based on the type
        if type_id == TypeId::of::<u64>() {
            fakegen_downcast!(self.next_u64())
        } else if type_id == TypeId::of::<usize>() {
            fakegen_downcast!(self.next_u64() as usize)
        } else if type_id == TypeId::of::<u32>() {
            fakegen_downcast!(self.next_u64() as u32)
        } else if type_id == TypeId::of::<u16>() {
            fakegen_downcast!(self.next_u64() as u16)
        } else if type_id == TypeId::of::<u8>() {
            fakegen_downcast!(self.next_u64() as u8)
        } else if type_id == TypeId::of::<char>() {
            fakegen_downcast!(self.next_u64() as u8 as char)
        } else if type_id == TypeId::of::<i128>() {
            fakegen_downcast!(self.next_u64() as i128)
        } else if type_id == TypeId::of::<i64>() {
            fakegen_downcast!(self.next_u64() as i64)
        } else if type_id == TypeId::of::<i32>() {
            fakegen_downcast!(self.next_u64() as i32)
        } else if type_id == TypeId::of::<i16>() {
            fakegen_downcast!(self.next_u64() as i16)
        } else if type_id == TypeId::of::<i8>() {
            fakegen_downcast!(self.next_u64() as i8)
        } else if type_id == TypeId::of::<f64>() {
            fakegen_downcast!(self.next_f64())
        } else if type_id == TypeId::of::<f32>() {
            fakegen_downcast!(self.next_f64() as f32)
        } else if type_id == TypeId::of::<bool>() {
            fakegen_downcast!(self.next_bool())
        } else {
            // For unknown types, immediately panic with an error
            panic!(
                "FakeGenerator doesn't support type {:?}",
                std::any::type_name::<T>()
            )
        }
    }
}

impl Default for FakeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for FakeGenerator {
    fn seed(&self) -> u64 {
        // The seed is irrelevant for FakeGenerator, but we implement it for compatibility
        self.seed
    }

    fn gen<T>(&mut self) -> T
    where
        T: 'static,
        Standard: Distribution<T>,
    {
        self.next_value()
    }

    fn gen_bool(&mut self, _p: f64) -> bool {
        self.next_bool()
    }

    fn gen_probability(&mut self) -> f64 {
        let val = self.next_f64();
        val.clamp(0.0, 1.0)
    }

    fn shuffle<T>(&mut self, _slice: &mut [T]) {
        // No shuffling for fake generator
    }

    fn reseed(&mut self, seed: u64) {
        // Reseeding resets the indices
        self.u64_index = 0;
        self.f64_index = 0;
        self.bool_index = 0;
        self.seed = seed;
    }

    fn sample<D, T>(&mut self, _dist: &D) -> T
    where
        T: 'static,
        D: Distribution<T>,
    {
        // Will return indices provided as input, cannot check if index is within range
        self.next_value()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use rand::distributions::WeightedIndex;

    use crate::random::RandomSource;

    use super::*;

    #[test]
    fn fake_rng_defaults() {
        // Test new FakeGenerator defaults
        let mut fake_rng = FakeGenerator::new();
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.gen();
        assert_eq!(val, 0);
        let val: f64 = fake_rng.gen();
        assert_eq!(val, 0.0);
        assert!(!fake_rng.gen::<bool>());
    }

    #[test]
    fn fake_rng_with_values() {
        // Test FakeGenerator with pre-configured values
        let values = (15..25).collect::<Vec<u64>>();
        let mut fake_rng = FakeGenerator::from_u64_values(values.clone());
        assert_eq!(fake_rng.seed(), 0);
        for i in 0..10 {
            let val: u64 = fake_rng.gen();
            assert_eq!(val, values[i % values.len()]);
        }
        let val: f64 = fake_rng.gen();
        assert_eq!(val, 0.0); // Default for f64
        assert!(!fake_rng.gen::<bool>()); // Default for bool
    }

    #[test]
    #[should_panic(expected = "FakeGenerator doesn't support type")]
    fn fake_rng_panic() {
        // Test that FakeGenerator panics for an unsupported type (isize)
        let mut fake_rng = FakeGenerator::new();
        fake_rng.gen::<isize>();
    }

    #[test]
    fn fake_rng_gen_bool() {
        // Test that p makes no difference for gen_bool in FakeGenerator
        let values = vec![true, false, true, false];
        let mut fake_rng = FakeGenerator::new();
        fake_rng.add_bool_values(values.clone());
        for i in 0..10 {
            assert_eq!(fake_rng.gen_bool(0.3), values[i % values.len()]);
        }
        fake_rng.reseed(0);
        for i in 0..10 {
            assert_eq!(fake_rng.gen_bool(0.7), values[i % values.len()]);
        }
    }

    #[test]
    fn fake_rng_with_diff_types() {
        // Test FakeGenerator with different value types
        let values = vec![5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 33];
        let mut fake_rng = FakeGenerator::from_u64_values(values.clone());
        assert_eq!(fake_rng.seed(), 0);
        let val: usize = fake_rng.gen();
        assert_eq!(val, values[0] as usize);
        let val: u64 = fake_rng.gen();
        assert_eq!(val, values[1]);
        let val: u32 = fake_rng.gen();
        assert_eq!(val, values[2] as u32);
        let val: u16 = fake_rng.gen();
        assert_eq!(val, values[3] as u16);
        let val: u8 = fake_rng.gen();
        assert_eq!(val, values[4] as u8);
        let val: i128 = fake_rng.gen();
        assert_eq!(val, values[5] as i128);
        let val: i64 = fake_rng.gen();
        assert_eq!(val, values[6] as i64);
        let val: i32 = fake_rng.gen();
        assert_eq!(val, values[7] as i32);
        let val: i16 = fake_rng.gen();
        assert_eq!(val, values[8] as i16);
        let val: i8 = fake_rng.gen();
        assert_eq!(val, values[9] as i8);
        let val: char = fake_rng.gen();
        assert_eq!(val, values[10] as u8 as char);
        let val: f64 = fake_rng.gen();
        assert_eq!(val, 0.0); // Default for f64
        let val: bool = fake_rng.gen();
        assert!(!val); // Default for bool
    }

    #[test]
    fn fake_rng_reproducibility() {
        // Test that creating new instances with the same values produces the same sequence
        let values = vec![0.1, 0.2, 0.3];
        let mut rng1 = FakeGenerator::from_f64_values(values.clone());
        let val1: f64 = rng1.gen();
        let val2: f32 = rng1.gen();

        let mut rng2 = FakeGenerator::from_f64_values(values);
        let val1_repeat: f64 = rng2.gen();
        let val2_repeat: f32 = rng2.gen();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
    }

    #[test]
    fn fake_reseed_empty() {
        // Test that reseeding does not do anything to an empty FakeGenerator
        let mut fake_rng = FakeGenerator::new();
        assert_eq!(fake_rng.seed(), 0);
        let val1: f64 = fake_rng.gen();

        fake_rng.reseed(42);
        assert_eq!(fake_rng.seed(), 42);
        let val1_repeat: f64 = fake_rng.gen();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val1, 0.0); // Default value after reseed
    }

    #[test]
    fn fake_reseed_with_values() {
        // Test that reseeding resets pre-configured value counters
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let mut fake_rng = FakeGenerator::from_f64_values(values.clone());
        let val1: f64 = fake_rng.gen();
        for i in 1..10 {
            let val: f64 = fake_rng.gen();
            assert_eq!(val, values[i % values.len()]);
        }

        fake_rng.reseed(42);
        let val1_repeat: f64 = fake_rng.gen();
        assert_eq!(val1, val1_repeat);
        assert_eq!(val1, values[0]); // Should return to the first value after reseed
    }

    #[test]
    fn fake_add_values_f64() {
        let mut fake_rng = FakeGenerator::new();
        let values = vec![0.1, 0.2, 0.3, 0.4];
        fake_rng.add_f64_values(values.clone());
        for value in values.iter() {
            let val: f64 = fake_rng.gen();
            assert_eq!(val, *value);
        }
        let val: f64 = fake_rng.gen();
        assert_eq!(val, values[0]); // Wraps around
        let val: u64 = fake_rng.gen();
        assert_eq!(val, 0); // Default for u64
        assert!(!fake_rng.gen::<bool>()); // Default for bool
    }

    #[test]
    fn fake_add_more_values_f64() {
        let mut fake_rng = FakeGenerator::new();
        let values = vec![0.1, 0.2, 0.3, 0.4];
        fake_rng.add_f64_values(values.clone());
        for value in values.iter() {
            let val: f64 = fake_rng.gen();
            assert_eq!(val, *value);
        }
        let more_values = vec![0.5, 0.6, 0.7, 0.8];
        fake_rng.add_f64_values(more_values.clone());
        for value in more_values.iter() {
            let val: f64 = fake_rng.gen();
            assert_eq!(val, *value);
        }
        let val: f64 = fake_rng.gen();
        assert_eq!(val, values[0]); // Wraps around
        let val: u64 = fake_rng.gen();
        assert_eq!(val, 0); // Default for u64
        assert!(!fake_rng.gen::<bool>()); // Default for bool
    }

    #[test]
    fn fake_add_values_u64() {
        let mut fake_rng = FakeGenerator::new();
        let values = (1..10).collect::<Vec<u64>>();
        fake_rng.add_u64_values(values.clone());
        for value in values.iter() {
            let val: u64 = fake_rng.gen();
            assert_eq!(val, *value);
        }
        let val: u64 = fake_rng.gen();
        assert_eq!(val, values[0]); // Wraps around
        let val: f64 = fake_rng.gen();
        assert_eq!(val, 0.0); // Default for f64
        assert!(!fake_rng.gen::<bool>()); // Default for bool
    }

    #[test]
    fn fake_add_values_bool() {
        let mut fake_rng = FakeGenerator::new();
        let source = vec![true, false, true, false];
        fake_rng.add_bool_values(source.clone());
        for value in source.iter() {
            let val: bool = fake_rng.gen();
            assert_eq!(val, *value);
        }
        let val: bool = fake_rng.gen();
        assert_eq!(val, source[0]); // Wraps around
        let val: u64 = fake_rng.gen();
        assert_eq!(val, 0); // Default for u64
        let val: f64 = fake_rng.gen();
        assert_eq!(val, 0.0); // Default for f64
    }

    #[test]
    fn fake_rng_probabilities() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 4.0, 5.0];
        let mut fake_rng = FakeGenerator::from_f64_values(values.clone());
        for _value in values.iter() {
            assert!((0.0..=1.0).contains(&fake_rng.gen_probability()));
        }
    }

    #[test]
    fn fake_different_values() {
        let mut fake_rng = FakeGenerator::from_f64_values(vec![0.1, 0.2, 0.3]);
        let val1: f64 = fake_rng.gen();
        let mut fake_rng2 = FakeGenerator::from_f64_values(vec![0.4, 0.5, 0.6]);
        let val2: f64 = fake_rng2.gen();
        assert_ne!(val1, val2);
    }

    #[test]
    fn fake_shuffle() {
        let mut rng = FakeGenerator::new();
        let mut vec = vec![1, 2, 3, 4, 5];
        let original_vec = vec.clone();
        rng.shuffle(&mut vec);
        assert_eq!(vec, original_vec);
    }

    #[test]
    fn fake_sample_default() {
        let mut rng = FakeGenerator::new();
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(rng.sample(&dist), 0);
    }

    #[test]
    fn fake_sample() {
        let mut rng = FakeGenerator::from_u64_values(vec![2, 0, 1]);
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(rng.sample(&dist), 2);
        assert_eq!(rng.sample(&dist), 0);
        assert_eq!(rng.sample(&dist), 1);
    }
}
