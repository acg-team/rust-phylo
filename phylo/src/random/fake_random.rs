use std::any::TypeId;
use std::sync::Mutex;

use rand::distributions::{
    uniform::{SampleRange, SampleUniform},
    Distribution, Standard,
};

use crate::random::RandomSource;

/// A fake random number generator for deterministic testing.
/// Returns predictable values from pre-configured sequences.
pub struct FakeGenerator {
    u64_values: Mutex<Vec<u64>>,
    f64_values: Mutex<Vec<f64>>,
    bool_values: Mutex<Vec<bool>>,
    u64_index: Mutex<usize>,
    f64_index: Mutex<usize>,
    bool_index: Mutex<usize>,
}

impl FakeGenerator {
    /// Create a new FakeGenerator with empty value sequences,
    /// which will default to 0, 0.0, or false if no values are provided
    pub fn new() -> Self {
        Self {
            u64_values: Mutex::new(Vec::new()),
            f64_values: Mutex::new(Vec::new()),
            bool_values: Mutex::new(Vec::new()),
            u64_index: Mutex::new(0),
            f64_index: Mutex::new(0),
            bool_index: Mutex::new(0),
        }
    }

    /// Create a FakeGenerator with pre-configured u64 values
    pub fn from_u64_values(values: Vec<u64>) -> Self {
        Self {
            u64_values: Mutex::new(values),
            f64_values: Mutex::new(Vec::new()),
            bool_values: Mutex::new(Vec::new()),
            u64_index: Mutex::new(0),
            f64_index: Mutex::new(0),
            bool_index: Mutex::new(0),
        }
    }

    /// Create a FakeGenerator with pre-configured f64 values
    pub fn from_f64_values(values: Vec<f64>) -> Self {
        Self {
            u64_values: Mutex::new(Vec::new()),
            f64_values: Mutex::new(values),
            bool_values: Mutex::new(Vec::new()),
            u64_index: Mutex::new(0),
            f64_index: Mutex::new(0),
            bool_index: Mutex::new(0),
        }
    }

    /// Add more u64 values to the sequence
    pub fn add_u64_values(&self, values: Vec<u64>) {
        let mut u64_values = self.u64_values.lock().unwrap();
        u64_values.extend(values);
    }

    /// Add more f64 values to the sequence
    pub fn add_f64_values(&self, values: Vec<f64>) {
        let mut f64_values = self.f64_values.lock().unwrap();
        f64_values.extend(values);
    }

    /// Add more bool values to the sequence
    pub fn add_bool_values(&self, values: Vec<bool>) {
        let mut bool_values = self.bool_values.lock().unwrap();
        bool_values.extend(values);
    }

    /// Get the next u64 value, default is 0
    fn next_u64(&self) -> u64 {
        let mut index = self.u64_index.lock().unwrap();
        let values = self.u64_values.lock().unwrap();
        if values.is_empty() {
            0
        } else {
            let value = values[*index % values.len()];
            *index += 1;
            value
        }
    }

    /// Get the next f64 value, default is 0.0
    fn next_f64(&self) -> f64 {
        let mut index = self.f64_index.lock().unwrap();
        let values = self.f64_values.lock().unwrap();
        if values.is_empty() {
            0.0
        } else {
            let value = values[*index % values.len()];
            *index += 1;
            value
        }
    }

    /// Get the next bool value, default is false
    fn next_bool(&self) -> bool {
        let mut index = self.bool_index.lock().unwrap();
        let values = self.bool_values.lock().unwrap();
        if values.is_empty() {
            false
        } else {
            let value = values[*index % values.len()];
            *index += 1;
            value
        }
    }
}

impl Default for FakeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for FakeGenerator {
    fn gen<T>(&self) -> T
    where
        T: 'static,
        Standard: Distribution<T>,
    {
        // This is a bit tricky since we need to handle any type T.
        // We'll use type erasure and handle the most common types.

        let type_id = TypeId::of::<T>();
        if type_id == TypeId::of::<u64>() {
            let value = self.next_u64();
            unsafe { std::mem::transmute_copy(&value) }
        } else if type_id == TypeId::of::<f64>() {
            let value = self.next_f64();
            unsafe { std::mem::transmute_copy(&value) }
        } else if type_id == TypeId::of::<bool>() {
            let value = self.next_bool();
            unsafe { std::mem::transmute_copy(&value) }
        } else {
            // For other types, fall back to using u64 as the source
            let value = self.next_u64();
            unsafe { std::mem::transmute_copy(&value) }
        }
    }

    fn gen_range<T, Range>(&self, range: Range) -> T
    where
        T: 'static + SampleUniform,
        Range: SampleRange<T>,
    {
        // Handle the most common range types
        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<u64>() {
            let range = unsafe { std::mem::transmute_copy::<Range, std::ops::Range<u64>>(&range) };
            let value = self.next_u64();
            let result = if range.is_empty() {
                range.start
            } else {
                range.start + (value % (range.end - range.start))
            };
            unsafe { std::mem::transmute_copy(&result) }
        } else {
            // For other types, use the first value from our sequences as a fallback
            let value = self.next_u64();
            unsafe { std::mem::transmute_copy(&value) }
        }
    }

    fn gen_bool(&self, _p: f64) -> bool {
        self.next_bool()
    }

    fn gen_probability(&self) -> f64 {
        self.next_f64()
    }

    fn shuffle<T>(&mut self, _slice: &mut [T]) {
        // No shuffling for fake generator
    }

    fn reseed(&self, _seed: u64) {
        // Reseeding just resets the indices
        *self.u64_index.lock().unwrap() = 0;
        *self.f64_index.lock().unwrap() = 0;
        *self.bool_index.lock().unwrap() = 0;
    }
}
