use crate::Int;
use crate::{assert, assert_eq};

pub struct UniformRandom {
    bytes: Vec<u8>,
}

impl UniformRandom {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Uniform random in [0, b)
    pub fn rand_below<R: rand::Rng>(&mut self, b: &Int, rng: &mut R, result: &mut Int) {
        let l = b.significant_bits();
        let clear_mask = (1 << (l % 8)) - 1;
        self.bytes.resize(l as usize / 8 + 1, 0);

        rng.fill_bytes(&mut self.bytes);
        *self.bytes.last_mut().unwrap() &= clear_mask;

        result.assign_digits(&self.bytes, rug::integer::Order::LsfLe);
        assert!(result.significant_bits() <= l);
        while &*result >= b {
            rng.fill_bytes(&mut self.bytes);
            *self.bytes.last_mut().unwrap() &= clear_mask;
            result.assign_digits(&self.bytes, rug::integer::Order::LsfLe);
        }
    }

    /// Uniform random in [-b, b]
    pub fn rand_signed<R: rand::Rng>(&mut self, b: &Int, rng: &mut R, result: &mut Int) {
        let l = b.significant_bits();
        let clear_mask = (1 << (l % 8)) - 1;
        self.bytes.resize(l as usize / 8 + 1, 0);

        rng.fill_bytes(&mut self.bytes);
        *self.bytes.last_mut().unwrap() &= clear_mask;

        result.assign_digits(&self.bytes, rug::integer::Order::LsfLe);
        assert!(result.significant_bits() <= l);
        while &*result > b {
            rng.fill_bytes(&mut self.bytes);
            *self.bytes.last_mut().unwrap() &= clear_mask;
            result.assign_digits(&self.bytes, rug::integer::Order::LsfLe);
        }

        let sign = rng.gen_bool(0.5);
        if sign {
            *result *= -1;
        }
    }

    /// Random b-bits integer
    pub fn rand_size<R: rand::Rng>(&mut self, mut bit_count: u32, rng: &mut R, result: &mut Int) {
        bit_count -= 1;

        self.bytes.resize(bit_count as usize / 8 + 1, 0);
        let clear_mask = (1 << (bit_count % 8)) - 1;
        rng.fill_bytes(&mut self.bytes);
        *self.bytes.last_mut().unwrap() &= clear_mask;
        result.assign_digits(&self.bytes, rug::integer::Order::LsfLe);
        result.set_bit(bit_count, true);
        assert_eq!(result.significant_bits(), bit_count + 1);
    }
}
