//! Most significant coefficients first

use crate::Int;

#[derive(Clone)]
pub struct PolyZp {
    pub p: Int,
    /// Values are in range [0, p).
    pub f: Vec<Int>,
}

impl PolyZp {
    pub fn new(mut f: PolyZ, modulo: &Int) -> PolyZp {
        assert_eq!(modulo.get_bit(0), true);
        assert_ne!(modulo.is_probably_prime(16), rug::integer::IsPrime::No);
        for c in &mut f.f {
            *c %= modulo;
            if &*c < &0 {
                *c += modulo;
            }
        }
        assert!(f.f.iter().all(|c| c >= &0 && c <= modulo));
        PolyZp {
            p: modulo.clone(),
            f: f.f,
        }
    }

    pub fn evaluate(&self, at: &Int) -> Int {
        let mut result = Int::from(0);
        for f in self.f.iter() {
            result *= at;
            result += f;
            result %= &self.p;
        }
        result
    }

    pub fn degree(&self) -> usize {
        assert!(!self.f.is_empty());
        self.f.len() - 1
    }
}

#[derive(Clone, Debug)]
pub struct PolyZ {
    pub f: Vec<Int>,
}

/// q^d / l
pub fn q_d_div_l(q: &Int, d: u32, l: &Int) -> PolyZ {
    assert!(q > &0);
    assert!(l < q);
    let mut digits = Vec::new();
    digits.reserve(d as usize);

    let mut x = Int::from(1);
    for _ in 0..d {
        x *= q;
        let (quotient, remainder) = x.div_rem(l.clone());
        x = remainder;
        digits.push(quotient);
    }

    PolyZ { f: digits }
}

impl PolyZ {
    /// To Z((p-1)/2)[X]
    pub fn new_bounded(poly: PolyZp) -> Self {
        assert_eq!(poly.p.get_bit(0), true);
        assert_ne!(poly.p.is_probably_prime(16), rug::integer::IsPrime::No);
        let mut f = poly.f;
        let bound = Int::from(&poly.p >> 1);
        for a in f.iter_mut() {
            if &*a > &bound {
                *a -= &poly.p;
            }
        }
        assert!(f.iter().all(|c| c <= &bound));
        PolyZ { f }
    }

    pub fn new_positive(f: PolyZp) -> Self {
        assert!(f.f.iter().all(|c| c >= &0));
        PolyZ { f: f.f }
    }

    pub fn add(mut self, mut other: PolyZ) -> PolyZ {
        if self.degree() < other.degree() {
            std::mem::swap(&mut self, &mut other);
        }
        let diff = self.degree() - other.degree();
        for (x, y) in self.f.iter_mut().skip(diff).zip(other.f.iter()) {
            *x += y;
        }
        self
    }

    pub fn substract(mut self, mut other: PolyZ) -> PolyZ {
        assert!(!self.f.is_empty());
        assert!(!other.f.is_empty());
        use rug::ops::NegAssign;
        let mut swapped = false;
        if self.degree() < other.degree() {
            std::mem::swap(&mut self, &mut other);
            swapped = true;
        }
        assert!(self.degree() >= other.degree());
        let diff = self.degree() - other.degree();
        for (x, y) in self.f.iter_mut().skip(diff).zip(other.f.iter()) {
            *x -= y;
        }
        if swapped {
            for x in &mut self.f {
                x.neg_assign();
            }
        }
        self
    }

    pub fn split_negative(&self, degree: usize) -> (PolyZ, PolyZ) {
        let mut pos = self.f[..=degree]
            .iter()
            .skip_while(|x| *x < &0)
            .cloned()
            .map(|x| if &x < &0 { Int::from(0) } else { x })
            .collect::<Vec<_>>();
        let mut neg = self.f[..=degree]
            .iter()
            .skip_while(|x| *x > &0)
            .cloned()
            .map(|x| if &x > &0 { Int::from(0) } else { -x })
            .collect::<Vec<_>>();
        if pos.is_empty() {
            pos.push(Int::from(0));
        }
        if neg.is_empty() {
            neg.push(Int::from(0));
        }
        (PolyZ { f: pos }, PolyZ { f: neg })
    }

    pub fn multiply(&self, other: &PolyZ) -> PolyZ {
        assert!(!self.f.is_empty());
        assert!(!other.f.is_empty());
        assert!(self.f.iter().all(|c| c >= &0));
        assert!(other.f.iter().all(|c| c >= &0));
        use rug::integer::Order;
        let deg = std::cmp::min(self.degree(), other.degree()) as f64;
        let max_g = self.f.iter().map(|x| x.significant_bits()).max().unwrap() as f64;
        let max_f = other.f.iter().map(|x| x.significant_bits()).max().unwrap() as f64;
        let digit_count = ((max_g + max_f + (deg + 1.0).log2()) / 64.0).ceil() as usize;

        fn evaluate_2(f: &PolyZ, digit_count: usize) -> Int {
            let mut vec: Vec<u64> = Vec::new();
            for x in f.f.iter().rev() {
                let digits = x.to_digits::<u64>(Order::Lsf);
                vec.extend_from_slice(&digits);
                vec.extend(std::iter::repeat(0u64).take(digit_count - digits.len()));
            }
            Int::from_digits(&vec, Order::Lsf)
        }

        let g = evaluate_2(self, digit_count);
        let f = evaluate_2(other, digit_count);
        let gf = g * f;
        let digits = gf.to_digits::<u64>(Order::Lsf);
        let len = (digits.len() as f64 / digit_count as f64).ceil() as usize;
        let mut result = Vec::new();
        for i in 0..len {
            let start = i * digit_count;
            let end = std::cmp::min(start + digit_count, digits.len());
            result.push(Int::from_digits(&digits[start..end], Order::Lsf));
        }
        result.reverse();
        if result.is_empty() {
            result.push(Int::from(0));
        }
        PolyZ { f: result }
    }

    pub fn evaluate_z(&self, at: &Int, degree: usize) -> Int {
        assert!(degree < self.f.len());
        let mut result = Int::from(0);
        for f in self.f.iter().take(degree + 1) {
            result *= at;
            result += f;
        }
        result
    }

    pub fn multi_evaluate_modulo(&self, at: &[Int], modulo: &Int, degree: usize) -> Vec<Int> {
        at.iter()
            .map(|at| self.evaluate_modulo(at, modulo, degree))
            .collect()
    }

    pub fn evaluate_modulo(&self, at: &Int, modulo: &Int, degree: usize) -> Int {
        assert!(degree < self.f.len());
        let mut result = Int::from(0);
        for f in self.f.iter().take(degree + 1) {
            result *= at;
            result += f;
            result %= modulo;
        }
        result
    }

    /// self = alpha * f_L + f_H
    pub fn shink(&mut self, alpha: &Int) {
        let len = self.f.len();
        let len_h = (len + 1) / 2;
        let len_l = len - len_h;
        let mut f_low = self.f.split_off(len_h);
        for (fh, fl) in self.f.iter_mut().take(len_l).zip(f_low.iter_mut()) {
            *fl *= alpha;
            *fh += &*fl;
        }
        self.f.truncate(len_h);
    }

    pub fn degree(&self) -> usize {
        assert!(!self.f.is_empty());
        self.f.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UniformRandom;
    use crate::{assert, assert_eq};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rug::ops::Pow;

    fn init_randomness() -> (ChaCha20Rng, UniformRandom) {
        let seed = [8u8; 32];
        let chacha = ChaCha20Rng::from_seed(seed);
        let uniform = UniformRandom::new();
        (chacha, uniform)
    }

    fn factor(mut x: Int, q: &Int) -> PolyZ {
        assert!(x != 0);
        assert!(x > 0);
        let mut result = Vec::new();
        while x > 0 {
            let mut r = q.clone();
            x.div_rem_mut(&mut r);
            result.push(r);
        }
        result.reverse();
        PolyZ { f: result }
    }

    #[test]
    fn test_q_d_div_l() {
        const TESTS: usize = 50;
        const Q_BITS: u32 = 4120;
        const L_BITS: u32 = 512;
        const D: u32 = 213;
        let (mut rng, mut uniform) = init_randomness();

        let mut q = Int::from(0);
        let mut l = Int::from(0);
        for _ in 0..TESTS {
            uniform.rand_size(Q_BITS, &mut rng, &mut q);
            uniform.rand_size(L_BITS, &mut rng, &mut l);

            let qq = q.clone().pow(D) / &l;
            let poly = factor(qq, &q);

            let num = q_d_div_l(&q, D, &l);

            assert_eq!(poly.f.len(), num.f.len());
            for (i, (a, b)) in poly.f.iter().zip(num.f.iter()).enumerate() {
                assert_eq!((i, a), (i, b));
            }
        }
    }
}
