use super::uniform::UniformRandom;
use crate::assert;
use crate::Int;

const MILLER_RABIN_ROUND: u32 = 60;
const EULER_MASCHERONI: f64 = 0.5772156649;

fn next_prime(mut p: u32) -> u32 {
    p += 2;
    while !super::miller::is_prime(p) {
        p += 2;
        assert!(p < 4_000_000_000);
    }
    p
}

fn gen_m(l: u32) -> (Int, Int, f64) {
    let mut m = Int::from(1);
    let mut prime = 3;
    let mut carmichael_lambda_m = Int::from(prime - 1);
    let mut euler_totient = 1f64;
    while m.significant_bits() < l {
        m *= prime;
        euler_totient *= (prime - 1) as f64;
        carmichael_lambda_m.lcm_u_mut(prime - 1);
        prime = next_prime(prime);
    }
    (m, carmichael_lambda_m, euler_totient)
}

/// Implement "Close to Uniform Prime Number Generation With Fewer Random Bits"
pub struct PierreGenPrime {
    /// Bit count of output prime
    n: u32,
    /// (n - l) = bit count of m
    l: u32,

    m: Int,

    carmichael_lambda_m: Int,
    /// m-1
    m_1: Int,
    /// Random in uniform distribution
    uniform: UniformRandom,
    /// [0, 2^n / m)
    bound_alpha: Int,
    /// phi(m)
    euler_totient_m: f64,
}

impl PierreGenPrime {
    /// b: bit count of generated prime
    /// l: bit count of m
    pub fn new(n: u32, l: u32) -> Self {
        assert!(n > l);
        let (m, carmichael_lambda_m, euler_totient_m) = gen_m(n - l);
        let mut bound_alpha = Int::from(1);
        bound_alpha <<= n;
        bound_alpha /= &m;
        Self {
            n,
            l,
            m_1: m.clone() - 1,
            m,
            carmichael_lambda_m,
            uniform: UniformRandom::new(),
            bound_alpha,
            euler_totient_m,
        }
    }

    /// Algorithm 2
    pub fn gen<R: rand::Rng>(&mut self, rng: &mut R) -> Int {
        let b = {
            let mut b = Int::from(0);
            let mut r = Int::from(0);
            self.uniform.rand_below(&self.m_1, rng, &mut b);
            b += 1;
            loop {
                let u = 1 - b
                    .clone()
                    .pow_mod(&self.carmichael_lambda_m, &self.m)
                    .unwrap();
                if u == 0 {
                    break;
                }
                self.uniform.rand_below(&self.m_1, rng, &mut r);
                r += 1;
                r *= &u;
                r %= &self.m;
                b += &r;
            }
            b
        };
        assert!(b < self.m);
        assert!(1 == self.m.clone().gcd(&b));
        loop {
            let mut alpha = Int::from(0);
            self.uniform.rand_below(&self.bound_alpha, rng, &mut alpha);
            alpha *= &self.m;
            alpha += &b;
            if !matches!(
                alpha.is_probably_prime(MILLER_RABIN_ROUND),
                rug::integer::IsPrime::No
            ) {
                break alpha;
            }
        }
    }

    /// Theorem 1
    pub fn statistical_distance(&self) -> f64 {
        let x = self.m.to_f64() * self.bound_alpha.to_f64();
        let logx = x.ln();
        3.0 * logx * logx * (self.euler_totient_m / x).sqrt()
    }

    /// Expected number of prime test calls
    pub fn expected_prime_tests(&self) -> f64 {
        (self.n as f64) * 2f64.ln() * (-EULER_MASCHERONI).exp()
            / (2f64.ln().ln() + ((self.n - self.l) as f64).ln())
    }
}
