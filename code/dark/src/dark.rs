use common::traits::*;
use common::*;
use common::{assert, assert_eq};
use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use rug::ops::Pow;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const SIZE_R: u32 = 3100;
const LAMBDA: u32 = 120;
const MAX_JOINED_POLY: u32 = 4;

#[derive(Debug, Clone)]
pub struct Instance {
    pub commitment: Int,
    pub z: Vec<Int>,
    pub y: Vec<Int>,
    pub degree: usize,
    pub bound: Int,
}

pub struct DARK<G: Group> {
    pub p: Int,
    q: Int,
    pub max_d: usize,

    group: G,
    g: G::Element,

    /// (p - 1) / 2
    bound_alpha: Int,
    /// (p + 1) / 2
    p_1_2: Int,
    /// Size of hiding mask `r`.
    size_r: u32,

    cache: PreComputation<G>,
    rng: ChaCha20Rng,
    uniform: UniformRandom,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct VerifiableKey<G: Group> {
    p: Int,
    q: Int,
    max_d: usize,
    group: G,
    g: G::Element,
    /// g_powers[k] = g^(q^k)
    g_powers: Vec<G::Element>,
    /// Proof of correct computation of `g_powers`
    proof: VecDeque<ProofElement>,
}

fn proof_of_precomputation(
    mut prover: Prover<()>,
    key: &VerifiableKey<RSAGroup>,
    fiat: &mut FiatShamirRng,
) -> Result<(), String> {
    if &key.g_powers[0] != &key.g {
        return Err("g[0]".to_owned());
    }
    for i in 1..key.g_powers.len() {
        prover = crate::proof_of_exponentation(
            prover,
            &key.group,
            &key.g_powers[i - 1],
            &key.g_powers[i],
            &key.q,
            fiat,
        )?;
    }
    Ok(())
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PreComputation<G: Group> {
    /// g_powers[k] = g^(q^k)
    g_powers: Arc<[G::Element]>,
    /// Proof of correct computation of `g_powers`
    proof: VecDeque<ProofElement>,
}

impl PreComputation<RSAGroup> {
    fn new(group: &RSAGroup, g: &Int, q: &Int, max_d: usize) -> Self {
        let seed: &[&dyn FeedHasher] = &[group, g, q, &max_d];
        let mut fiat = FiatShamirRng::new(seed, PierreGenPrime::new(128, 64));
        let mut g_powers = vec![g.clone()];
        for i in 1..=(max_d + 1) {
            if i % 10000 == 0 {
                trace!("Setup {}", i);
            }
            let last = g_powers.last().unwrap();
            let l = fiat.verifier_rand_prime();
            let (k, r) = q.clone().div_rem(l.clone());
            let temp = group.power(last, &k);
            fiat.prover_send(&temp);
            let p = group.mul(&group.power(&temp, &l), &group.power(last, &r));
            g_powers.push(p);
        }
        Self {
            g_powers: g_powers.into(),
            proof: fiat.proofs,
        }
    }
}

impl DARK<RSAGroup> {
    /// Setup with 120-bit security, for polynomial with degree < max_d
    pub fn setup(max_d: usize) -> Self {
        let mut rng = init_chacha20();
        let mut uniform = UniformRandom::new();
        let size_r = SIZE_R;
        let log2_d = ((max_d + 1) as f64).log2().ceil() as u32;
        let p = PierreGenPrime::new(LAMBDA + log2_d + 1, 80).gen(&mut rng);
        let group = RSAGroup::new(3048);
        let g = {
            let mut result = Int::from(0);
            loop {
                uniform.rand_below(&group.modulo, &mut rng, &mut result);
                if result > 1 && group.is_element(&result) {
                    break result;
                }
            }
        };
        let mut q = Int::from(0);
        let q_size = p.significant_bits() * (2 * log2_d + 4 + MAX_JOINED_POLY);
        uniform.rand_size(q_size, &mut rng, &mut q);
        q.set_bit(0, true); // q must be odd
        let mut bound_alpha = Int::from(&p - 1);
        bound_alpha >>= 1;
        let mut p_1_2 = p.clone();
        p_1_2 += 1;
        p_1_2 >>= 1;
        Self {
            cache: PreComputation::new(&group, &g, &q, max_d),
            p,
            q,
            max_d,
            group,
            g,
            rng,
            uniform,
            bound_alpha,
            p_1_2,
            size_r,
        }
    }

    /// Load a key and verify its proof.
    pub fn from_key<P: AsRef<Path>>(path: P, verify: bool) -> Result<Self, String> {
        let mut key: VerifiableKey<RSAGroup> = serde_json::from_reader(
            std::fs::OpenOptions::new()
                .read(true)
                .open(path)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let proof = std::mem::take(&mut key.proof);
        if verify {
            let seed: &[&dyn FeedHasher] = &[&key.group, &key.g, &key.q, &key.max_d];
            let mut fiat = FiatShamirRng::new(seed, PierreGenPrime::new(128, 64));
            proof_of_precomputation(Prover::Proof(proof.clone()), &key, &mut fiat)?;
        }
        let VerifiableKey {
            p,
            q,
            group,
            g,
            max_d,
            g_powers,
            ..
        } = key;
        let rng = init_chacha20();
        let uniform = UniformRandom::new();
        let size_r = SIZE_R;
        let mut bound_alpha = Int::from(&p - 1);
        bound_alpha >>= 1;
        let mut p_1_2 = p.clone();
        p_1_2 += 1;
        p_1_2 >>= 1;
        Ok(Self {
            cache: PreComputation {
                g_powers: g_powers.into(),
                proof,
            },
            p,
            q,
            max_d,
            group,
            g,
            rng,
            uniform,
            bound_alpha,
            p_1_2,
            size_r,
        })
    }

    /// Generate proof of correct pre-computation and save the key.
    pub fn save_key<P: AsRef<Path>>(&self, path: P, pretty: bool) -> Result<(), String> {
        let key = VerifiableKey {
            p: self.p.clone(),
            q: self.q.clone(),
            group: self.group.clone(),
            g: self.g.clone(),
            max_d: self.max_d,
            g_powers: self.cache.g_powers.to_vec(),
            proof: self.cache.proof.clone(),
        };
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        if pretty {
            serde_json::to_writer_pretty(f, &key).map_err(|e| e.to_string())
        } else {
            serde_json::to_writer(f, &key).map_err(|e| e.to_string())
        }
    }

    /// Commit a polynomial in Z_p[X].
    pub fn commit_zp(&mut self, polynomial: PolyZp) -> Result<Int, String> {
        check(&polynomial.p == &self.p, "polynomial has different modulo")?;

        let deg = polynomial.f.len() - 1;
        self.commit_z(PolyZ::new_bounded(polynomial), deg)
    }

    /// Commit a polynomial in Z(b)[X].
    pub fn commit_z(&mut self, mut p: PolyZ, degree: usize) -> Result<Int, String> {
        check(degree <= self.max_d, "degree > max_d")?;
        check(degree < p.f.len(), "coefficient overflow")?;

        p.f.truncate(degree + 1);
        let f =
            p.f.into_iter()
                .enumerate()
                .map(|(i, coeff)| (degree - i, coeff))
                .filter(|(_, coeff)| coeff != &0)
                .collect::<Vec<_>>();
        let cpus = num_cpus::get();
        let size = f.len() / cpus;
        let mut ranges = (0..cpus)
            .map(|i| (size * i, size * (i + 1)))
            .collect::<Vec<_>>();
        ranges.last_mut().unwrap().1 = f.len();
        let p: Arc<[(usize, Int)]> = f.into();
        let mut tasks = Vec::new();

        if log_enabled!(log::Level::Trace) {
            let max_bits: u32 = ranges
                .iter()
                .map(|(from, to)| {
                    p[*from..*to]
                        .iter()
                        .map(|(_, coeff)| coeff.significant_bits())
                        .sum()
                })
                .max()
                .unwrap();
            trace!("exp bits: {}", max_bits);
        }

        for (from, to) in ranges {
            let p = p.clone();
            let g = self.cache.g_powers.clone();
            let group = self.group.clone();
            tasks.push(move || {
                let mut result = Int::from(1);
                for i in from..to {
                    let temp = group.power(&g[p[i].0], &p[i].1);
                    result = group.mul(&result, &temp);
                }
                result
            });
        }

        let nums = tasks.into_par_iter().map(|f| f()).collect::<Vec<_>>();
        let mut result = Int::from(1);
        for x in nums {
            result = self.group.mul(&result, &x);
        }
        Ok(result)
    }

    pub fn multi_eval_z(
        &mut self,
        prover: Prover<PolyZ>,
        instance: Instance,
        fiat: &mut FiatShamirRng,
    ) -> Result<Prover<()>, String> {
        let Instance {
            mut degree,
            commitment,
            mut y,
            z,
            mut bound,
        } = instance;
        let mut commit = commitment;
        assert_eq!(z.len(), y.len());

        let log2_d = ((degree + 1) as f64).log2();

        let mut outer_prover = Some(prover);
        while degree != 0 {
            let prover = outer_prover.take().unwrap();

            if (degree + 1) % 2 != 0 {
                commit = self.group.power(&commit, &self.q);
                for (y, z) in y.iter_mut().zip(z.iter()) {
                    *y *= z;
                    *y %= &self.p;
                }
            }

            let deg_h = degree / 2;

            let ((y_high, commit_high), prover) = prover!(fiat, prover, (p) => {
                trace!("degree = {}", degree);
                check(p.degree() == degree, "invalid witness degree")?;
                let y_high = p.multi_evaluate_modulo(&z, &self.p, deg_h);
                let start = Instant::now();
                let commit_high = self.commit_z(
                    PolyZ {
                        f: p.f[..(deg_h + 1)].to_owned(),
                    },
                    deg_h,
                )?;
                trace!("perf:commit: {:?}", Instant::now() - start);
                ((y_high, commit_high), (p))
            });

            let l = fiat.verifier_rand_prime();

            let (qq, prover) = prover!(fiat, prover, (p) => {
                trace!("PoE");
                let start = Instant::now();
                let (positive, negative) = p.split_negative(deg_h);
                let q_d_div_l_0 = poly::q_d_div_l(&self.q, deg_h as u32 + 1, &l);
                let q_d_div_l_1 = q_d_div_l_0.clone();
                let (a, b) = rayon::join(
                    move || q_d_div_l_0.multiply(&positive),
                    move || q_d_div_l_1.multiply(&negative),
                );
                let temp = a.substract(b);
                let temp_deg = temp.degree();
                let qq = self.commit_z(temp, temp_deg)?;
                trace!("perf:PoE: {:?}", Instant::now() - start);
                (qq, (p))
            });

            let dd = Int::from(deg_h + 1);
            let r = Int::from(&self.q % &l).pow_mod(&dd, &l).unwrap();
            let qq_l = self.group.power(&qq, &l);
            let c_r = self.group.power(&commit_high, &r);
            let commit_low = self.group.div(&commit, &self.group.mul(&qq_l, &c_r));

            let y_low = (0..z.len())
                .map(|i| y[i].clone() - z[i].clone().pow_mod(&dd, &self.p).unwrap() * &y_high[i])
                .collect::<Vec<_>>();

            let alpha = fiat.verifier_rand_signed(&self.bound_alpha);

            degree /= 2;

            commit = self.group.power(&commit_low, &alpha);
            commit = self.group.mul(&commit, &commit_high);

            y = (0..z.len())
                .map(|i| (y_low[i].clone() * &alpha + &y_high[i]) % &self.p)
                .collect();

            bound *= &self.p_1_2;

            let (_, prover) = prover!(fiat, prover, (mut p) => {
                p.shink(&alpha);
                ((), (p))
            });

            outer_prover = Some(prover);
        }

        let prover = outer_prover.unwrap();
        let (f, prover) = prover!(fiat, prover, (mut p) => {
            check(p.f.len() == 1, "invalid witness degree")?;
            (p.f.pop().unwrap(), ())
        });

        for y in y {
            check((y - &f).is_divisible(&self.p), "y != f mod p")?;
        }

        let mut lhs = self.p.clone().pow(log2_d.ceil() as u32);
        lhs *= &bound;
        check(&lhs < &self.q, "sigma")?;

        check(&f.clone().abs() <= &bound, "bound")?;

        crate::proof_of_exponentation(prover, &self.group, &self.g, &commit, &f, fiat)
    }

    pub fn hiding_commit_zp(
        &mut self,
        polynomial: PolyZp,
    ) -> Result<(Int, (PolyZ, usize, Int)), String> {
        let poly_z = PolyZ::new_bounded(polynomial);
        let (commit, (deg, r)) = self.hiding_commit_z(poly_z.clone())?;
        Ok((commit, (poly_z, deg, r)))
    }

    pub fn hiding_commit_z(&mut self, polynomial: PolyZ) -> Result<(Int, (usize, Int)), String> {
        let mut r = Int::from(0);
        self.uniform.rand_size(self.size_r, &mut self.rng, &mut r);
        let deg = polynomial.degree();
        let temp = self.commit_z(polynomial, deg)?;
        let commit = self
            .group
            .mul(&temp, &self.group.power(&self.cache.g_powers[deg + 1], &r));
        Ok((commit, (deg, r)))
    }

    pub fn multi_zk_eval(
        &mut self,
        prover: Prover<(PolyZ, Int)>,
        instance: Instance,
        fiat: &mut FiatShamirRng,
    ) -> Result<Prover<()>, String> {
        let Instance {
            degree,
            commitment,
            y,
            z,
            bound,
        } = instance;
        let commit = commitment;
        assert_eq!(z.len(), y.len());

        let bound_coeff_k = bound.clone() * &self.bound_alpha * (Int::from(1) << LAMBDA);
        let ((commit_k, y_k), prover) = prover!(fiat, prover, (p, r) => {
            let mut k = Vec::new();
            k.resize_with(degree + 1, || {
                let mut result = Int::from(0);
                self.uniform
                    .rand_signed(&bound_coeff_k, &mut self.rng, &mut result);
                result
            });
            let k = PolyZ { f: k };
            let (commit_k, (deg, r_k)) = self.hiding_commit_z(k.clone())?;
            assert_eq!(deg, degree);
            let y_k = k.multi_evaluate_modulo(&z, &self.p, deg);
            ((commit_k, y_k), (p, r, k, r_k))
        });

        let c = fiat.verifier_rand_signed(&self.bound_alpha);

        let (r_s, prover) = prover!(fiat, prover, (p, r, k, r_k) => {
            assert_eq!(p.f.len(), k.f.len());
            let mut s = p;
            for (a, b) in s.f.iter_mut().zip(k.f.iter()) {
                *a *= &c;
                *a += b;
            }
            let r_s = r * &c + &r_k;
            (r_s, (s))
        });

        let commit_s = self.group.mul(
            &self.group.mul(&self.group.power(&commit, &c), &commit_k),
            &self
                .group
                .inverse(&self.group.power(&self.cache.g_powers[degree + 1], &r_s)),
        );
        let y_s = (0..z.len())
            .map(|i| (y[i].clone() * &c + &y_k[i]) % &self.p)
            .collect();
        let bound_s = bound.clone() * &self.bound_alpha + &bound_coeff_k;

        self.multi_eval_z(
            prover,
            Instance {
                degree,
                commitment: commit_s,
                y: y_s,
                z,
                bound: bound_s,
            },
            fiat,
        )
    }

    /// Combining a vector of instances and witnesses in to ONE.
    pub fn combine(
        &mut self,
        prover: Prover<Vec<(PolyZ, Int)>>,
        mut instances: Vec<Instance>,
        fiat: &mut FiatShamirRng,
    ) -> Result<(Prover<(PolyZ, Int)>, Instance), String> {
        assert!(!instances.is_empty());
        assert!(instances.len() <= MAX_JOINED_POLY as usize);
        let k = instances[0].z.len();
        for x in &instances {
            assert_eq!(x.z.len(), k);
            assert_eq!(x.y.len(), k);
        }
        let (_, prover) = prover!(fiat, prover, (witnesses) => {
            assert_eq!(witnesses.len(), instances.len());
            for i in 0..witnesses.len() {
                let result = witnesses[i]
                    .0
                    .multi_evaluate_modulo(&instances[i].z, &self.p, witnesses[i].0.degree());
                assert!(result
                    .into_iter()
                    .zip(instances[i].y.iter())
                    .all(|(a, b)| (a - b).is_divisible(&self.p)));
            }
            ((), witnesses)
        });

        let max_deg = instances.iter().map(|x| x.degree).max().unwrap();

        // Shift the polynomials and their commitments,
        // to equalize the degree in each statements
        let mut outer_prover = Some(prover);
        for (i, x) in instances.iter_mut().enumerate() {
            let deg_diff = max_deg - x.degree;
            if deg_diff == 0 {
                continue;
            }

            let prover = outer_prover.take().unwrap();

            let l = fiat.verifier_rand_prime();

            let (qq, prover) = prover!(fiat, prover, (mut witnesses) => {
                trace!("deg_diff = {}", deg_diff);
                let (f, r) = witnesses.get_mut(i).unwrap();

                let ff = PolyZ {
                    f: std::iter::once(r.clone())
                        .chain(f.f.iter().cloned())
                        .collect(),
                };
                let (ff_pos, ff_neg) = ff.split_negative(ff.degree());
                let h_0 = poly::q_d_div_l(&self.q, deg_diff as u32, &l);
                let h_1 = h_0.clone();
                let (a, b) = rayon::join(
                    move || h_0.multiply(&ff_pos),
                    move || h_1.multiply(&ff_neg),
                );
                let temp = a.substract(b);
                let temp_deg = temp.degree();
                let qq = self.commit_z(temp, temp_deg)?;

                f.f.extend(std::iter::repeat(Int::from(0)).take(deg_diff));
                (qq, witnesses)
            });

            let r = Int::from(&self.q)
                .pow_mod(&Int::from(deg_diff), &l)
                .unwrap();

            x.commitment = self.group.mul(
                &self.group.power(&qq, &l),
                &self.group.power(&x.commitment, &r),
            );
            x.degree = max_deg;
            for (y, z) in x.y.iter_mut().zip(x.z.iter()) {
                *y *= z.clone().pow_mod(&Int::from(deg_diff), &self.p).unwrap();
                *y %= &self.p;
            }

            outer_prover = Some(prover);
        }

        let mut result = instances.pop().unwrap();
        let prover = outer_prover.unwrap();
        let (_, prover) = prover!(fiat, prover, (mut witnesses) => {
            let result = witnesses.pop().unwrap();
            ((), (witnesses, result))
        });

        let mut outer_prover = Some(prover);
        while let Some(instance) = instances.pop() {
            let prover = outer_prover.take().unwrap();

            let c = fiat.verifier_rand_signed(&self.bound_alpha);

            result.commitment = self.group.mul(
                &result.commitment,
                &self.group.power(&instance.commitment, &c),
            );
            assert_eq!(result.degree, instance.degree);
            result.bound += instance.bound * &self.bound_alpha;

            for (y, y_1) in result.y.iter_mut().zip(instance.y.into_iter()) {
                *y += y_1.clone() * &c;
            }
            assert!(result.z.iter().zip(instance.z.iter()).all(|(a, b)| a == b));

            let (_, prover) = prover!(fiat, prover, (mut witnesses, result) => {
                let (mut f, mut r) = result;
                let (f_1, r_1) = witnesses.pop().unwrap();

                assert_eq!(f.degree(), f_1.degree());

                for (f, f_1) in f.f.iter_mut().zip(f_1.f.into_iter()) {
                    *f += f_1 * &c;
                }

                r += r_1 * &c;

                ((), (witnesses, (f, r)))
            });

            outer_prover = Some(prover);
        }
        let prover = outer_prover.unwrap();

        let (_, prover) = prover!(fiat, prover, (_, witness) => {
            assert!({
                let evaluated = witness.0.multi_evaluate_modulo(&result.z, &self.p, witness.0.degree());
                evaluated.into_iter().zip(result.y.iter()).all(|(a, b)| (a - b).is_divisible(&self.p))
            });
            ((), witness)
        });
        assert!(&result.bound > &0);
        Ok((prover, result))
    }
}

/// Initialise ChaCha20 with seed from OS's randomness.
fn init_chacha20() -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    ChaCha20Rng::from_seed(seed)
}

fn check<S: std::string::ToString>(ok: bool, err_str: S) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(err_str.to_string())
    }
}
