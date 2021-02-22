#![allow(warnings)]

extern crate dark;

use common::*;
use dark::{Instance, DARK};
use rand::Rng;
use rand::SeedableRng;
use rug::ops::Pow;
use std::collections::VecDeque;
use std::time::Instant;

fn init_randomness() -> (ChaCha20Rng, UniformRandom) {
    let seed = [8u8; 32];
    let chacha = ChaCha20Rng::from_seed(seed);
    let uniform = UniformRandom::new();
    (chacha, uniform)
}

fn rand_number(
    max_d: usize,
    q: &Int,
    rng: &mut ChaCha20Rng,
    uniform: &mut UniformRandom,
) -> Vec<Int> {
    let d = rng.gen_range(max_d / 2, max_d);
    let mut digits = Vec::with_capacity(d);
    digits.resize_with(d, || {
        let mut x = Int::from(0);
        uniform.rand_below(&q, rng, &mut x);
        x
    });
    digits
}

fn bench_q_d_div_l() {
    let (mut rng, mut uniform) = init_randomness();
    let mut q = Int::from(0);
    let mut l = Int::from(0);
    let d = 100000;
    uniform.rand_size(3000, &mut rng, &mut q);
    uniform.rand_size(512, &mut rng, &mut l);

    println!("Start q_d_div_l");
    let start = Instant::now();
    poly::q_d_div_l(&q, d, &l);
    println!("q_d_div_l: {:?}", Instant::now() - start);
}

fn bench_gmp_big_mul() {
    let (mut rng, mut uniform) = init_randomness();
    let mut q = Int::from(0);
    let mut l0 = Int::from(0);
    let mut l1 = Int::from(0);
    let d = 100000;
    uniform.rand_size(3000, &mut rng, &mut q);
    uniform.rand_size(512, &mut rng, &mut l0);
    uniform.rand_size(512, &mut rng, &mut l1);

    let qq0 = q.clone().pow(d) / &l0;
    let qq1 = q.clone().pow(d) / &l1;

    dbg!(qq0.significant_bits());
    dbg!(qq1.significant_bits());
    println!("Start GMP big mul");
    let start = Instant::now();
    let x = qq0 * qq1;
    println!("GMP big mul: {:?}", Instant::now() - start);
    dbg!(x.significant_bits());
}

fn bench_poly_multiplication() {
    let (mut rng, mut uniform) = init_randomness();
    let d = 100000;
    let mut q = Int::from(0);
    let mut l = Int::from(0);
    uniform.rand_size(3000, &mut rng, &mut q);
    uniform.rand_size(512, &mut rng, &mut l);
    let q_d_div_l = poly::q_d_div_l(&q, d, &l);
    let f = std::iter::repeat_with(|| {
        let mut x = Int::from(0);
        uniform.rand_size(512, &mut rng, &mut x);
        x
    })
    .take(d as usize)
    .collect::<Vec<_>>();
    let f = PolyZ { f };
    println!("Start poly mul");
    let start = Instant::now();
    q_d_div_l.multiply(&f);
    println!("Poly mul: {:?}", Instant::now() - start);
}

fn bench_verification() {
    let key_path = "bench_data/bench_verification_key.json";
    let n = 10000;
    let mut dark = {
        match DARK::<RSAGroup>::from_key(key_path, false)
            .map_err(|_| ())
            .and_then(|dark| if dark.max_d != n { Err(()) } else { Ok(dark) })
        {
            Ok(dark) => dark,
            Err(e) => {
                let dark = DARK::<RSAGroup>::setup(n);
                dark.save_key(key_path, false).unwrap();
                dark
            }
        }
    };

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Proof {
        y: Int,
        z: Int,
        commit: Int,
        degree: usize,
        bound: Int,
        proof: VecDeque<ProofElement>,
    }

    let Proof {
        y,
        z,
        commit,
        degree,
        bound,
        proof,
    } = {
        let proof_path = "bench_data/bench_verification_proof.json";
        let read_proof = || -> Result<Proof, ()> {
            let bytes = std::fs::read(proof_path).map_err(|_| ())?;
            serde_json::from_slice(&bytes).map_err(|_| ())
        };
        match read_proof() {
            Ok(proof) => proof,
            Err(_) => {
                let mut rng = ChaCha20Rng::from_seed([1; 32]);
                let mut uniform = UniformRandom::new();
                let p = dark.p.clone();
                let f = (0..n)
                    .map(|_| {
                        let mut r = Int::from(0);
                        uniform.rand_below(&p, &mut rng, &mut r);
                        r
                    })
                    .collect::<Vec<_>>();

                let poly = PolyZp {
                    p: p.clone(),
                    f: f.clone(),
                };

                let (c, (p_z, _deg, r)) = dark.hiding_commit_zp(poly.clone()).unwrap();

                let z = Int::from(1998);
                let y = poly.evaluate(&z);
                let degree = poly.degree();
                let bound: Int = (p - 1) / 2;

                let gen_prime = PierreGenPrime::new(128, 64);
                let mut fiat = FiatShamirRng::new("hello", gen_prime);

                dark.multi_zk_eval(
                    Prover::Witness((p_z, r)),
                    Instance {
                        commitment: c.clone(),
                        bound: bound.clone(),
                        y: vec![y.clone()],
                        z: vec![z.clone()],
                        degree,
                    },
                    &mut fiat,
                )
                .unwrap();
                let proof = Proof {
                    y,
                    z,
                    degree,
                    bound,
                    commit: c,
                    proof: fiat.proofs,
                };

                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(proof_path)
                    .map_err(|e| e.to_string())
                    .unwrap();
                serde_json::to_writer(f, &proof).unwrap();

                proof
            }
        }
    };

    let mut durations = Vec::new();
    let count = 60;

    for _ in 0..count {
        let proof = proof.clone();
        let gen_prime = PierreGenPrime::new(128, 64);
        let mut fiat = FiatShamirRng::new("hello", gen_prime);

        let start = Instant::now();
        dark.multi_zk_eval(
            Prover::Proof(proof),
            Instance {
                commitment: commit.clone(),
                bound: bound.clone(),
                y: vec![y.clone()],
                z: vec![z.clone()],
                degree,
            },
            &mut fiat,
        )
        .unwrap();
        let elapsed = Instant::now() - start;
        durations.push(elapsed.as_secs_f64());
    }

    let a = durations.into_iter().collect::<average::Variance>();
    println!("VERIFY {} ± {}.", a.mean(), a.error());
}

fn main() {
    bench_verification();
}
