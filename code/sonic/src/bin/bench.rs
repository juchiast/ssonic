use poly_commit::*;
use rand::Rng;
use std::collections::VecDeque;
use std::time::Instant;
use std::{assert, assert_eq};
use supersonic::sonic;
use supersonic::sonic::*;

fn setup_sonic(max_deg: usize) -> Sonic {
    let key_path = format!("keys/test_key_{}.json", max_deg);
    let dark = {
        match DARK::<RSAGroup>::from_key(&key_path, true) {
            Ok(dark) => dark,
            Err(_) => {
                println!("SETTING UP");
                let dark = DARK::<RSAGroup>::setup(max_deg);
                dark.save_key(&key_path, false).unwrap();
                dark
            }
        }
    };
    Sonic { dark }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Proof {
    proof: VecDeque<ProofElement>,
    sk: SK,
}

fn read_proof() -> Result<Proof, Box<dyn std::error::Error>> {
    let r = serde_json::from_str(&std::fs::read_to_string("keys/powmod_proof.json")?)?;
    Ok(r)
}

fn bench_sonic_powmod() {
    let mut sonic = setup_sonic(5500);
    let proof = match read_proof() {
        Err(_) => {
            println!("PROVING");
            let mut rng = rand::thread_rng();
            // g^x = y mod p
            let p = sonic.dark.p.clone();
            let mut g = Int::from(0);
            UniformRandom::new().rand_below(&p, &mut rng, &mut g);
            let (x, x_bits) = {
                let mut x = rng.gen::<u64>();
                let temp = x;
                let mut x_bits = Vec::new();
                for _ in 0..64 {
                    x_bits.push(Int::from(x & 1));
                    x >>= 1;
                }
                x_bits.reverse();
                (temp, x_bits)
            };

            let circuit = supersonic::modulo::exp(64);
            let input = std::iter::once(g.clone())
                .chain(x_bits.into_iter())
                .collect::<Vec<_>>();
            let output = supersonic::circuit::evaluate(&circuit, &input, &p);

            assert_eq!(&output[1], &g);
            assert!((g.clone().pow_mod(&Int::from(x), &p).unwrap() - &output[0]).is_divisible(&p));

            let linear_circuit = supersonic::linear_circuit::convert(circuit);
            let left_input = input;
            let right_input = output;

            let uvwk = linear_circuit.to_constrains(&left_input, &p);
            let n = linear_circuit.n();
            let sk = to_sk(&uvwk, n, &p);

            let abc = linear_circuit.evaluate(&left_input, &right_input, &p);
            let rx_1 = to_rx_1(abc, &p);

            let mut fiat = FiatShamirRng::new("asdf", PierreGenPrime::new(128, 64));
            sonic.prove(Prover::Witness(rx_1), &sk, &mut fiat).unwrap();
            let proof = fiat.proofs;
            let r = Proof { proof, sk };
            std::fs::write(
                "keys/powmod_proof.json",
                &serde_json::to_string(&r).unwrap(),
            )
            .unwrap();
            r
        }
        Ok(proof) => proof,
    };

    println!(
        "Length: {} bytes",
        poly_commit::fiat_shamir::proof_length(&proof.proof)
    );

    let mut durations = Vec::new();
    let count = 30;

    for _ in 0..count {
        let mut fiat = FiatShamirRng::new("asdf", PierreGenPrime::new(128, 64));

        let p = proof.proof.clone();
        let start = Instant::now();
        sonic.prove(Prover::Proof(p), &proof.sk, &mut fiat).unwrap();
        durations.push((Instant::now() - start).as_secs_f64());
    }

    let a = durations.into_iter().collect::<average::Variance>();
    println!("VERIFY {} ± {}.", a.mean(), a.error());
}

fn bench_s_sha256() {
    let sonic = setup_sonic(5500);
    let p = sonic.dark.p.clone();

    let input = {
        let input: [u32; 16] = rand::random();

        let mut r = Vec::new();
        for u in &input {
            let mut u = *u;
            for _ in 0..32 {
                r.push(Int::from(u & 1));
                u >>= 1;
            }
        }
        r
    };

    let uint32_circuit = supersonic::uint32::sha256();
    let circuit = supersonic::circuit::convert(uint32_circuit);

    let linear_circuit = supersonic::linear_circuit::convert(circuit);
    let left_input = input;

    let uvwk = linear_circuit.to_constrains(&left_input, &p);
    let n = linear_circuit.n();
    let sk = to_sk(&uvwk, n, &p);

    let count = 1;
    let mut durations = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        sk.s.evaluate(&Int::from(1234), &Int::from(5678));
        durations.push((Instant::now() - start).as_secs_f64());
    }
    let a = durations.into_iter().collect::<average::Variance>();
    println!("S(X, Y) {} ± {}.", a.mean(), a.error());
}

fn main() {
    bench_s_sha256();
    bench_sonic_powmod();
}
