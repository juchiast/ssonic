#[macro_use]
extern crate log;

use poly_commit::*;
use std::time::Instant;
use supersonic::sonic::*;

fn setup_sonic(max_deg: usize) -> Sonic {
    trace!("Reading key");
    let key_path = format!("keys/test_key_{}.json", max_deg);
    let dark = {
        match DARK::<RSAGroup>::from_key(&key_path, true) {
            Ok(dark) => dark,
            Err(_) => {
                let start = Instant::now();
                let dark = DARK::<RSAGroup>::setup(max_deg);
                println!("Setup: {:?}", Instant::now() - start);
                dark.save_key(&key_path, false).unwrap();
                dark
            }
        }
    };
    Sonic { dark }
}

fn main() {
    let env = env_logger::Env::new().filter("LOG");
    env_logger::from_env(env)
        .format({
            use std::io::Write;
            let start = Instant::now();
            move |buf, record| {
                writeln!(
                    buf,
                    "{:?} {}: {}",
                    (Instant::now() - start),
                    record.level(),
                    record.args()
                )
            }
        })
        .init();

    // let mut sonic = setup_sonic(3000000);
    let mut sonic = setup_sonic(5500);
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
    let output = supersonic::circuit::evaluate(&circuit, &input, &p);

    let start = Instant::now();
    let linear_circuit = supersonic::linear_circuit::convert(circuit);
    let left_input = input;
    let right_input = output;

    let uvwk = linear_circuit.to_constrains(&left_input, &p);
    let n = linear_circuit.n();
    let sk = to_sk(&uvwk, n, &p);

    let abc = linear_circuit.evaluate(&left_input, &right_input, &p);
    let rx_1 = to_rx_1(abc, &p);

    let mut fiat = FiatShamirRng::new("hello", PierreGenPrime::new(128, 64));
    sonic.prove(Prover::Witness(rx_1), &sk, &mut fiat).unwrap();
    println!("Prove: {:?}", Instant::now() - start);

    println!("Length: {} bytes", fiat.proof_length());

    let proof = fiat.proofs;
    let mut fiat = FiatShamirRng::new("hello", PierreGenPrime::new(128, 64));
    let start = Instant::now();
    sonic.prove(Prover::Proof(proof), &sk, &mut fiat).unwrap();
    println!("Verify: {:?}", Instant::now() - start);
}
