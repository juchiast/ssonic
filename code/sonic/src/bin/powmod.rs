use common::*;
use dark::DARK;
use rand::Rng;
use sonic::sonic::*;
use std::time::Instant;

fn setup_sonic(max_deg: usize) -> Sonic {
    let key_path = format!("keys/{}.json", max_deg);
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
        .target(env_logger::Target::Stdout)
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

    let bit_size: usize = std::env::args()
        .skip(1)
        .next()
        .map(|x| x.parse().unwrap())
        .expect("need an argument");

    let mut sonic = setup_sonic(5500);
    let mut rng = rand::thread_rng();

    // g^x = y mod p
    let p = sonic.dark.p.clone();

    let mut g = Int::from(0);
    UniformRandom::new().rand_below(&p, &mut rng, &mut g);

    let x_bits = {
        let mut x_bits = Vec::new();
        for _ in 0..bit_size {
            x_bits.push(Int::from(if rng.gen_bool(0.5) { 1 } else { 0 }));
        }
        x_bits
    };

    let circuit = sonic::modulo::exp(bit_size);
    let input = std::iter::once(g.clone())
        .chain(x_bits.into_iter())
        .collect::<Vec<_>>();
    let output = sonic::circuit::evaluate(&circuit, &input, &p);

    let start = Instant::now();
    let linear_circuit = sonic::linear_circuit::convert(circuit);
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
