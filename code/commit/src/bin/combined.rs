use poly_commit::dark::Instance;
use poly_commit::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

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

    let n: usize = std::env::args()
        .skip(1)
        .next()
        .map(|x| x.parse().unwrap())
        .expect("need an argument");

    let mut dark = {
        let key_path = format!("keys/{}_prover.json", n);
        let start = Instant::now();
        match DARK::<RSAGroup>::from_key(&key_path, true) {
            Ok(dark) => {
                println!("READ KEY: {:?}", Instant::now() - start);
                dark.save_key(&key_path, false).unwrap();
                dark
            }
            Err(e) => {
                println!("Error loading key {:?}", e);

                let start = Instant::now();
                let dark = DARK::<RSAGroup>::setup(n);
                println!("SETUP {:?}", Instant::now() - start);

                dark.save_key(&key_path, false).unwrap();

                let start = Instant::now();
                let dark = DARK::<RSAGroup>::from_key(&key_path, true).unwrap();
                println!("READ KEY {:?}", Instant::now() - start);

                dark
            }
        }
    };

    let mut rng = ChaCha20Rng::from_seed([1; 32]);
    let mut uniform = UniformRandom::new();
    let p = dark.p.clone();

    let k = 4;
    let poly = (0..k)
        .map(|i| PolyZp {
            p: p.clone(),
            f: (0..(n - i * (n / 11)))
                .map(|_| {
                    let mut r = Int::from(0);
                    uniform.rand_below(&p, &mut rng, &mut r);
                    r
                })
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let start = Instant::now();
    let commit_results = poly
        .into_iter()
        .map(|poly| {
            let (c, (p_z, _deg, r)) = dark.hiding_commit_zp(poly).unwrap();
            (c, (p_z, r))
        })
        .collect::<Vec<_>>();
    println!("COMMIT {:?}", (Instant::now() - start));

    let (commits, witnesses): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();

    let z: Vec<Int> = vec![
        1998.into(),
        2020.into(),
        128231.into(),
        3333.into(),
        99823.into(),
    ];
    let instances = (0..k)
        .map(|i| Instance {
            commitment: commits[i].clone(),
            y: witnesses[i]
                .0
                .multi_evaluate_modulo(&z, &p, witnesses[i].0.degree()),
            z: z.clone(),
            degree: witnesses[i].0.degree(),
            bound: (p.clone() - 1) / 2,
        })
        .collect::<Vec<_>>();

    let mut fiat = FiatShamirRng::new("hello", PierreGenPrime::new(128, 64));
    let prover = Prover::Witness(witnesses);

    let start = Instant::now();
    let (prover, instance) = dark.combine(prover, instances.clone(), &mut fiat).unwrap();
    println!("COMBINE {:?}", Instant::now() - start);

    let start = Instant::now();
    dark.multi_zk_eval(prover, instance, &mut fiat).unwrap();
    println!("EVAL {:?}", Instant::now() - start);
    println!("Proof length: {:?} bytes", fiat.proof_length());

    let proof = fiat.proofs;
    let mut fiat = FiatShamirRng::new("hello", PierreGenPrime::new(128, 64));
    let prover = Prover::Proof(proof);
    let start = Instant::now();
    let (prover, instance) = dark.combine(prover, instances, &mut fiat).unwrap();
    dark.multi_zk_eval(prover, instance, &mut fiat).unwrap();
    println!("VERIFY {:?}", Instant::now() - start);
}
