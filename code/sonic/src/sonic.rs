use crate::{SparseBiPolyZp, SparsePolyZp};
use common::{assert, assert_eq};
use common::{prover, FiatShamirRng, Int, PolyZ, PolyZp, Prover, RSAGroup};
use dark::DARK;
use rug::Assign;
use std::num::NonZeroUsize;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ABC {
    pub a: Int,
    pub b: Int,
    pub c: Int,
}

impl ABC {
    pub fn new() -> Self {
        ABC {
            a: Int::from(0),
            b: Int::from(0),
            c: Int::from(0),
        }
    }
}

// for testing
pub fn to_rx_y(abc: &[ABC], modulo: &Int, n: usize) -> SparseBiPolyZp {
    let n = n as i32;
    let mut r = Vec::new();
    for (i, abc) in abc.iter().enumerate().map(|(i, abc)| (i as i32 + 1, abc)) {
        r.push((abc.a.clone(), i, i));
        r.push((abc.b.clone(), -i, -i));
        r.push((abc.c.clone(), -i - n, -i - n));
    }
    SparseBiPolyZp {
        p: modulo.clone(),
        coeff: r,
    }
}

pub fn to_rx_1(abc: Vec<ABC>, modulo: &Int) -> PolyZp {
    let a = abc.iter().map(|abc| abc.a.clone()).rev();
    let b = abc.iter().map(|abc| abc.b.clone());
    let c = abc.iter().map(|abc| abc.c.clone());
    let f = a
        .chain(std::iter::once(Int::from(0)))
        .chain(b)
        .chain(c)
        .collect();
    PolyZp::new(PolyZ { f }, modulo)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UVWK {
    pub u: Option<(Int, NonZeroUsize)>,
    pub v: Option<(Int, NonZeroUsize)>,
    pub w: Option<(Int, NonZeroUsize)>,
    pub k: Int,
}

pub fn to_sk(uvwk: &[UVWK], n: usize, modulo: &Int) -> SK {
    let n = n as i32;
    let mut s = Vec::new();
    let mut k = Vec::new();
    for (q, uvwk) in uvwk
        .iter()
        .enumerate()
        .map(|(q, uvwk)| (q as i32 + 1, uvwk))
    {
        if matches!(&uvwk.u, Some((value, _)) if value != &0) {
            let (value, i) = uvwk.u.as_ref().unwrap();
            let (value, i) = (value.clone(), i.get() as i32);
            s.push((value, -i, q + n));
        }
        if matches!(&uvwk.v, Some((value, _)) if value != &0) {
            let (value, i) = uvwk.v.as_ref().unwrap();
            let (value, i) = (value.clone(), i.get() as i32);
            s.push((value, i, q + n));
        }
        if matches!(&uvwk.w, Some((value, _)) if value != &0) {
            let (value, i) = uvwk.w.as_ref().unwrap();
            let (value, i) = (value.clone(), i.get() as i32);
            s.push((value, i + n, q + n));
        }
        if &uvwk.k != &0 {
            k.push((uvwk.k.clone(), q + n));
        }
    }
    for i in 1..=n {
        s.push((Int::from(-1), i + n, i));
        s.push((Int::from(-1), i + n, -i));
    }

    SK {
        n: n as usize,
        modulo: modulo.clone(),
        s: SparseBiPolyZp {
            coeff: s,
            p: modulo.clone(),
        },
        k: SparsePolyZp {
            coeff: k,
            p: modulo.clone(),
        },
    }
}

pub fn is_sastified(abc: &[ABC], uvwks: &[UVWK], modulo: &Int) -> Result<(), usize> {
    for (q, uvwk) in uvwks.iter().enumerate() {
        let mut sum = Int::from(0);
        if let Some((u, index)) = &uvwk.u {
            sum += u.clone() * &abc[index.get() - 1].a;
        }
        if let Some((v, index)) = &uvwk.v {
            sum += v.clone() * &abc[index.get() - 1].b;
        }
        if let Some((w, index)) = &uvwk.w {
            sum += w.clone() * &abc[index.get() - 1].c;
        }
        sum -= &uvwk.k;
        if !sum.is_divisible(modulo) {
            return Err(q);
        }
    }
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct SK {
    /// Size of vector a, b, c
    pub n: usize,
    pub modulo: Int,
    pub s: SparseBiPolyZp,
    pub k: SparsePolyZp,
}

pub struct Sonic {
    pub dark: DARK<RSAGroup>,
}

impl Sonic {
    pub fn prove(
        &mut self,
        prover: Prover<PolyZp>,
        sk: &SK,
        fiat: &mut FiatShamirRng,
    ) -> Result<(), String> {
        let n = sk.n;
        info!("n = {}", n);
        info!("S(X, Y) length: {}", sk.s.coeff.len());
        info!("k(Y) length: {}", sk.k.coeff.len());
        let (rx_1_commit, prover) = prover!(fiat, prover, (rx_1) => {
            assert_eq!(&sk.modulo, &rx_1.p);
            assert_eq!(rx_1.degree(), 3 * n);
            trace!("Commit r(X, 1)");
            let (commitment, (rx_1_z, _, r_rx_1)) = self.dark.hiding_commit_zp(rx_1.clone())?;
            (commitment, (rx_1, rx_1_z, r_rx_1))
        });

        let y = fiat.verifier_rand_below(&sk.modulo);

        // t(X, y) = r(X, 1) x [r(X, y) + s(X, y)]
        let (tx_y_commit, prover) = prover!(fiat, prover, (rx_1, rx_1_z, r_rx_1) => {
            trace!("Commit t(X, y)");
            let tx_y = get_tx_y(&rx_1, sk, n, &y);
            assert_eq!(&tx_y.f[tx_y.degree() - 4 * n], &0);
            let (commit, (tx_y_z, _, r_tx_y)) = self.dark.hiding_commit_zp(tx_y.clone())?;
            (commit, (vec![(rx_1_z, r_rx_1), (tx_y_z, r_tx_y)]))
        });

        // coefficient query
        let ((tx_y_l_commit, tx_y_h_commit), prover) = prover!(fiat, prover, (mut witnesses) => {
            trace!("Commit t_low/high");

            let at = 4 * n;
            let (tx_y, _r) = &witnesses[1];

            let deg = tx_y.degree();
            assert!(tx_y.f[deg - at].is_divisible(&sk.modulo));
            let tx_y_l = PolyZp::new(
                PolyZ {
                    f: tx_y.f[(deg - at + 1)..=deg].to_vec(),
                },
                &sk.modulo,
            );
            let tx_y_h = PolyZp::new(
                PolyZ {
                    f: tx_y.f[0..deg - at].to_vec(),
                },
                &sk.modulo,
            );

            let (tx_y_l_commit, (tx_y_l, _, r_l)) = self.dark.hiding_commit_zp(tx_y_l.clone())?;
            let (tx_y_h_commit, (tx_y_h, _, r_h)) = self.dark.hiding_commit_zp(tx_y_h.clone())?;

            witnesses.push((tx_y_l, r_l));
            witnesses.push((tx_y_h, r_h));

            ((tx_y_l_commit, tx_y_h_commit), witnesses)
        });

        let z = fiat.verifier_rand_below(&self.dark.p);
        let yz = y.clone() * &z;
        let query_points = vec![z.clone(), yz];

        let ((y_r, y_t, y_low, y_high), prover) = prover!(fiat, prover, (witnesses) => {
            let y = witnesses
                .iter()
                .map(|w| w.0.multi_evaluate_modulo(&query_points, &self.dark.p, w.0.degree()))
                .collect::<Vec<_>>();
            ((y[0].clone(), y[1].clone(), y[2].clone(), y[3].clone()), witnesses)
        });

        // t(z, y)
        let t_zy = y_t[0].clone();
        // r(z, 1)
        let r_z1 = y_r[0].clone();
        // r(z, y)
        let r_zy = y_r[1].clone();
        // s(z, y)
        let s_zy = sk.s.evaluate(&z, &y);
        // k(y)
        let k_y = sk.k.evaluate(&y);
        let p = &self.dark.p;
        let rhs = r_z1
            * (r_zy * y.clone().pow_mod(&Int::from(-2 * (n as i32)), p).unwrap()
                + s_zy * z.clone().pow_mod(&Int::from(2 * n), p).unwrap())
            - k_y * z.clone().pow_mod(&Int::from(4 * n), p).unwrap();
        check((t_zy.clone() - rhs).is_divisible(p), "t(X, Y) check failed")?;

        let lhs = y_low[0].clone()
            + y_high[0].clone() * z.clone().pow_mod(&Int::from(4 * n + 1), p).unwrap();
        check(
            (lhs - t_zy).is_divisible(p),
            "t(X, Y) coefficient check failed",
        )?;

        let bound: Int = (self.dark.p.clone() - 1) / 2;
        let instances = vec![
            dark::Instance {
                commitment: rx_1_commit,
                degree: sk.n * 3,
                bound: bound.clone(),
                y: y_r.clone(),
                z: query_points.clone(),
            },
            dark::Instance {
                commitment: tx_y_commit,
                degree: sk.n * 7,
                bound: bound.clone(),
                y: y_t.clone(),
                z: query_points.clone(),
            },
            dark::Instance {
                commitment: tx_y_l_commit,
                degree: sk.n * 4 - 1,
                bound: bound.clone(),
                y: y_low.clone(),
                z: query_points.clone(),
            },
            dark::Instance {
                commitment: tx_y_h_commit,
                degree: 3 * sk.n - 1,
                bound: bound.clone(),
                y: y_high.clone(),
                z: query_points.clone(),
            },
        ];
        let (prover, instance) = self.dark.combine(prover, instances, fiat)?;

        self.dark.multi_zk_eval(prover, instance, fiat)?;

        Ok(())
    }
}

fn get_tx_y(rx_1: &PolyZp, sk: &SK, n: usize, y: &Int) -> PolyZp {
    let rx_y = PolyZ::new_positive(get_rx_y(&rx_1, &y, n));
    assert!({
        let z = Int::from(3299218);
        let lhs = rx_y.evaluate_modulo(&z, &rx_1.p, rx_y.degree());
        let rhs = rx_1.evaluate(&(z * y))
            * y.clone()
                .pow_mod(&Int::from(-2 * (n as i32)), &rx_1.p)
                .unwrap();
        (lhs - rhs).is_divisible(&rx_1.p)
    });
    let sx_y = PolyZ::new_positive(get_sx_y(&sk.s, &y, n));
    assert!({
        let z = Int::from(1998776);
        let lhs = sk.s.evaluate(&z, &y) * z.clone().pow_mod(&Int::from(n), &rx_1.p).unwrap();
        let rhs = sx_y.evaluate_modulo(&z, &rx_1.p, sx_y.degree());
        (lhs - rhs).is_divisible(&rx_1.p)
    });
    let r1_x_y = shifted_add(rx_y.clone(), sx_y.clone(), n);
    assert!({
        let z = Int::from(-321812);
        let lhs = r1_x_y.evaluate_modulo(&z, &rx_1.p, r1_x_y.degree());
        let rhs = rx_y.evaluate_modulo(&z, &rx_1.p, rx_y.degree())
            + sx_y.evaluate_modulo(&z, &rx_1.p, rx_y.degree())
                * z.pow_mod(&Int::from(n), &rx_1.p).unwrap();
        (lhs - rhs).is_divisible(&rx_1.p)
    });
    let mut tx_y = PolyZ::new_positive(rx_1.clone()).multiply(&r1_x_y);
    let deg = tx_y.degree();
    let k_y = sk.k.evaluate(&y);
    tx_y.f[deg - 4 * n] -= &k_y;
    let tx_y = PolyZp::new(tx_y, &sk.modulo);
    assert!({
        let z = Int::from(-32148812);
        let t_zy = tx_y.evaluate(&z);
        let r_z1 = rx_1.evaluate(&z);
        let rr = r1_x_y.evaluate_modulo(&z, &rx_1.p, r1_x_y.degree());
        let rhs = r_z1 * rr - k_y * z.pow_mod(&Int::from(4 * n), &rx_1.p).unwrap();
        (t_zy - rhs).is_divisible(&rx_1.p)
    });
    tx_y
}

fn shifted_add(r: PolyZ, s: PolyZ, n: usize) -> PolyZ {
    let mut f = Vec::new();
    f.resize(4 * n + 1, Int::from(0));

    for (c_f, c_r) in f.iter_mut().rev().zip(r.f.iter().rev()) {
        c_f.assign(c_r);
    }
    for (c_f, c_s) in f.iter_mut().zip(s.f.iter()) {
        *c_f += c_s;
    }

    PolyZ { f }
}

/// r(X, 1) -> r(X, y)
fn get_rx_y(rx_1: &PolyZp, y: &Int, n: usize) -> PolyZp {
    let n = n as i32;
    let mut rx_y = rx_1.clone();
    let mut e = y.clone().pow_mod(&Int::from(-2 * n), &rx_1.p).unwrap();
    for c in rx_y.f.iter_mut().rev() {
        *c *= &e;
        *c %= &rx_1.p;
        e *= y;
        e %= &rx_1.p;
    }
    rx_y
}

/// s(X, Y) -> s(X, y)
fn get_sx_y(s: &SparseBiPolyZp, y: &Int, n: usize) -> PolyZp {
    let p = s.p.clone();
    let mut f = Vec::new();
    f.resize(3 * n + 1, Int::from(0));

    // let max_i = s.coeff.iter().map(|(_, i, _)| *i).max().unwrap();
    for (v, i, j) in &s.coeff {
        let i = (*i + n as i32) as usize;
        f[i] += y.clone().pow_mod(&Int::from(*j), &p).unwrap() * v;
    }

    f.reverse();
    PolyZp::new(PolyZ { f }, &p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::*;
    use common::{assert, assert_eq};
    use rand::Rng;

    #[test]
    fn test_tx_y_powmod() {
        let mut rng = rand::thread_rng();
        // g^x = y mod p
        let p = PierreGenPrime::new(512, 256).gen(&mut rng);
        let g = Int::from(2020);
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

        let circuit = crate::modulo::exp(64);
        let input = std::iter::once(g.clone())
            .chain(x_bits.into_iter())
            .collect::<Vec<_>>();
        let output = crate::circuit::evaluate(&circuit, &input, &p);

        assert_eq!(&output[1], &g);
        assert!((g.clone().pow_mod(&Int::from(x), &p).unwrap() - &output[0]).is_divisible(&p));

        let linear_circuit = crate::linear_circuit::convert(circuit);
        let left_input = input;
        let right_input = output;
        run_test_tx_y(&linear_circuit, left_input, right_input, p);
    }

    #[test]
    fn test_tx_y_sha256() {
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
            assert_eq!(r.len(), 512);

            r
        };

        let modulo = Int::from(13441);
        let c = crate::uint32::sha256();
        let c0 = crate::circuit::convert(c);
        let c1 = crate::linear_circuit::convert(c0.clone());
        let output = crate::circuit::evaluate(&c0, &input, &modulo);

        let left_input = input;
        let right_input = output;
        run_test_tx_y(&c1, left_input, right_input, modulo);
    }

    fn run_test_tx_y(
        linear_circuit: &crate::linear_circuit::Circuit,
        left_input: Vec<Int>,
        right_input: Vec<Int>,
        p: Int,
    ) {
        let n = linear_circuit.n();
        let abc = linear_circuit.evaluate(&left_input, &right_input, &p);
        assert!(linear_circuit
            .output
            .iter()
            .map(|v| &abc[linear_circuit.out(*v).get() - 1].c)
            .all(|i| i == &0));
        let uvwk = linear_circuit.to_constrains(&left_input, &p);
        crate::sonic::is_sastified(&abc, &uvwk, &p).unwrap();
        let rx_y = to_rx_y(&abc, &p, n);
        let rx_1 = to_rx_1(abc, &p);
        {
            let x = Int::from(3212);
            let a = rx_1.evaluate(&x) * x.clone().pow_mod(&Int::from(-2 * (n as i32)), &p).unwrap();
            let b = rx_y.evaluate(&x, &Int::from(1));
            assert!((a - b).is_divisible(&p));
        }

        let sk = to_sk(&uvwk, n, &p);
        let y = Int::from(123459);
        let tx_y = get_tx_y(&rx_1, &sk, n, &y);
        {
            let x = Int::from(912);
            let a = tx_y.evaluate(&x) * x.clone().pow_mod(&Int::from(-4 * (n as i32)), &p).unwrap();
            let b = rx_y.evaluate(&x, &Int::from(1))
                * (rx_y.evaluate(&x, &y) + sk.s.evaluate(&x, &y))
                - sk.k.evaluate(&y);
            assert!((a - b).is_divisible(&p));
        }
        dbg!(tx_y.degree());
        assert_eq!(&tx_y.f[tx_y.degree() - 4 * n], &0);

        let z = Int::from(-998723);
        let t_zy = tx_y.evaluate(&z);
        let s_zy = sk.s.evaluate(&z, &y);
        let k_y = sk.k.evaluate(&y);
        let r_z1 = rx_1.evaluate(&z);
        let r_zy = rx_1.evaluate(&(z.clone() * &y));

        let rhs = r_z1
            * (r_zy * y.clone().pow_mod(&Int::from(-2 * (n as i32)), &p).unwrap()
                + s_zy * z.clone().pow_mod(&Int::from(2 * n), &p).unwrap())
            - k_y * z.clone().pow_mod(&Int::from(4 * n), &p).unwrap();
        assert!((t_zy - rhs).is_divisible(&p));
    }

    fn setup_sonic(max_deg: usize) -> Sonic {
        let key_path = format!("../keys/{}.json", max_deg);
        let dark = {
            match DARK::<RSAGroup>::from_key(&key_path, true) {
                Ok(dark) => dark,
                Err(_) => {
                    let dark = DARK::<RSAGroup>::setup(max_deg);
                    dark.save_key(&key_path, false).unwrap();
                    dark
                }
            }
        };
        Sonic { dark }
    }

    #[test]
    #[ignore]
    fn test_sonic_powmod() {
        use common::{FiatShamirRng, UniformRandom};

        let size = 8;
        let mut sonic = setup_sonic(1000);
        let mut rng = rand::thread_rng();
        // g^x = y mod p
        let p = sonic.dark.p.clone();
        let mut g = Int::from(0);
        UniformRandom::new().rand_below(&p, &mut rng, &mut g);
        let (x, x_bits) = {
            let mut x = rng.gen::<u64>();
            let temp = x;
            let mut x_bits = Vec::new();
            for _ in 0..size {
                x_bits.push(Int::from(x & 1));
                x >>= 1;
            }
            x_bits.reverse();
            (temp & ((1 << size) - 1), x_bits)
        };

        let circuit = crate::modulo::exp(size);
        let input = std::iter::once(g.clone())
            .chain(x_bits.into_iter())
            .collect::<Vec<_>>();
        let output = crate::circuit::evaluate(&circuit, &input, &p);

        assert_eq!(&output[1], &g);
        assert!((g.clone().pow_mod(&Int::from(x), &p).unwrap() - &output[0]).is_divisible(&p));

        let linear_circuit = crate::linear_circuit::convert(circuit);
        let left_input = input;
        let right_input = output;
        run_test_tx_y(
            &linear_circuit,
            left_input.clone(),
            right_input.clone(),
            p.clone(),
        );

        let uvwk = linear_circuit.to_constrains(&left_input, &p);
        let n = linear_circuit.n();
        let sk = to_sk(&uvwk, n, &p);

        let abc = linear_circuit.evaluate(&left_input, &right_input, &p);
        let rx_1 = to_rx_1(abc, &p);

        let mut fiat = FiatShamirRng::new("asdf", PierreGenPrime::new(128, 64));
        sonic.prove(Prover::Witness(rx_1), &sk, &mut fiat).unwrap();

        let proof = fiat.proofs;
        let mut fiat = FiatShamirRng::new("asdf", PierreGenPrime::new(128, 64));
        sonic.prove(Prover::Proof(proof), &sk, &mut fiat).unwrap();
    }
}

fn check<S: std::string::ToString>(ok: bool, err_str: S) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(err_str.to_string())
    }
}
