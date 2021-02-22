#![allow(non_snake_case)]
/// Proof of Exponentation
use common::{prover, traits::Group, FiatShamirRng, Int, Prover};

/// Statement: u^x = w
pub fn proof_of_exponentation<G: Group, W>(
    prover: Prover<W>,
    group: &G,
    u: &G::Element,
    w: &G::Element,
    x: &Int,
    fiat: &mut FiatShamirRng,
) -> Result<Prover<W>, String> {
    let l = fiat.verifier_rand_prime();
    let (q, r) = x.clone().div_rem_euc(l.clone());

    let (Q, prover) = prover!(prover, (w) => {
        let Q = group.power(u, &q);
        (Q, (w))
    });
    fiat.prover_send(&Q);

    let Q_l = group.power(&Q, &l);
    let u_r = group.power(u, &r);
    let Q_l_u_r = group.mul(&Q_l, &u_r);
    if group.equal(w, &Q_l_u_r) {
        Ok(prover)
    } else {
        Err("PoE check failed".to_owned())
    }
}
