use crate::circuit::Vertex;

pub fn exp(bit_size: usize) -> Vec<Vertex> {
    let input = Vertex::input(0, false);
    let one = Vertex::constant(1);

    // most significant bit first
    let exponent = (1..=bit_size)
        .map(|i| Vertex::input(i, true))
        .collect::<Vec<_>>();

    let mut result = one.clone();
    let mut e = input.clone();
    for b in exponent.iter().rev() {
        result = select(
            result.clone(),
            result.clone() * e.clone(),
            b.clone(),
            one.clone(),
        );
        e = e.clone() * e.clone();
    }
    vec![result, input]
}

pub fn select(x: Vertex, y: Vertex, b: Vertex, one: Vertex) -> Vertex {
    (one - b.clone()) * x + b.clone() * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Int, PierreGenPrime};
    use rand::Rng;

    #[test]
    fn test_exp() {
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

        let circuit = exp(64);
        let input = std::iter::once(g.clone())
            .chain(x_bits.into_iter())
            .collect::<Vec<_>>();
        let output = crate::circuit::evaluate(&circuit, &input, &p);
        assert_eq!(&output[1], &g);
        assert!((g.clone().pow_mod(&Int::from(x), &p).unwrap() - &output[0]).is_divisible(&p));
    }
}
