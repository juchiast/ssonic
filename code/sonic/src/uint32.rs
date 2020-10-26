use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;
use poly_commit::{assert, assert_eq};

pub enum VertexType {
    Input(usize),
    Constant(u32),
    RightRotate(Vertex, usize),
    Xor(Vertex, Vertex),
    ShiftRight(Vertex, usize),
    Add(Vertex, Vertex),
    Not(Vertex),
    And(Vertex, Vertex),
}

#[derive(Clone)]
pub struct Vertex {
    pub inner: Rc<VertexType>,
}

impl Eq for Vertex {}

impl PartialEq for Vertex {
    fn eq(&self, other: &Vertex) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::into_raw(self.inner.clone()) as usize).hash(state);
    }
}

impl Vertex {
    fn zero() -> Self {
        Self {
            inner: Rc::new(VertexType::Constant(0)),
        }
    }

    fn input(index: usize) -> Self {
        Self {
            inner: Rc::new(VertexType::Input(index)),
        }
    }

    fn constant(x: u32) -> Self {
        Self {
            inner: Rc::new(VertexType::Constant(x)),
        }
    }

    fn right_rotate(&self, rhs: usize) -> Self {
        Self {
            inner: Rc::new(VertexType::RightRotate(self.clone(), rhs)),
        }
    }
}

pub fn collect_vertices(mut vertices: Vec<Vertex>) -> Vec<Vertex> {
    let mut set = HashSet::new();
    let mut result = Vec::new();
    while !vertices.is_empty() {
        let queue = std::mem::take(&mut vertices);

        for v in queue {
            if !set.contains(&v) {
                match v.inner.as_ref() {
                    VertexType::Not(a)
                    | VertexType::RightRotate(a, _)
                    | VertexType::ShiftRight(a, _) => vertices.push(a.clone()),
                    VertexType::Add(a, b) | VertexType::And(a, b) | VertexType::Xor(a, b) => {
                        vertices.push(a.clone());
                        vertices.push(b.clone());
                    }
                    VertexType::Constant(_) | VertexType::Input(_) => {}
                }
                set.insert(v.clone());
                result.push(v);
            }
        }
    }
    result.reverse();
    result
}

impl std::ops::BitXor<Vertex> for Vertex {
    type Output = Vertex;

    fn bitxor(self, rhs: Vertex) -> Self {
        Self {
            inner: Rc::new(VertexType::Xor(self, rhs)),
        }
    }
}

impl std::ops::Shr<usize> for Vertex {
    type Output = Vertex;
    fn shr(self, rhs: usize) -> Vertex {
        Self {
            inner: Rc::new(VertexType::ShiftRight(self, rhs)),
        }
    }
}

impl std::ops::Add<Vertex> for Vertex {
    type Output = Vertex;
    fn add(self, rhs: Vertex) -> Vertex {
        Self {
            inner: Rc::new(VertexType::Add(self, rhs)),
        }
    }
}

impl std::ops::Not for Vertex {
    type Output = Vertex;
    fn not(self) -> Vertex {
        Self {
            inner: Rc::new(VertexType::Not(self)),
        }
    }
}

impl std::ops::BitAnd<Vertex> for Vertex {
    type Output = Vertex;
    fn bitand(self, rhs: Vertex) -> Vertex {
        Self {
            inner: Rc::new(VertexType::And(self, rhs)),
        }
    }
}

pub fn sha256_what(input: &[u32]) -> Vec<u32> {
    let k: Vec<u32> = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ]
    .to_vec();

    let mut r: Vec<u32> = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]
    .to_vec();

    let mut w: Vec<u32> = Vec::new();
    w.resize_with(64, || 0);

    for i in 0..16 {
        w[i] = input[i].clone();
    }

    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15].clone() >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2].clone() >> 10);
        w[i] = w[i - 16]
            .clone()
            .wrapping_add(s0)
            .wrapping_add(w[i - 7].clone())
            .wrapping_add(s1);
    }

    let mut a = r[0].clone();
    let mut b = r[1].clone();
    let mut c = r[2].clone();
    let mut d = r[3].clone();
    let mut e = r[4].clone();
    let mut f = r[5].clone();
    let mut g = r[6].clone();
    let mut h = r[7].clone();

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e.clone() & f.clone()) ^ (!e.clone() & g.clone());
        let temp1 = h
            .clone()
            .wrapping_add(s1.clone())
            .wrapping_add(ch.clone())
            .wrapping_add(k[i].clone())
            .wrapping_add(w[i].clone());
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a.clone() & b.clone()) ^ (a.clone() & c.clone()) ^ (b.clone() & c.clone());
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1.clone());
        d = c;
        c = b;
        b = a;
        a = temp1.clone().wrapping_add(temp2.clone());
    }

    r[0] = r[0].clone().wrapping_add(a);
    r[1] = r[1].clone().wrapping_add(b);
    r[2] = r[2].clone().wrapping_add(c);
    r[3] = r[3].clone().wrapping_add(d);
    r[4] = r[4].clone().wrapping_add(e);
    r[5] = r[5].clone().wrapping_add(f);
    r[6] = r[6].clone().wrapping_add(g);
    r[7] = r[7].clone().wrapping_add(h);

    r
}

pub fn sha256() -> Vec<Vertex> {
    let k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ]
    .iter()
    .map(|x| Vertex::constant(*x))
    .collect::<Vec<_>>();

    let mut r = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]
    .iter()
    .map(|x| Vertex::constant(*x))
    .collect::<Vec<_>>();

    let input = (0..16).map(|i| Vertex::input(i)).collect::<Vec<_>>();

    let mut w: Vec<Vertex> = Vec::new();
    w.resize_with(64, || Vertex::zero());

    for i in 0..16 {
        w[i] = input[i].clone();
    }

    for i in 16..64 {
        let s0 = w[i - 15].right_rotate(7) ^ w[i - 15].right_rotate(18) ^ (w[i - 15].clone() >> 3);
        let s1 = w[i - 2].right_rotate(17) ^ w[i - 2].right_rotate(19) ^ (w[i - 2].clone() >> 10);
        w[i] = w[i - 16].clone() + s0 + w[i - 7].clone() + s1;
    }

    let mut a = r[0].clone();
    let mut b = r[1].clone();
    let mut c = r[2].clone();
    let mut d = r[3].clone();
    let mut e = r[4].clone();
    let mut f = r[5].clone();
    let mut g = r[6].clone();
    let mut h = r[7].clone();

    for i in 0..64 {
        let s1 = e.right_rotate(6) ^ e.right_rotate(11) ^ e.right_rotate(25);
        let ch = (e.clone() & f.clone()) ^ (!e.clone() & g.clone());
        let temp1 = h.clone() + s1.clone() + ch.clone() + k[i].clone() + w[i].clone();
        let s0 = a.right_rotate(2) ^ a.right_rotate(13) ^ a.right_rotate(22);
        let maj = (a.clone() & b.clone()) ^ (a.clone() & c.clone()) ^ (b.clone() & c.clone());
        let temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1.clone();
        d = c;
        c = b;
        b = a;
        a = temp1.clone() + temp2.clone();
    }

    r[0] = r[0].clone() + a;
    r[1] = r[1].clone() + b;
    r[2] = r[2].clone() + c;
    r[3] = r[3].clone() + d;
    r[4] = r[4].clone() + e;
    r[5] = r[5].clone() + f;
    r[6] = r[6].clone() + g;
    r[7] = r[7].clone() + h;

    r
}

pub fn evaluate(circuit: &[Vertex], input: &[u32]) -> Vec<u32> {
    let mut cache: HashMap<Vertex, u32> = HashMap::new();
    circuit
        .iter()
        .map(|v| evaluate_single(v, input, &mut cache))
        .collect()
}

fn evaluate_single(v: &Vertex, input: &[u32], cache: &mut HashMap<Vertex, u32>) -> u32 {
    if let Some(value) = cache.get(v) {
        return *value;
    }

    let result = match v.inner.as_ref() {
        VertexType::Constant(u) => *u,
        VertexType::Input(i) => input[*i],
        VertexType::Not(a) => !evaluate_single(a, input, cache),
        VertexType::Add(a, b) => {
            evaluate_single(a, input, cache).wrapping_add(evaluate_single(b, input, cache))
        }
        VertexType::And(a, b) => {
            evaluate_single(a, input, cache) & evaluate_single(b, input, cache)
        }
        VertexType::Xor(a, b) => {
            evaluate_single(a, input, cache) ^ evaluate_single(b, input, cache)
        }
        VertexType::RightRotate(a, i) => evaluate_single(a, input, cache).rotate_right(*i as u32),
        VertexType::ShiftRight(a, i) => evaluate_single(a, input, cache) >> *i,
    };
    cache.insert(v.clone(), result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::digest::Digest;
    use std::convert::TryInto;
    use std::{assert, assert_eq};

    #[test]
    fn test_sha256() {
        let unpadded_bytes = [97u8; 55];

        let mut bytes: [u8; 64] = [0; 64];
        bytes[..55].copy_from_slice(&unpadded_bytes);
        bytes[55] = 0b10000000;
        bytes[56..].copy_from_slice(&((unpadded_bytes.len() * 8) as u64).to_be_bytes());

        let correct = {
            let mut hasher = sha2::Sha256::new();
            hasher.update(unpadded_bytes.as_ref());
            hasher.finalize()
        };

        let mut input = [0u32; 16];
        for i in 0..16 {
            input[i] = u32::from_be_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        let circuit = sha256();
        let result = evaluate(&circuit, &input);
        let mut result_bytes = Vec::new();
        for u in result {
            result_bytes.extend_from_slice(&u.to_be_bytes());
        }

        assert_eq!(result_bytes.as_slice(), correct.as_slice());
    }
}
