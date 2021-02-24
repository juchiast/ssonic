use crate::uint32;
use common::Int;
use std::collections::HashMap;
use std::rc::Rc;

pub enum VertexType {
    /// true => input in {0, 1}
    Input(usize, bool),
    Constant(Int),
    ConstantInverse(Int),
    Add(Vertex, Vertex),
    Substract(Vertex, Vertex),
    Multiply(Vertex, Vertex),
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

impl std::ops::Mul<Vertex> for Vertex {
    type Output = Vertex;
    fn mul(self, rhs: Vertex) -> Vertex {
        Vertex {
            inner: Rc::new(VertexType::Multiply(self, rhs)),
        }
    }
}

impl std::ops::Sub<Vertex> for Vertex {
    type Output = Vertex;
    fn sub(self, rhs: Vertex) -> Vertex {
        Vertex {
            inner: Rc::new(VertexType::Substract(self, rhs)),
        }
    }
}

impl std::ops::Add<Vertex> for Vertex {
    type Output = Vertex;
    fn add(self, rhs: Vertex) -> Vertex {
        Vertex {
            inner: Rc::new(VertexType::Add(self, rhs)),
        }
    }
}

impl Vertex {
    pub fn input(index: usize, is_bool: bool) -> Self {
        Self {
            inner: Rc::new(VertexType::Input(index, is_bool)),
        }
    }

    pub fn constant(x: usize) -> Self {
        Self {
            inner: Rc::new(VertexType::Constant(Int::from(x))),
        }
    }
}

pub fn convert(a: Vec<uint32::Vertex>) -> Vec<Vertex> {
    // [0, 1, 2, 2^-1]
    let consts: Vec<Vertex> = (0..=2)
        .map(Vertex::constant)
        .chain(std::iter::once(Vertex {
            inner: Rc::new(VertexType::ConstantInverse(Int::from(2))),
        }))
        .collect();
    let mut cache: HashMap<uint32::Vertex, Rc<[Vertex]>> = HashMap::new();
    a.into_iter()
        .map(|v| convert_single(&v, &consts, &mut cache).to_vec())
        .flatten()
        .collect()
}

fn convert_single(
    u: &uint32::Vertex,
    consts: &[Vertex],
    cache: &mut HashMap<uint32::Vertex, Rc<[Vertex]>>,
) -> Rc<[Vertex]> {
    if let Some(vertices) = cache.get(u) {
        return vertices.clone();
    }

    use uint32::VertexType as Type;
    let result = match u.inner.as_ref() {
        Type::Input(index) => (0..32)
            .map(|i| Vertex::input(i + 32 * index, true))
            .collect(),
        Type::Constant(x) => convert_constant(*x, consts),
        Type::And(a, b) => and_gate(
            &convert_single(a, consts, cache),
            &convert_single(b, consts, cache),
        ),
        Type::Xor(a, b) => xor_gate(
            &convert_single(a, consts, cache),
            &convert_single(b, consts, cache),
        ),
        Type::Not(a) => not_gate(&convert_single(a, consts, cache), consts),
        Type::ShiftRight(a, i) => right_shift_gate(&convert_single(a, consts, cache), *i, consts),
        Type::RightRotate(a, i) => right_rotate_gate(&convert_single(a, consts, cache), *i),
        Type::Add(a, b) => add_gate(
            &convert_single(a, consts, cache),
            &convert_single(b, consts, cache),
            consts,
        ),
    };
    cache.insert(u.clone(), result.clone());
    result
}

fn convert_constant(mut x: u32, consts: &[Vertex]) -> Rc<[Vertex]> {
    let mut result = Vec::new();
    result.reserve_exact(32);
    for _ in 0..32 {
        result.push(consts[(x & 1) as usize].clone());
        x >>= 1;
    }
    result.into()
}

fn right_shift_gate(a: &[Vertex], i: usize, consts: &[Vertex]) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    a.iter()
        .skip(i)
        .cloned()
        .chain(std::iter::repeat(consts[0].clone()).take(i))
        .collect()
}

fn right_rotate_gate(a: &[Vertex], i: usize) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    a.iter().skip(i).chain(a.iter().take(i)).cloned().collect()
}

fn and_gate(a: &[Vertex], b: &[Vertex]) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    assert_eq!(b.len(), 32);
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() * b.clone())
        .collect()
}

fn xor_gate(a: &[Vertex], b: &[Vertex]) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    assert_eq!(b.len(), 32);
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let temp = a.clone() - b.clone();
            temp.clone() * temp
        })
        .collect()
}

fn not_gate(a: &[Vertex], consts: &[Vertex]) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    a.iter().map(|x| consts[1].clone() - x.clone()).collect()
}

fn add_1_bit_gate(x: &Vertex, y: &Vertex, c: &Vertex, consts: &[Vertex]) -> (Vertex, Vertex) {
    let xy = x.clone() + y.clone();
    let b = xy.clone() * (xy.clone() - consts[1].clone()) * consts[3].clone();
    let xy_c = xy.clone() * c.clone();
    let c_out = xy_c * (consts[1].clone() - b.clone()) + b;
    let s = xy + c.clone() - consts[2].clone() * c_out.clone();
    (s, c_out)
}

fn add_gate(a: &[Vertex], b: &[Vertex], consts: &[Vertex]) -> Rc<[Vertex]> {
    assert_eq!(a.len(), 32);
    assert_eq!(b.len(), 32);
    let mut carry = consts[0].clone();
    let mut result = Vec::new();
    result.reserve_exact(32);
    for (x, y) in a.iter().zip(b.iter()) {
        let (s, c_out) = add_1_bit_gate(x, y, &carry, consts);
        result.push(s);
        carry = c_out;
    }
    result.into()
}

pub fn evaluate(circuit: &[Vertex], input: &[Int], modulo: &Int) -> Vec<Int> {
    let mut cache: HashMap<Vertex, Int> = HashMap::new();
    circuit
        .iter()
        .map(|v| evaluate_single(v, input, modulo, &mut cache))
        .collect()
}

fn evaluate_single(
    v: &Vertex,
    input: &[Int],
    modulo: &Int,
    cache: &mut HashMap<Vertex, Int>,
) -> Int {
    if let Some(value) = cache.get(v) {
        return value.clone();
    }

    let result = match v.inner.as_ref() {
        VertexType::Input(i, is_bool) => {
            let r = input[*i].clone();
            assert!(!is_bool || r == 0 || r == 1);
            r
        }
        VertexType::Constant(u) => u.clone(),
        VertexType::ConstantInverse(u) => u.clone().pow_mod(&Int::from(-1), modulo).unwrap(),
        VertexType::Add(a, b) => {
            evaluate_single(a, input, modulo, cache) + evaluate_single(b, input, modulo, cache)
        }
        VertexType::Multiply(a, b) => {
            evaluate_single(a, input, modulo, cache) * evaluate_single(b, input, modulo, cache)
        }
        VertexType::Substract(a, b) => {
            evaluate_single(a, input, modulo, cache) - evaluate_single(b, input, modulo, cache)
        }
    } % modulo;
    cache.insert(v.clone(), result.clone());
    result
}

mod toposort;
pub use toposort::toposort;

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn test_convert() {
        let unpadded_bytes = [97u8; 55];

        let mut bytes: [u8; 64] = [0; 64];
        bytes[..55].copy_from_slice(&unpadded_bytes);
        bytes[55] = 0b10000000;
        bytes[56..].copy_from_slice(&((unpadded_bytes.len() * 8) as u64).to_be_bytes());

        let mut input = [0u32; 16];
        for i in 0..16 {
            input[i] = u32::from_be_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        let circuit = crate::uint32::sha256();
        let correct = {
            let result = crate::uint32::evaluate(&circuit, &input);
            let mut result_bits = Vec::new();
            for mut u in result {
                for _ in 0..32 {
                    result_bits.push((u & 1) as i32);
                    u >>= 1;
                }
            }
            result_bits
        };

        let bit_circuit = convert(circuit);
        let mut bit_input = Vec::new();
        for u in input.iter() {
            let mut u = *u;
            for _ in 0..32 {
                bit_input.push(Int::from(u & 1));
                u >>= 1;
            }
        }

        let result = evaluate(&bit_circuit, &bit_input, &Int::from(13441))
            .iter()
            .map(|u| u.to_i32().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(result, correct);
        println!("{:?}", result);
    }
}
