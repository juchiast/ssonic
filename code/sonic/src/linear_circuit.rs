//! Abstract toposort-ed arithmetic circuit

use crate::*;
use common::Int;
use common::{assert, assert_eq};
use rug::Assign;
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;

pub enum VertexType {
    Input,
    Constant(Int),
    ConstantInverse(Int),
    Add(Vertex, Vertex),
    Substract(Vertex, Vertex),
    Multiply(Vertex, Vertex),
}

pub type Vertex = std::num::NonZeroU32;

pub struct Circuit {
    pub vertices: Vec<VertexType>,
    /// outs[v] is i such that c_i is output of v
    pub outs: Vec<NonZeroUsize>,
    pub left_input: Vec<Vertex>,
    pub right_input: Vec<Vertex>,
    pub output: Vec<Vertex>,
}

impl Circuit {
    pub fn n(&self) -> usize {
        self.vertices
            .iter()
            .map(|v| match v {
                VertexType::Input
                | VertexType::Constant(_)
                | VertexType::ConstantInverse(_)
                | VertexType::Multiply(_, _) => 1,
                VertexType::Add(_, _) | VertexType::Substract(_, _) => 2,
            })
            .sum()
    }

    fn iter_vertices(&self) -> impl Iterator<Item = (Vertex, &'_ VertexType)> {
        self.vertices
            .iter()
            .enumerate()
            .map(|(i, t)| (Vertex::new(i as u32 + 1).unwrap(), t))
    }

    pub fn evaluate(&self, left_input: &[Int], right_input: &[Int], modulo: &Int) -> Vec<ABC> {
        assert_eq!(left_input.len(), self.left_input.len());
        assert_eq!(right_input.len(), self.right_input.len());
        fn to_index(v: Vertex) -> usize {
            v.get() as usize - 1
        }
        let get_out = |v: &Vertex| -> usize { self.out(*v).get() - 1 };
        let mut result = Vec::new();
        let n = self.n();
        result.resize(n, ABC::new());
        let mut assigned = Vec::new();
        assigned.resize(n, false);
        for (&i, value) in self
            .left_input
            .iter()
            .zip(left_input.iter())
            .chain(self.right_input.iter().zip(right_input.iter()))
        {
            let abc = result.get_mut(to_index(i)).unwrap();
            abc.a.assign(1);
            abc.b.assign(value);
            abc.c.assign(value);
            assigned[to_index(i)] = true;
        }
        for (i, kind) in self.iter_vertices() {
            match kind {
                VertexType::Input => {}
                VertexType::Constant(value) => {
                    let abc = result.get_mut(to_index(i)).unwrap();
                    abc.a.assign(1);
                    abc.b.assign(value);
                    abc.c.assign(value);
                    assigned[to_index(i)] = true;
                }
                VertexType::ConstantInverse(value) => {
                    let value = value
                        .clone()
                        .pow_mod(&Int::from(modulo - 2), modulo)
                        .unwrap();
                    let abc = result.get_mut(to_index(i)).unwrap();
                    abc.a.assign(1);
                    abc.b.assign(&value);
                    abc.c.assign(&value);
                    assigned[to_index(i)] = true;
                }
                VertexType::Multiply(a, b) => {
                    assert!(*a < i && *b < i);
                    let value_a = result[get_out(a)].c.clone();
                    let value_b = result[get_out(b)].c.clone();
                    let abc = result.get_mut(to_index(i)).unwrap();
                    abc.c = value_a.clone() * &value_b % modulo;
                    abc.a = value_a;
                    abc.b = value_b;
                    assigned[to_index(i)] = true;
                }
                VertexType::Add(a, b) => {
                    assert!(*a < i && *b < i);
                    let value_a = result[get_out(a)].c.clone();
                    let value_b = result[get_out(b)].c.clone();

                    let abc = result.get_mut(to_index(i)).unwrap();
                    abc.c = value_a.clone() * &value_b % modulo;
                    abc.a.assign(&value_a);
                    abc.b.assign(&value_b);
                    assigned[to_index(i)] = true;

                    let o = get_out(&i);

                    let abc = result.get_mut(o).unwrap();
                    abc.c = (value_a.clone() + &value_b) % modulo;
                    abc.a.assign(1);
                    abc.b.assign(&abc.c);
                    assigned[o] = true;
                }
                VertexType::Substract(a, b) => {
                    assert!(*a < i && *b < i);
                    let value_a = result[get_out(a)].c.clone();
                    let value_b = result[get_out(b)].c.clone();

                    let abc = result.get_mut(to_index(i)).unwrap();
                    abc.c = value_a.clone() * &value_b % modulo;
                    abc.a.assign(&value_a);
                    abc.b.assign(&value_b);
                    assigned[to_index(i)] = true;

                    let o = get_out(&i);
                    let abc = result.get_mut(o).unwrap();
                    abc.c = (value_a.clone() - &value_b) % modulo;
                    if abc.c < 0 {
                        abc.c += modulo;
                    }
                    abc.a.assign(1);
                    abc.b.assign(&abc.c);
                    assigned[o] = true;
                }
            }
        }

        #[cfg(any(not(feature = "no-assert"), test))]
        {
            for i in self.output.iter() {
                assert_eq!(&result[get_out(i)].c, &0);
            }

            for abc in result.iter() {
                assert!((abc.a.clone() * &abc.b - &abc.c).is_divisible(modulo));
                assert!(0 <= abc.a && &abc.a < modulo);
                assert!(0 <= abc.b && &abc.b < modulo);
                assert!(0 <= abc.c && &abc.c < modulo);
            }
        }

        assert!(assigned.into_iter().all(|b| b));

        result
    }

    pub fn to_constrains(&self, left_input: &[Int], modulo: &Int) -> Vec<UVWK> {
        let mut result = Vec::new();

        // left input constraints
        for (vertex, value) in self.left_input.iter().zip(left_input.iter()) {
            result.push(UVWK {
                u: None,
                v: None,
                w: Some((1.into(), self.out(*vertex))),
                k: value.clone(),
            });
        }

        // output = 0
        for vertex in self.output.iter() {
            result.push(UVWK {
                u: None,
                v: None,
                w: Some((1.into(), self.out(*vertex))),
                k: 0.into(),
            });
        }

        for (vertex, vertex_type) in self.iter_vertices() {
            let vertex = NonZeroUsize::new(vertex.get() as usize).unwrap();
            match vertex_type {
                VertexType::Add(a, b)
                | VertexType::Substract(a, b)
                | VertexType::Multiply(a, b) => {
                    // left edge (a -> vertex)
                    // left_inp(vertex) - out(a) = 0
                    result.push(UVWK {
                        u: Some((1.into(), vertex)),
                        v: None,
                        w: Some((Int::from(-1), self.out(*a))),
                        k: 0.into(),
                    });

                    // right edge (b -> vertex)
                    // right_inp(vertex) - out(b) = 0
                    result.push(UVWK {
                        u: None,
                        v: Some((1.into(), vertex)),
                        w: Some((Int::from(-1), self.out(*b))),
                        k: 0.into(),
                    });
                }
                _ => {}
            }
        }

        for (vertex, vertex_type) in self.iter_vertices() {
            match vertex_type {
                VertexType::Constant(value) => {
                    result.push(UVWK {
                        u: None,
                        v: None,
                        w: Some((1.into(), self.out(vertex))),
                        k: value.clone(),
                    });
                }
                VertexType::ConstantInverse(value) => {
                    let value = value.clone().invert(modulo).unwrap();
                    result.push(UVWK {
                        u: None,
                        v: None,
                        w: Some((1.into(), self.out(vertex))),
                        k: value,
                    });
                }
                VertexType::Add(_, _) => {
                    result.push(UVWK {
                        u: Some((1.into(), NonZeroUsize::new(vertex.get() as usize).unwrap())),
                        v: Some((1.into(), NonZeroUsize::new(vertex.get() as usize).unwrap())),
                        w: Some((Int::from(-1), self.out(vertex))),
                        k: 0.into(),
                    });
                }
                VertexType::Substract(_, _) => {
                    result.push(UVWK {
                        u: Some((1.into(), NonZeroUsize::new(vertex.get() as usize).unwrap())),
                        v: Some((
                            Int::from(-1),
                            NonZeroUsize::new(vertex.get() as usize).unwrap(),
                        )),
                        w: Some((Int::from(-1), self.out(vertex))),
                        k: 0.into(),
                    });
                }
                VertexType::Input | VertexType::Multiply(_, _) => {}
            }
        }

        result
    }

    pub fn out(&self, v: Vertex) -> NonZeroUsize {
        self.outs[v.get() as usize - 1]
    }
}

/// C -> C'
/// C(w) = x -> C'(w, x) = 0
pub fn convert(vertices: Vec<crate::circuit::Vertex>) -> Circuit {
    let output_vertices = vertices;
    let all_vertices = crate::circuit::toposort(output_vertices.clone());

    let index_map: HashMap<circuit::Vertex, Vertex> = all_vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), Vertex::new(i as u32 + 1).unwrap()))
        .collect();

    let mut left_input: BTreeMap<usize, Vertex> = BTreeMap::new();

    #[cfg(any(not(feature = "no-assert"), test))]
    let vertices_count = all_vertices.len();
    let mut bools = Vec::new();
    let mut vertices = all_vertices
        .into_iter()
        .enumerate()
        .map(|(i, v)| match v.inner.as_ref() {
            circuit::VertexType::Add(a, b) => VertexType::Add(index_map[a], index_map[b]),
            circuit::VertexType::Substract(a, b) => {
                VertexType::Substract(index_map[a], index_map[b])
            }
            circuit::VertexType::Multiply(a, b) => VertexType::Multiply(index_map[a], index_map[b]),
            circuit::VertexType::Constant(x) => VertexType::Constant(x.clone()),
            circuit::VertexType::ConstantInverse(x) => VertexType::ConstantInverse(x.clone()),
            circuit::VertexType::Input(index, is_bool) => {
                let vertex = Vertex::new(i as u32 + 1).unwrap();
                left_input.insert(*index, vertex);
                if *is_bool {
                    bools.push(vertex);
                }
                VertexType::Input
            }
        })
        .collect::<Vec<_>>();

    let left_input = left_input
        .into_iter()
        .map(|(_key, value)| value)
        .collect::<Vec<_>>();

    let mut right_input = Vec::new();
    let mut output = Vec::new();
    #[cfg(any(not(feature = "no-assert"), test))]
    let output_count = output_vertices.len();
    for v in output_vertices {
        vertices.push(VertexType::Input);
        let ri = Vertex::new(vertices.len() as u32).unwrap();
        right_input.push(ri);

        vertices.push(VertexType::Substract(ri, index_map[&v]));
        output.push(Vertex::new(vertices.len() as u32).unwrap());
    }

    for &v in &bools {
        vertices.push(VertexType::Multiply(v, v));
        let temp_vertex = Vertex::new(vertices.len() as u32).unwrap();
        vertices.push(VertexType::Substract(temp_vertex, v));
        output.push(Vertex::new(vertices.len() as u32).unwrap());
    }

    let mut outs = Vec::new();
    let mut count = 0;
    for (i, v) in vertices.iter().enumerate() {
        let temp = match v {
            VertexType::Add(_, _) | VertexType::Substract(_, _) => {
                let r = vertices.len() + count;
                count += 1;
                r
            }
            _ => i,
        };
        outs.push(NonZeroUsize::new(temp + 1).unwrap());
    }

    let result = Circuit {
        vertices,
        outs,
        left_input,
        right_input,
        output,
    };

    assert_eq!(
        result.vertices.len(),
        vertices_count + 2 * output_count + 2 * bools.len()
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{assert, assert_eq};
    use rand::Rng;

    #[test]
    fn test_convert() {
        let c = crate::uint32::sha256();
        let c0 = crate::circuit::convert(c);
        let c1 = convert(c0.clone());

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

        let output = crate::circuit::evaluate(&c0, &input, &modulo);

        assert_eq!(output.len(), 256);

        let left_input = input;
        let right_input = output;

        let abc = c1.evaluate(&left_input, &right_input, &modulo);
        assert!(c1
            .output
            .iter()
            .map(|v| &abc[c1.out(*v).get() - 1].c)
            .all(|i| i == &0));

        let uvwk = c1.to_constrains(&left_input, &modulo);

        crate::sonic::is_sastified(&abc, &uvwk, &modulo).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_boolean_constrain() {
        let bit_length = 32;
        let modulo = Int::from(13441);
        let c0 = crate::modulo::exp(bit_length);
        let c1 = convert(c0.clone());
        let mut input = Vec::new();
        input.push(Int::from(1998));
        let mut rng = rand::thread_rng();
        for _ in 0..bit_length {
            input.push(Int::from(rng.gen::<i8>()));
        }
        let output = crate::circuit::evaluate(&c0, &input, &modulo);
        let left_input = input;
        let right_input = output;
        let abc = c1.evaluate(&left_input, &right_input, &modulo);
        let uvwk = c1.to_constrains(&left_input, &modulo);
        crate::sonic::is_sastified(&abc, &uvwk, &modulo).unwrap();
    }
}
