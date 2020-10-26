use super::*;
use std::collections::HashMap;
use std::collections::HashSet;

type Graph = HashMap<Vertex, Node>;

#[derive(Default)]
struct Node {
    in_deg: u32,
    next: Vec<Vertex>,
}

fn add_edge(g: &mut Graph, from: &Vertex, to: &Vertex) {
    g.entry(from.clone()).or_default().next.push(to.clone());
}

pub fn toposort(vertices: Vec<super::Vertex>) -> Vec<super::Vertex> {
    let mut g = Graph::new();

    let mut queue = vertices;
    let mut deg_0 = Vec::new();
    let mut all_vertices = HashSet::new();
    while !queue.is_empty() {
        for u in std::mem::take(&mut queue) {
            if !all_vertices.contains(&u) {
                let in_deg = match u.inner.as_ref() {
                    VertexType::Add(a, b)
                    | VertexType::Substract(a, b)
                    | VertexType::Multiply(a, b) => {
                        add_edge(&mut g, a, &u);
                        add_edge(&mut g, b, &u);
                        queue.push(a.clone());
                        queue.push(b.clone());
                        2
                    }
                    VertexType::Input(_, _)
                    | VertexType::Constant(_)
                    | VertexType::ConstantInverse(_) => {
                        deg_0.push(u.clone());
                        0
                    }
                };
                g.entry(u.clone()).or_default().in_deg = in_deg;
                all_vertices.insert(u);
            }
        }
    }
    assert_eq!(all_vertices.len(), g.len());

    let mut result = Vec::new();

    let mut deg_1 = HashSet::new();
    while !deg_0.is_empty() {
        for u in std::mem::take(&mut deg_0) {
            result.push(u.clone());
            for v in &g[&u].next {
                if deg_1.contains(v) {
                    deg_1.remove(v);
                    deg_0.push(v.clone());
                } else {
                    deg_1.insert(v.clone());
                }
            }
        }
    }

    assert_eq!(all_vertices.len(), result.len());

    result
}
