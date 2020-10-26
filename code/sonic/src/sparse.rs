//! Sparse univariate and bivariate polynomial

use crate::*;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct SparsePolyZp {
    pub p: Int,
    pub coeff: Vec<(Int, i32)>,
}

impl SparsePolyZp {
    pub fn evaluate(&self, at: &Int) -> Int {
        let mut result = Int::from(0);
        for (c, d) in &self.coeff {
            result += at.clone().pow_mod(&Int::from(*d), &self.p).unwrap() * c;
        }
        result % &self.p
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct SparseBiPolyZp {
    pub p: Int,
    pub coeff: Vec<(Int, i32, i32)>,
}

impl SparseBiPolyZp {
    pub fn evaluate(&self, x: &Int, y: &Int) -> Int {
        let mut result = Int::from(0);
        for (c, i, j) in &self.coeff {
            result += c
                * x.clone().pow_mod(&Int::from(*i), &self.p).unwrap()
                * y.clone().pow_mod(&Int::from(*j), &self.p).unwrap();
        }
        result % &self.p
    }
}
