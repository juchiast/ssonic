use crate::traits;
use crate::Int;
use crate::{assert, assert_eq};

/// Multiplicative group of unknown order
pub trait Group {
    type Element: std::cmp::PartialEq<Self::Element>
        + std::fmt::Debug
        + std::fmt::Display
        + serde::de::DeserializeOwned
        + serde::Serialize
        + Clone
        + traits::ProverMessage;

    /// base^exponent
    fn power(&self, base: &Self::Element, exponent: &Int) -> Self::Element;

    /// a * b
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// a == b
    fn equal(&self, a: &Self::Element, b: &Self::Element) -> bool;

    fn is_element(&self, a: &Self::Element) -> bool;

    /// a / b
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    fn inverse(&self, x: &Self::Element) -> Self::Element;
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RSAGroup {
    pub modulo: Int,
}

impl Group for RSAGroup {
    type Element = Int;

    fn inverse(&self, x: &Self::Element) -> Self::Element {
        x.clone().invert(&self.modulo).unwrap()
    }

    fn power(&self, base: &Self::Element, exponent: &Int) -> Self::Element {
        assert!(self.is_element(base));
        let result = base.clone().pow_mod(exponent, &self.modulo).unwrap();
        assert!(self.is_element(&result));
        result
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        assert!(self.is_element(a));
        assert!(self.is_element(b));
        let mut result = a.clone();
        result *= b;
        result %= &self.modulo;
        assert!(self.is_element(&result));
        result
    }

    fn equal(&self, a: &Self::Element, b: &Self::Element) -> bool {
        assert!(self.is_element(a));
        assert!(self.is_element(b));
        a == b
    }

    fn is_element(&self, a: &Self::Element) -> bool {
        a > &0 && a < &self.modulo && a.clone().gcd(&self.modulo) == 1
    }

    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        assert!(self.is_element(a));
        assert!(self.is_element(b));
        let mut result = b.clone().invert(&self.modulo).unwrap();
        result *= a;
        result %= &self.modulo;
        result
    }
}

impl RSAGroup {
    pub fn new(bit_count: u32) -> Self {
        let rsa = openssl::rsa::Rsa::generate(bit_count).unwrap();
        let n = rsa.n().to_vec();
        let modulo = Int::from_digits(&n, rug::integer::Order::MsfBe);
        assert_eq!(rsa.n().to_string(), modulo.to_string());
        Self { modulo }
    }
}
