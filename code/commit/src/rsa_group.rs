use crate::integer::*;
use crate::{assert, assert_eq};

struct RSAGroup {
    n: RSAGroupElement,
}

type RSAGroupElement = u64;

impl RSAGroup {
    /// Calculate g^{f(q)}
    fn exp_polynomial(&self, g: RSAGroupElement, f: IntegerPolynomial, q: Integer) -> RSAGroupElement {
        assert!(self.is_group_element(g));
    }

    fn is_group_element(&self, g: RSAGroupElement) -> bool {
        gcd(self.n, g) == 1
    }
}

fn gcd(mut a: RSAGroupElement, mut b: RSAGroupElement) -> RSAGroupElement {
    while b != 0 {
        let temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}
