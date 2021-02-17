#![feature(test)]

#[macro_use]
extern crate log;

pub mod dark;
pub mod fiat_shamir;
pub mod group;
pub mod poe;
pub mod poly;

pub use rand_chacha::ChaCha20Rng;
pub use rug;
/// Big int type
pub use rug::Integer as Int;

pub use dark::DARK;
pub use fiat_shamir::gen_prime::PierreGenPrime;
pub use fiat_shamir::uniform::UniformRandom;
pub use fiat_shamir::{FiatShamirRng, ProofElement, Prover};
pub use group::RSAGroup;
pub use poe::proof_of_exponentation;
pub use poly::{PolyZ, PolyZp};

pub mod traits {
    pub use crate::fiat_shamir::{FeedHasher, ProverMessage};
    pub use crate::group::Group;
}

#[macro_export]
macro_rules! assert {
    ($ ($ arg : tt) *) => {
        if !cfg!(feature = "no-assert") || cfg!(test) {
            std::assert!($( $arg ) *);
        }
    };
}

#[macro_export]
macro_rules! assert_eq {
    ($ ($ arg : tt) *) => {
        if !cfg!(feature = "no-assert") || cfg!(test) {
            std::assert_eq!($( $arg ) *);
        }
    };
}

#[cfg(test)]
mod test {
    #[cfg(feature = "no-assert")]
    #[test]
    #[should_panic]
    fn test_no_assert() {
        assert!(false);
    }
}
