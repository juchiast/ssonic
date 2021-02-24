#![allow(clippy::many_single_char_names, clippy::let_and_return)]

#[macro_use]
extern crate log;

pub mod circuit;
pub mod linear_circuit;
pub mod modulo;
pub mod sonic;
pub mod sparse;
pub mod uint32;

pub use crate::sonic::{ABC, SK, UVWK};
pub use sparse::{SparseBiPolyZp, SparsePolyZp};

#[cfg(test)]
mod test {
    #[cfg(feature = "no-assert")]
    #[test]
    #[should_panic]
    fn test_no_assert() {
        common::assert!(false);
    }
}
