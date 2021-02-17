#[macro_use]
extern crate log;
#[macro_use]
extern crate poly_commit;

pub mod circuit;
pub mod linear_circuit;
pub mod modulo;
pub mod sonic;
pub mod sparse;
pub mod uint32;

pub use poly_commit::Int;
pub use sonic::{ABC, SK, UVWK};
pub use sparse::{SparseBiPolyZp, SparsePolyZp};

#[cfg(test)]
mod test {
    #[cfg(feature = "no-assert")]
    #[test]
    #[should_panic]
    fn test_no_assert() {
        assert!(false);
    }
}
