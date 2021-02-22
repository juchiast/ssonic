#[macro_use]
extern crate log;

mod dark;
pub mod poe;

pub use crate::dark::{Instance, VerifiableKey, DARK};
pub use poe::proof_of_exponentation;

#[cfg(test)]
mod test {
    #[cfg(feature = "no-assert")]
    #[test]
    #[should_panic]
    fn test_no_assert() {
        common::assert!(false);
        common::assert_eq!(1, 2);
    }
}
