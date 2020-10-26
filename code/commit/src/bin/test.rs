#![allow(warnings)]

use poly_commit::traits::*;
use poly_commit::*;
use rug::ops::Pow;
use std::time::Instant;

fn t1() {
    let group = group::RSAGroup::new(3048);
    let mut rng = rand::thread_rng();
    let mut uniform = UniformRandom::new();
    let mut gen_prime = PierreGenPrime::new(512, 304);
    let mut q = Int::from(0);
    uniform.rand_size(6100, &mut rng, &mut q);
    q.set_bit(0, true);
    let d = 10000usize;
    let g = {
        let mut result = Int::from(0);
        loop {
            uniform.rand_below(&group.modulo, &mut rng, &mut result);
            if result > 1 && group.is_element(&result) {
                break result;
            }
        }
    };
    let l = gen_prime.gen(&mut rng);

    let start = Instant::now();
    let qq = q / &l;
    println!("{:?}", Instant::now() - start);
    // dbg!(qq.significant_bits());
}

fn main() {
    t1();
}
