pub fn is_prime(n: u32) -> bool {
    miller(n as u64)
}

fn pow(mut a: u64, mut n: u64, m: u64) -> u64 {
    let mut result = 1;
    a = a % m;
    while n > 0 {
        if n & 1 == 1 {
            result = result * a % m;
        }
        n >>= 1;
        a = a * a % m;
    }
    return result;
}

fn witness_test(s: u32, d: u64, n: u64, witness: u64) -> bool {
    if n == witness {
        return true;
    }
    let mut p = pow(witness, d, n);
    if p == 1 {
        return true;
    }
    for _ in 0..s {
        if p == n - 1 {
            return true;
        }
        p = p * p % n;
    }
    return false;
}

fn miller(n: u64) -> bool {
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    witness_test(s, d, n, 2) && witness_test(s, d, n, 7) && witness_test(s, d, n, 61)
}
