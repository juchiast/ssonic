fn main() {
    let c = sonic::uint32::sha256();
    println!("{}", sonic::uint32::collect_vertices(c.clone()).len());
    let c = sonic::circuit::convert(c);
    println!("{}", sonic::circuit::toposort(c.clone()).len());
}
