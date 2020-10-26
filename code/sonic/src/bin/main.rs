fn main() {
    let c = supersonic::uint32::sha256();
    println!("{}", supersonic::uint32::collect_vertices(c.clone()).len());
    let c = supersonic::circuit::convert(c);
    println!("{}", supersonic::circuit::toposort(c.clone()).len());
}
