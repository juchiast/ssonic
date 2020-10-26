
fn main() {
    let circuit = supersonic::modulo::exp(64);
    let linear_circuit = supersonic::linear_circuit::convert(circuit);
    println!("POW: {}", linear_circuit.vertices.len());

    let uint32_circuit = supersonic::uint32::sha256();
    let circuit = supersonic::circuit::convert(uint32_circuit);
    let linear_circuit = supersonic::linear_circuit::convert(circuit);
    println!("SHA: {}", linear_circuit.vertices.len());
}
