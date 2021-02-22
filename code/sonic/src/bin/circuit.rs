fn main() {
    let circuit = sonic::modulo::exp(64);
    let linear_circuit = sonic::linear_circuit::convert(circuit);
    println!("POW: {}", linear_circuit.vertices.len());

    let uint32_circuit = sonic::uint32::sha256();
    let circuit = sonic::circuit::convert(uint32_circuit);
    let linear_circuit = sonic::linear_circuit::convert(circuit);
    println!("SHA: {}", linear_circuit.vertices.len());
}
