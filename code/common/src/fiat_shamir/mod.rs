use crate::Int;
use digest::Digest;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::VecDeque;

pub mod gen_prime;
pub mod miller;
pub mod uniform;

use gen_prime::PierreGenPrime;
use uniform::UniformRandom;

pub type Hasher = blake3::Hasher;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ProofElement {
    Text(String),
    Bytes(Vec<u8>),
    Number(Int),
    Numbers(Vec<Int>),
}

pub trait FeedHasher {
    fn feed_hasher(&self, hasher: &mut Hasher, buffer: &mut Vec<u8>);
}

impl FeedHasher for crate::RSAGroup {
    fn feed_hasher(&self, hasher: &mut Hasher, buffer: &mut Vec<u8>) {
        self.modulo.feed_hasher(hasher, buffer)
    }
}

impl FeedHasher for usize {
    fn feed_hasher(&self, hasher: &mut Hasher, _buffer: &mut Vec<u8>) {
        let x = *self as u64;
        hasher.update(&x.to_le_bytes());
    }
}

impl FeedHasher for Vec<Int> {
    fn feed_hasher(&self, hasher: &mut Hasher, buffer: &mut Vec<u8>) {
        for i in self {
            i.feed_hasher(hasher, buffer);
        }
    }
}

impl FeedHasher for Int {
    fn feed_hasher(&self, hasher: &mut Hasher, buffer: &mut Vec<u8>) {
        let size = self.significant_digits::<u8>();
        size.feed_hasher(hasher, buffer);
        buffer.resize(size, 0);
        self.write_digits(buffer.as_mut_slice(), rug::integer::Order::LsfLe);
        hasher.update(buffer);
    }
}

impl FeedHasher for str {
    fn feed_hasher(&self, hasher: &mut Hasher, _buffer: &mut Vec<u8>) {
        hasher.update(self.as_bytes());
    }
}

impl<'a> FeedHasher for [&'a dyn FeedHasher] {
    fn feed_hasher(&self, hasher: &mut Hasher, buffer: &mut Vec<u8>) {
        self.len().feed_hasher(hasher, buffer);
        for i in self {
            i.feed_hasher(hasher, buffer);
        }
    }
}

pub trait ProverMessage
where
    Self: FeedHasher + Sized + std::fmt::Debug,
{
    fn to_proof_element(&self) -> ProofElement;
    fn from_proof_element(p: ProofElement) -> Option<Self>;
}

impl ProverMessage for Int {
    fn to_proof_element(&self) -> ProofElement {
        ProofElement::Number(self.clone())
    }

    fn from_proof_element(p: ProofElement) -> Option<Self> {
        match p {
            ProofElement::Number(i) => Some(i),
            _ => None,
        }
    }
}

impl ProverMessage for Vec<Int> {
    fn to_proof_element(&self) -> ProofElement {
        ProofElement::Numbers(self.clone())
    }

    fn from_proof_element(p: ProofElement) -> Option<Self> {
        match p {
            ProofElement::Numbers(v) => Some(v),
            _ => None,
        }
    }
}

pub struct FiatShamirRng {
    seed: [u8; 32], // 256 bits
    prime_gen: PierreGenPrime,
    uniform: UniformRandom,
    rng: ChaCha20Rng,

    buffer: Vec<u8>,
    pub proofs: VecDeque<ProofElement>,
}

impl FiatShamirRng {
    pub fn new<S: FeedHasher + ?Sized>(seed: &S, prime_gen: PierreGenPrime) -> Self {
        let proofs = VecDeque::new();
        let mut hasher = Hasher::new();
        let mut buffer = Vec::new();
        seed.feed_hasher(&mut hasher, &mut buffer);
        let hash = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&hash);
        let rng = ChaCha20Rng::from_seed(seed);
        Self {
            rng,
            seed,
            prime_gen,
            uniform: UniformRandom::new(),
            buffer,
            proofs,
        }
    }

    pub fn prover_send<M: ProverMessage>(&mut self, msg: &M) {
        let mut hasher = Hasher::new();
        hasher.update(&self.seed);
        msg.feed_hasher(&mut hasher, &mut self.buffer);
        self.proofs.push_back(msg.to_proof_element());
        let hash = hasher.finalize();
        self.seed.copy_from_slice(&hash);
        self.rng = ChaCha20Rng::from_seed(self.seed);
    }

    /// Generate a random b-bits primes.
    /// Used in proof of exponentation.
    pub fn verifier_rand_prime(&mut self) -> Int {
        self.prime_gen.gen(&mut self.rng)
    }

    /// Uniform random in [0, b)
    pub fn verifier_rand_below(&mut self, b: &Int) -> Int {
        let mut result = Int::from(0);
        self.uniform.rand_below(b, &mut self.rng, &mut result);
        result
    }

    /// Uniform random in [-b, b]
    pub fn verifier_rand_signed(&mut self, b: &Int) -> Int {
        assert!(b > &0);
        let mut result = Int::from(0);
        self.uniform.rand_signed(b, &mut self.rng, &mut result);
        result
    }

    /// Proof's length in bytes.
    pub fn proof_length(&self) -> usize {
        proof_length(&self.proofs)
    }
}

pub fn proof_length(proof: &VecDeque<ProofElement>) -> usize {
    let mut result = 0;
    for p in proof {
        result += match p {
            ProofElement::Number(n) => n.significant_digits::<u8>() as usize,
            ProofElement::Numbers(vec) => vec
                .iter()
                .map(|n| n.significant_digits::<u8>() as usize)
                .sum(),
            ProofElement::Text(t) => t.len(),
            ProofElement::Bytes(b) => b.len(),
        };
    }
    result
}

pub enum Prover<T> {
    Witness(T),
    Proof(VecDeque<ProofElement>),
}

pub trait MessagesBatch
where
    Self: Sized + std::fmt::Debug,
{
    fn extract_prover_messages(
        proof: VecDeque<ProofElement>,
    ) -> Result<(Self, VecDeque<ProofElement>), String>;

    fn prover_send_batch(&self, fiat: &mut FiatShamirRng);
}

impl<T> MessagesBatch for T
where
    T: ProverMessage,
{
    fn extract_prover_messages(
        mut proof: VecDeque<ProofElement>,
    ) -> Result<(Self, VecDeque<ProofElement>), String> {
        match proof.pop_front() {
            Some(p) => match <T as ProverMessage>::from_proof_element(p) {
                Some(t) => Ok((t, proof)),
                None => Err("Wrong proof type".to_owned()),
            },
            None => Err("proof is empty".to_owned()),
        }
    }

    fn prover_send_batch(&self, fiat: &mut FiatShamirRng) {
        fiat.prover_send(self);
    }
}

/// When prover doesn't send anything
impl MessagesBatch for () {
    fn extract_prover_messages(
        proof: VecDeque<ProofElement>,
    ) -> Result<(Self, VecDeque<ProofElement>), String> {
        Ok(((), proof))
    }

    fn prover_send_batch(&self, _fiat: &mut FiatShamirRng) {}
}

impl<A, B> MessagesBatch for (A, B)
where
    A: MessagesBatch,
    B: MessagesBatch,
{
    fn extract_prover_messages(
        proof: VecDeque<ProofElement>,
    ) -> Result<(Self, VecDeque<ProofElement>), String> {
        let (a, proof) = <A as MessagesBatch>::extract_prover_messages(proof)?;
        let (b, proof) = <B as MessagesBatch>::extract_prover_messages(proof)?;
        Ok(((a, b), proof))
    }

    fn prover_send_batch(&self, fiat: &mut FiatShamirRng) {
        self.0.prover_send_batch(fiat);
        self.1.prover_send_batch(fiat);
    }
}

impl<A, B, C, D> MessagesBatch for (A, B, C, D)
where
    A: MessagesBatch,
    B: MessagesBatch,
    C: MessagesBatch,
    D: MessagesBatch,
{
    fn extract_prover_messages(
        proof: VecDeque<ProofElement>,
    ) -> Result<(Self, VecDeque<ProofElement>), String> {
        let (a, proof) = <A as MessagesBatch>::extract_prover_messages(proof)?;
        let (b, proof) = <B as MessagesBatch>::extract_prover_messages(proof)?;
        let (c, proof) = <C as MessagesBatch>::extract_prover_messages(proof)?;
        let (d, proof) = <D as MessagesBatch>::extract_prover_messages(proof)?;
        Ok(((a, b, c, d), proof))
    }

    fn prover_send_batch(&self, fiat: &mut FiatShamirRng) {
        self.0.prover_send_batch(fiat);
        self.1.prover_send_batch(fiat);
        self.2.prover_send_batch(fiat);
        self.3.prover_send_batch(fiat);
    }
}

#[macro_export]
macro_rules! prover {
    ($fiat:expr, $prover:ident, ( $( $name:pat),* ) => $code:block) => {{
        #[allow(unused_parens)]
        let (batch, prover) = match $prover {
            $crate::Prover::Witness(($($name),*)) => {
                let (r, w) = $code;
                (r, $crate::Prover::Witness(w))
            },
            $crate::Prover::Proof(p) => {
                let (r, p) = $crate::fiat_shamir::MessagesBatch::extract_prover_messages(p)?;
                (r, $crate::Prover::Proof(p))
            },
        };
        $crate::fiat_shamir::MessagesBatch::prover_send_batch(&batch, $fiat);
        (batch, prover)
    }};
}
