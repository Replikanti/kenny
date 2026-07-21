//! SplitMix64 — hand-rolled per ADR-0021 (`rand` is banned; determinism is a
//! feature). Used only to generate fixture tensor values; per-tensor streams
//! are derived from (seed, tensor name) via blake3 so the bytes of one tensor
//! never depend on generation order.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Stream keyed by (seed, name): blake3(seed_le || name), first 8 bytes.
    pub fn for_name(seed: u64, name: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&seed.to_le_bytes());
        h.update(name.as_bytes());
        let digest = h.finalize();
        let mut b = [0u8; 8];
        b.copy_from_slice(&digest.as_bytes()[..8]);
        SplitMix64::new(u64::from_le_bytes(b))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-0.5, 0.5). The top 24 bits divided by 2^24 are exactly
    /// representable in f32, so the mapping is bit-deterministic everywhere.
    pub fn next_unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0 - 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference vectors from Vigna's splitmix64.c, cross-checked in Python.
    #[test]
    fn reference_vectors() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
        let mut r = SplitMix64::new(42);
        assert_eq!(r.next_u64(), 0xBDD7_3226_2FEB_6E95);
        assert_eq!(r.next_u64(), 0x28EF_E333_B266_F103);
    }

    #[test]
    fn unit_range_and_determinism() {
        let mut a = SplitMix64::for_name(42, "model.norm.weight");
        let mut b = SplitMix64::for_name(42, "model.norm.weight");
        let mut c = SplitMix64::for_name(42, "lm_head.weight");
        let mut differs = false;
        for _ in 0..1000 {
            let va = a.next_unit_f32();
            assert!((-0.5..0.5).contains(&va));
            assert_eq!(va.to_bits(), b.next_unit_f32().to_bits());
            if va.to_bits() != c.next_unit_f32().to_bits() {
                differs = true;
            }
        }
        assert!(differs, "distinct names must yield distinct streams");
    }
}
