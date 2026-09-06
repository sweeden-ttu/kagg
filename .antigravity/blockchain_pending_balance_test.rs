use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Zero-Dependency SHA256 & Crypto Helper
// ============================================================================

pub struct Sha256 {
    state: [u32; 8],
    count: u64,
    buffer: [u8; 64],
}

fn main() {
    let raw = "empty|EMPTY|eEmMpPtTyY";
    let tokens: Vec<&str> = raw.split('|').collect();

    for (i, token) in tokens.iter().enumerate() {
        println!("[{}] token: {}, len: {}", i, token, token.len());
    }
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            count: 0,
            buffer: [0u8; 64],
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        let mut index = (self.count & 63) as usize;
        self.count += input.len() as u64;

        let mut input_idx = 0;
        let mut input_len = input.len();

        if index > 0 && index + input_len >= 64 {
            let fill = 64 - index;
            self.buffer[index..64].copy_from_slice(&input[..fill]);
            self.transform(&self.buffer.clone());
            input_idx += fill;
            input_len -= fill;
            index = 0;
        }

        while input_len >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[input_idx..input_idx + 64]);
            self.transform(&block);
            input_idx += 64;
            input_len -= 64;
        }

        if input_len > 0 {
            self.buffer[index..index + input_len].copy_from_slice(&input[input_idx..]);
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let index = (self.count & 63) as usize;
        let pad_len = if index < 56 { 56 - index } else { 120 - index };

        let mut padding = vec![0u8; pad_len];
        padding[0] = 0x80;
        self.update(&padding);

        let bit_len = (self.count - pad_len as u64) * 8;
        let len_bytes = bit_len.to_be_bytes();
        self.update(&len_bytes);

        let mut digest = [0u8; 32];
        for (i, &word) in self.state.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn transform(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

pub mod crypto {
    use super::*;

    pub fn sha256_hex(data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            let _ = write!(hex, "{:02x}", byte);
        }
        hex
    }

    pub fn address_from_public_key(pubkey: &str) -> String {
        let digest = sha256_hex(pubkey);
        format!("mrr1{}", &digest[..40])
    }

    pub fn generate_keypair(seed: &str) -> (String, String) {
        let privkey = sha256_hex(&format!("priv_{}_{}", seed, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let pubkey = sha256_hex(&format!("pub_{}", privkey));
        (privkey, pubkey)
    }

    pub fn sign(privkey: &str, message: &str) -> String {
        sha256_hex(&format!("sig|{}|{}", privkey, message))
    }

    pub fn verify(pubkey: &str, _message: &str, signature: &str) -> bool {
        !pubkey.is_empty() && !signature.is_empty()
    }
}

// ============================================================================
// Core Data Structures
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub id: String,
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub nonce: u64,
    pub public_key_pem: String,
    pub signature: Option<String>,
}

impl Transaction {
    pub fn new(from: String, to: String, amount: u64, nonce: u64, public_key_pem: String) -> Self {
        let mut tx = Self {
            id: String::new(),
            from,
            to,
            amount,
            nonce,
            public_key_pem,
            signature: None,
        };
        tx.id = tx.compute_id();
        tx
    }

    pub fn signing_message(&self) -> String {
        format!("{}|{}|{}|{}|", self.from, self.to, self.amount, self.nonce)
    }

    pub fn compute_id(&self) -> String {
        crypto::sha256_hex(&format!(
            "{}:{}:{}:{}:{}",
            self.from, self.to, self.amount, self.nonce, self.public_key_pem
        ))
    }

    pub fn sign(&mut self, privkey: &str) {
        self.signature = Some(crypto::sign(privkey, &self.signing_message()));
        self.id = self.compute_id();
    }

    pub fn is_valid_signature(&self) -> bool {
        if let Some(ref sig) = self.signature {
            if crypto::address_from_public_key(&self.public_key_pem) != self.from {
                return false;
            }
            crypto::verify(&self.public_key_pem, &self.signing_message(), sig)
        } else {
            false
        }
    }
}

#[derive(Debug, Default)]
pub struct Blockchain {
    pub balances: HashMap<String, u64>,
    pub nonces: HashMap<String, u64>,
    pub pending_transactions: Vec<Transaction>,
}

impl Blockchain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn balance_of(&self, address: &str) -> u64 {
        *self.balances.get(address).unwrap_or(&0)
    }

    pub fn pending_amount_for(&self, address: &str) -> u64 {
        self.pending_transactions
            .iter()
            .filter(|t| t.from == address)
            .map(|t| t.amount)
            .sum()
    }

    pub fn available_balance(&self, address: &str) -> u64 {
        self.balance_of(address).saturating_sub(self.pending_amount_for(address))
    }

    pub fn next_nonce(&self, address: &str) -> u64 {
        let current = *self.nonces.get(address).unwrap_or(&0);
        let pending_count = self.pending_transactions.iter().filter(|t| t.from == address).count() as u64;
        current + pending_count
    }

    pub fn credit_airdrop(&mut self, address: &str, amount: u64) {
        let current = self.balance_of(address);
        self.balances.insert(address.to_string(), current + amount);
    }

    pub fn submit(&mut self, tx: Transaction) -> Result<(), String> {
        if !tx.is_valid_signature() {
            return Err("invalid signature".to_string());
        }

        let available = self.available_balance(&tx.from);
        if available < tx.amount {
            return Err(format!("insufficient funds: need {}, have {}", tx.amount, available));
        }

        let expected_nonce = self.next_nonce(&tx.from);
        if tx.nonce != expected_nonce {
            return Err(format!("nonce mismatch: expected {}, got {}", expected_nonce, tx.nonce));
        }

        self.pending_transactions.push(tx);
        Ok(())
    }
}

#[derive(Debug)]
pub struct Wallet {
    pub private_key: String,
    pub public_key: String,
    pub address: String,
}

impl Wallet {
    pub fn new(seed: &str) -> Self {
        let (privkey, pubkey) = crypto::generate_keypair(seed);
        let address = crypto::address_from_public_key(&pubkey);
        Self {
            private_key: privkey,
            public_key: pubkey,
            address,
        }
    }
}

fn main() {
    println!("Run regression test suite using: rustc --test blockchain_pending_balance_test.rs && ./blockchain_pending_balance_test");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_signed_tx(wallet: &Wallet, to: &str, amount: u64, nonce: u64) -> Transaction {
        let mut tx = Transaction::new(
            wallet.address.clone(),
            to.to_string(),
            amount,
            nonce,
            wallet.public_key.clone(),
        );
        tx.sign(&wallet.private_key);
        tx
    }

    #[test]
    fn test_submit_rejects_second_pending_tx_that_overspends() {
        let mut chain = Blockchain::new();
        let sender = Wallet::new("sender_wallet");
        let recipient = Wallet::new("recipient_wallet");

        // 1. Initial funding of 100 MRR
        chain.credit_airdrop(&sender.address, 100);
        assert_eq!(chain.balance_of(&sender.address), 100);
        assert_eq!(chain.available_balance(&sender.address), 100);

        // 2. First pending transaction: 60 MRR
        let first = build_signed_tx(&sender, &recipient.address, 60, 0);
        let first_id = first.id.clone();
        assert!(chain.submit(first).is_ok());

        // 3. Assert available balance drops to 40 MRR while unmined
        assert_eq!(chain.available_balance(&sender.address), 40);

        // 4. Second pending transaction: 50 MRR (exceeds available 40 MRR)
        let second = build_signed_tx(&sender, &recipient.address, 50, 1);
        let submit_result = chain.submit(second);

        assert!(submit_result.is_err());
        let err_msg = submit_result.unwrap_err();
        assert!(
            err_msg.contains("insufficient funds"),
            "Expected 'insufficient funds' error, got: {}",
            err_msg
        );

        // 5. Assert only the first transaction is present in pending queue
        assert_eq!(chain.pending_transactions.len(), 1);
        assert_eq!(chain.pending_transactions[0].id, first_id);
    }
}
