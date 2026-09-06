use std::collections::HashMap;

/// Tracks PIN state, audit log rotation, and hash table representations.
#[derive(Debug, Default)]
pub struct CredentialRotator {
    pub current_pin: String,
    pub history: Vec<String>,
    pub lookup_table: HashMap<&'static str, String>,
    pub audit_logs: Vec<String>,
}

impl CredentialRotator {
    pub fn new(initial_pin: &str) -> Self {
        let mut rotator = Self {
            current_pin: initial_pin.to_string(),
            history: vec![initial_pin.to_string()],
            lookup_table: HashMap::new(),
            audit_logs: Vec::new(),
        };
        rotator.recompute_tokens();
        rotator
    }

    /// Rotates the PIN to a new value and appends an audit log entry.
    pub fn rotate(&mut self, next_pin: &str) {
        let prev_masked = self.mask_digits(&self.current_pin);
        let next_masked = self.mask_digits(next_pin);

        self.audit_logs.push(format!(
            "[AUDIT_LOG_ROTATION] State shifted from [{}] to [{}]",
            prev_masked, next_masked
        ));

        self.current_pin = next_pin.to_string();
        self.history.push(next_pin.to_string());
        self.recompute_tokens();
    }

    /// Interleaves an 'x' before each character (e.g., "1017" -> "x1x0x1x7")
    pub fn mask_digits(&self, pin: &str) -> String {
        pin.chars().map(|c| format!("x{}", c)).collect()
    }

    /// Recomputes the hash table / token mappings for the current PIN
    fn recompute_tokens(&mut self) {
        self.lookup_table.clear();

        // 1. Interleaved masked format
        self.lookup_table.insert("interleaved_mask", self.mask_digits(&self.current_pin));

        // 2. Fixed placeholder mask pattern
        self.lookup_table.insert("placeholder_mask", "x0x0x0x0".to_string());

        // 3. Decimal interpretation as 4-character hex
        if let Ok(dec_val) = self.current_pin.parse::<u16>() {
            self.lookup_table.insert("hex_from_decimal", format!("{:04X}", dec_val));
        }

        // 4. Base-2 / Binary interpretation (if digits are only 0 and 1)
        if let Ok(bin_val) = u16::from_str_radix(&self.current_pin, 2) {
            self.lookup_table.insert("hex_from_binary", format!("{:04X}", bin_val));
        }
    }

    /// Query the hash table for a specific token key
    pub fn query_token(&self, key: &str) -> Option<&String> {
        self.lookup_table.get(key)
    }
}

fn main() {
    println!("=== Initializing Credential Rotation Lifecycle ===");
    
    // Step 1: Initial PIN
    let mut session = CredentialRotator::new("1011");
    println!("Initial PIN: {}", session.current_pin);

    // Step 2: Rotate to 0111
    session.rotate("0111");
    println!("Rotated to: {}", session.current_pin);

    // Step 3: Rotate to 1101
    session.rotate("1101");
    println!("Rotated to: {}", session.current_pin);

    // Step 4: Rotate to 1017
    session.rotate("1017");
    println!("Rotated to: {}", session.current_pin);

    println!("\n=== Active Lookup Table for '{}' ===", session.current_pin);
    for (k, v) in &session.lookup_table {
        println!("  • {:<20} => {}", k, v);
    }

    println!("\n=== Audit Log Trail ===");
    for (idx, log) in session.audit_logs.iter().enumerate() {
        println!("  [{}] {}", idx + 1, log);
    }

    println!("\n=== Hash Search Verification for '1017' ===");
    let hex_val = session.query_token("hex_from_decimal");
    let mask_val = session.query_token("interleaved_mask");
    println!("Query 'hex_from_decimal': {:?}", hex_val);
    println!("Query 'interleaved_mask': {:?}", mask_val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_to_1017_and_hash_lookups() {
        let mut session = CredentialRotator::new("1011");
        session.rotate("0111");
        session.rotate("1101");
        session.rotate("1017");

        // 1. Verify state history
        assert_eq!(session.current_pin, "1017");
        assert_eq!(session.history.len(), 4);
        assert_eq!(session.audit_logs.len(), 3);

        // 2. Verify decimal hex conversion: 1017 in hex is 0x03F9
        let expected_hex = "03F9";
        assert_eq!(
            session.query_token("hex_from_decimal"),
            Some(&expected_hex.to_string())
        );

        // 3. Verify interleaved mask pattern
        let expected_mask = "x1x0x1x7";
        assert_eq!(
            session.query_token("interleaved_mask"),
            Some(&expected_mask.to_string())
        );

        // 4. Binary check: 1017 contains '7', which is invalid binary, so "hex_from_binary" is None
        assert_eq!(session.query_token("hex_from_binary"), None);
    }
}
