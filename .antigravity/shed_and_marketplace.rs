use std::fmt;

// ============================================================================
// 1. 128-BIT FARMER #1 SHED (Crypto Payment Acceptance & Storage Bitmask)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoCurrency {
    MirrorCoin = 0, // MRR (Native game currency, 2112 bank)
    Bitcoin = 1,    // BTC
    Ethereum = 2,   // ETH
    Dogecoin = 3,   // DOGE
    Solana = 4,     // SOL
    Monero = 5,     // XMR
    Usdc = 6,       // USDC
    Usdt = 7,       // USDT
    FellowshipToken = 8, // Sub-agent 888 Stake Token
    ProofOfYield = 9,    // On-chain harvest receipt
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FarmerShed128 {
    pub raw: u128,
}

impl FarmerShed128 {
    pub const CAPACITY_BITS: usize = 128;

    /// Creates an empty 128-bit shed
    pub fn new() -> Self {
        Self { raw: 0 }
    }

    /// Initializes Farmer #1 shed with native default payment acceptance (MirrorCoin + FellowshipToken)
    pub fn new_farmer_one_default() -> Self {
        let mut shed = Self::new();
        shed.enable_currency(CryptoCurrency::MirrorCoin);
        shed.enable_currency(CryptoCurrency::FellowshipToken);
        // Set lower 32-bit storage partition flag for physical inventory
        shed.raw |= 1 << 64; // Storage slot 0 active
        shed
    }

    /// Enable a cryptocurrency payment type
    pub fn enable_currency(&mut self, currency: CryptoCurrency) {
        let shift = currency as usize;
        self.raw |= 1u128 << shift;
    }

    /// Disable a cryptocurrency payment type
    pub fn disable_currency(&mut self, currency: CryptoCurrency) {
        let shift = currency as usize;
        self.raw &= !(1u128 << shift);
    }

    /// Check if Farmer #1 accepts this payment type
    pub fn accepts_currency(&self, currency: CryptoCurrency) -> bool {
        let shift = currency as usize;
        (self.raw & (1u128 << shift)) != 0
    }

    /// List all accepted currencies
    pub fn list_accepted_currencies(&self) -> Vec<CryptoCurrency> {
        let all = [
            CryptoCurrency::MirrorCoin,
            CryptoCurrency::Bitcoin,
            CryptoCurrency::Ethereum,
            CryptoCurrency::Dogecoin,
            CryptoCurrency::Solana,
            CryptoCurrency::Monero,
            CryptoCurrency::Usdc,
            CryptoCurrency::Usdt,
            CryptoCurrency::FellowshipToken,
            CryptoCurrency::ProofOfYield,
        ];

        all.iter().cloned().filter(|c| self.accepts_currency(*c)).collect()
    }

    /// Returns hexadecimal 128-bit representation
    pub fn to_hex(&self) -> String {
        format!("0x{:032X}", self.raw)
    }
}

impl fmt::Display for FarmerShed128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FarmerShed128 [Hex: {}, Accepted Currencies: {:?}]",
            self.to_hex(),
            self.list_accepted_currencies()
        )
    }
}

// ============================================================================
// 2. 64-BIT CULTURE MARKETPLACE (Demand Dictionary: Seeds, Foods, Animals)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketItem {
    // --- Seeds (Bits 0..7) ---
    SeedWheat = 0,
    SeedCarrot = 1,
    SeedTomato = 2,
    SeedStrawberry = 3,
    SeedMelon = 4,

    // --- Foods & Harvested Crops (Bits 8..23) ---
    CropWheat = 8,
    CropCarrot = 9,
    CropTomato = 10,
    CropStrawberry = 11,
    CropMelon = 12,
    FoodEgg = 13,
    FoodMilk = 14,
    ProductFertilizer = 15,
    FoodFlour = 16,
    FoodBread = 17,

    // --- Animals (Bits 24..31) ---
    AnimalChicken = 24,
    AnimalCow = 25,
    AnimalSheep = 26,

    // --- Export Commodities (Bits 32..47) ---
    ExportWool = 32,
    ExportBulkGrain = 33,
    ExportDairyCheese = 34,
    ExportPreserves = 35,

    // --- Macro Demand Regime Flags (Bits 48..63) ---
    DomesticConsumptionMajority = 48,
    ExportDemandMajority = 49,
    HarvestSeasonSurge = 50,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CultureMarketplace64 {
    pub raw: u64,
}

impl CultureMarketplace64 {
    pub const CAPACITY_BITS: usize = 64;

    pub fn new() -> Self {
        Self { raw: 0 }
    }

    /// Default baseline culture market demand
    pub fn new_standard_regime() -> Self {
        let mut market = Self::new();
        // High consumption staple demand
        market.set_demand(MarketItem::SeedWheat, true);
        market.set_demand(MarketItem::SeedCarrot, true);
        market.set_demand(MarketItem::CropWheat, true);
        market.set_demand(MarketItem::CropCarrot, true);
        market.set_demand(MarketItem::FoodEgg, true);
        market.set_demand(MarketItem::FoodMilk, true);
        market.set_demand(MarketItem::AnimalChicken, true);
        market.set_demand(MarketItem::AnimalCow, true);
        // Export demand
        market.set_demand(MarketItem::ExportWool, true);
        market.set_demand(MarketItem::ExportBulkGrain, true);
        // Macro flags
        market.set_demand(MarketItem::DomesticConsumptionMajority, true);
        market.set_demand(MarketItem::ExportDemandMajority, true);
        market
    }

    pub fn set_demand(&mut self, item: MarketItem, active: bool) {
        let shift = item as usize;
        if active {
            self.raw |= 1u64 << shift;
        } else {
            self.raw &= !(1u64 << shift);
        }
    }

    pub fn is_in_demand(&self, item: MarketItem) -> bool {
        let shift = item as usize;
        (self.raw & (1u64 << shift)) != 0
    }

    pub fn to_hex(&self) -> String {
        format!("0x{:016X}", self.raw)
    }
}

impl fmt::Display for CultureMarketplace64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CultureMarketplace64 [Hex: {}]", self.to_hex())
    }
}

// ============================================================================
// 3. MAIN RUNNER & TEST HARNESS
// ============================================================================

fn main() {
    println!("=== Farmer #1 128-bit Shed & 64-bit Marketplace Engine ===");

    // 1. Initialize Farmer #1 Shed
    let mut shed = FarmerShed128::new_farmer_one_default();
    println!("Initial {}", shed);

    // Accept Dogecoin & Bitcoin
    println!("\nConfiguring Farmer #1 to accept Dogecoin (DOGE) & Bitcoin (BTC)...");
    shed.enable_currency(CryptoCurrency::Dogecoin);
    shed.enable_currency(CryptoCurrency::Bitcoin);
    println!("Updated {}", shed);
    println!("Accepts DOGE: {}", shed.accepts_currency(CryptoCurrency::Dogecoin));
    println!("Accepts MRR : {}", shed.accepts_currency(CryptoCurrency::MirrorCoin));
    println!("Accepts SOL : {}", shed.accepts_currency(CryptoCurrency::Solana));

    // 2. Initialize Culture Marketplace
    let market = CultureMarketplace64::new_standard_regime();
    println!("\nMarketplace State: {}", market);
    println!("Wheat Seed Demand    : {}", market.is_in_demand(MarketItem::SeedWheat));
    println!("Egg Food Demand       : {}", market.is_in_demand(MarketItem::FoodEgg));
    println!("Export Wool Demand    : {}", market.is_in_demand(MarketItem::ExportWool));
    println!("Strawberry Demand     : {}", market.is_in_demand(MarketItem::CropStrawberry));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farmer_shed_128_bit_capacity_and_currencies() {
        let mut shed = FarmerShed128::new();
        assert_eq!(FarmerShed128::CAPACITY_BITS, 128);
        assert_eq!(shed.raw, 0);

        shed.enable_currency(CryptoCurrency::MirrorCoin);
        shed.enable_currency(CryptoCurrency::Dogecoin);
        shed.enable_currency(CryptoCurrency::Bitcoin);

        assert!(shed.accepts_currency(CryptoCurrency::MirrorCoin));
        assert!(shed.accepts_currency(CryptoCurrency::Dogecoin));
        assert!(shed.accepts_currency(CryptoCurrency::Bitcoin));
        assert!(!shed.accepts_currency(CryptoCurrency::Monero));

        shed.disable_currency(CryptoCurrency::Bitcoin);
        assert!(!shed.accepts_currency(CryptoCurrency::Bitcoin));
    }

    #[test]
    fn test_culture_marketplace_64_bit_mapping() {
        let mut market = CultureMarketplace64::new();
        assert_eq!(CultureMarketplace64::CAPACITY_BITS, 64);

        market.set_demand(MarketItem::SeedWheat, true);
        market.set_demand(MarketItem::CropCarrot, true);
        market.set_demand(MarketItem::AnimalSheep, true);
        market.set_demand(MarketItem::ExportWool, true);

        assert!(market.is_in_demand(MarketItem::SeedWheat));
        assert!(market.is_in_demand(MarketItem::CropCarrot));
        assert!(market.is_in_demand(MarketItem::AnimalSheep));
        assert!(market.is_in_demand(MarketItem::ExportWool));
        assert!(!market.is_in_demand(MarketItem::SeedMelon));
    }
}
