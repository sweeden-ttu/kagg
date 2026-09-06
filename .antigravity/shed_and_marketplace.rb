# frozen_string_literal: true

# ============================================================================
# 1. 128-BIT FARMER #1 SHED (Ruby Implementation)
# ============================================================================

module CryptoCurrency
  MIRROR_COIN      = 0 # MRR
  BITCOIN          = 1 # BTC
  ETHEREUM         = 2 # ETH
  DOGECOIN         = 3 # DOGE
  SOLANA           = 4 # SOL
  MONERO           = 5 # XMR
  USDC             = 6 # USDC
  USDT             = 7 # USDT
  FELLOWSHIP_TOKEN = 8 # 888 Stake Token
  PROOF_OF_YIELD   = 9 # On-chain harvest receipt

  ALL = [
    MIRROR_COIN, BITCOIN, ETHEREUM, DOGECOIN, SOLANA,
    MONERO, USDC, USDT, FELLOWSHIP_TOKEN, PROOF_OF_YIELD
  ].freeze
end

class FarmerShed128
  CAPACITY_BITS = 128

  attr_accessor :raw

  def initialize(raw = 0)
    @raw = Integer(raw) & ((1 << CAPACITY_BITS) - 1)
  end

  def self.new_farmer_one_default
    shed = new
    shed.enable_currency(CryptoCurrency::MIRROR_COIN)
    shed.enable_currency(CryptoCurrency::FELLOWSHIP_TOKEN)
    shed.raw |= (1 << 64) # Storage slot 0 active
    shed
  end

  def enable_currency(currency_bit)
    @raw |= (1 << Integer(currency_bit))
  end

  def disable_currency(currency_bit)
    @raw &= ~(1 << Integer(currency_bit))
  end

  def accepts_currency?(currency_bit)
    (@raw & (1 << Integer(currency_bit))) != 0
  end

  def list_accepted_currencies
    CryptoCurrency::ALL.select { |c| accepts_currency?(c) }
  end

  def to_hex
    format("0x%032X", @raw)
  end
end

# ============================================================================
# 2. 64-BIT CULTURE MARKETPLACE (Ruby Implementation)
# ============================================================================

module MarketItem
  # Seeds (Bits 0..7)
  SEED_WHEAT         = 0
  SEED_CARROT        = 1
  SEED_TOMATO        = 2
  SEED_STRAWBERRY    = 3
  SEED_MELON         = 4

  # Foods & Crops (Bits 8..23)
  CROP_WHEAT         = 8
  CROP_CARROT        = 9
  CROP_TOMATO        = 10
  CROP_STRAWBERRY    = 11
  CROP_MELON         = 12
  FOOD_EGG           = 13
  FOOD_MILK          = 14
  PRODUCT_FERTILIZER = 15
  FOOD_FLOUR         = 16
  FOOD_BREAD         = 17

  # Animals (Bits 24..31)
  ANIMAL_CHICKEN     = 24
  ANIMAL_COW         = 25
  ANIMAL_SHEEP       = 26

  # Export Goods (Bits 32..47)
  EXPORT_WOOL        = 32
  EXPORT_BULK_GRAIN  = 33
  EXPORT_CHEESE      = 34
  EXPORT_PRESERVES   = 35

  # Macro Demands (Bits 48..63)
  DOMESTIC_MAJORITY  = 48
  EXPORT_MAJORITY    = 49
  HARVEST_SURGE      = 50
end

class CultureMarketplace64
  CAPACITY_BITS = 64

  attr_accessor :raw

  def initialize(raw = 0)
    @raw = Integer(raw) & ((1 << CAPACITY_BITS) - 1)
  end

  def self.new_standard_regime
    market = new
    market.set_demand(MarketItem::SEED_WHEAT, true)
    market.set_demand(MarketItem::SEED_CARROT, true)
    market.set_demand(MarketItem::CROP_WHEAT, true)
    market.set_demand(MarketItem::CROP_CARROT, true)
    market.set_demand(MarketItem::FOOD_EGG, true)
    market.set_demand(MarketItem::FOOD_MILK, true)
    market.set_demand(MarketItem::ANIMAL_CHICKEN, true)
    market.set_demand(MarketItem::ANIMAL_COW, true)
    market.set_demand(MarketItem::EXPORT_WOOL, true)
    market.set_demand(MarketItem::EXPORT_BULK_GRAIN, true)
    market.set_demand(MarketItem::DOMESTIC_MAJORITY, true)
    market.set_demand(MarketItem::EXPORT_MAJORITY, true)
    market
  end

  def set_demand(item_bit, active)
    if active
      @raw |= (1 << Integer(item_bit))
    else
      @raw &= ~(1 << Integer(item_bit))
    end
  end

  def in_demand?(item_bit)
    (@raw & (1 << Integer(item_bit))) != 0
  end

  def to_hex
    format("0x%016X", @raw)
  end
end

# ============================================================================
# Verification
# ============================================================================

if __FILE__ == $PROGRAM_NAME
  puts "=== Ruby: Farmer #1 128-bit Shed & 64-bit Marketplace ==="

  shed = FarmerShed128.new_farmer_one_default
  shed.enable_currency(CryptoCurrency::DOGECOIN)
  shed.enable_currency(CryptoCurrency::BITCOIN)

  puts "Shed Hex: #{shed.to_hex}"
  puts "Accepts DOGE: #{shed.accepts_currency?(CryptoCurrency::DOGECOIN)}"
  puts "Accepts MRR : #{shed.accepts_currency?(CryptoCurrency::MIRROR_COIN)}"
  puts "Accepts SOL : #{shed.accepts_currency?(CryptoCurrency::SOLANA)}"

  market = CultureMarketplace64.new_standard_regime
  puts "\nMarketplace Hex: #{market.to_hex}"
  puts "Wheat Seed Demand : #{market.in_demand?(MarketItem::SEED_WHEAT)}"
  puts "Egg Food Demand    : #{market.in_demand?(MarketItem::FOOD_EGG)}"
  puts "Export Wool Demand : #{market.in_demand?(MarketItem::EXPORT_WOOL)}"
  puts "Strawberry Demand  : #{market.in_demand?(MarketItem::CROP_STRAWBERRY)}"
end
