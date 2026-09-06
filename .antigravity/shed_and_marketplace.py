#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Farmer #1 128-bit Shed & 64-bit Culture Marketplace (Python Implementation)
"""

from enum import IntEnum
from typing import List


# ============================================================================
# 1. 128-BIT FARMER #1 SHED (Python Implementation)
# ============================================================================

class CryptoCurrency(IntEnum):
    MIRROR_COIN = 0  # MRR (Native game currency)
    BITCOIN = 1  # BTC
    ETHEREUM = 2  # ETH
    DOGECOIN = 3  # DOGE
    SOLANA = 4  # SOL
    MONERO = 5  # XMR
    USDC = 6  # USDC
    USDT = 7  # USDT
    FELLOWSHIP_TOKEN = 8  # Sub-agent 888 Stake Token
    PROOF_OF_YIELD = 9  # On-chain harvest receipt


class FarmerShed128:
    CAPACITY_BITS = 128

    def __init__(self, raw: int = 0):
        self.raw = int(raw) & ((1 << self.CAPACITY_BITS) - 1)

    @classmethod
    def new_farmer_one_default(cls) -> "FarmerShed128":
        shed = cls()
        shed.enable_currency(CryptoCurrency.MIRROR_COIN)
        shed.enable_currency(CryptoCurrency.FELLOWSHIP_TOKEN)
        shed.raw |= 1 << 64  # Storage partition 0 active
        return shed

    def enable_currency(self, currency: CryptoCurrency):
        self.raw |= 1 << int(currency)

    def disable_currency(self, currency: CryptoCurrency):
        self.raw &= ~(1 << int(currency))

    def accepts_currency(self, currency: CryptoCurrency) -> bool:
        return bool(self.raw & (1 << int(currency)))

    def list_accepted_currencies(self) -> List[CryptoCurrency]:
        return [c for c in CryptoCurrency if self.accepts_currency(c)]

    def to_hex(self) -> str:
        return f"0x{self.raw:032X}"


# ============================================================================
# 2. 64-BIT CULTURE MARKETPLACE (Python Implementation)
# ============================================================================

class MarketItem(IntEnum):
    # Seeds (Bits 0..7)
    SEED_WHEAT = 0
    SEED_CARROT = 1
    SEED_TOMATO = 2
    SEED_STRAWBERRY = 3
    SEED_MELON = 4

    # Foods & Crops (Bits 8..23)
    CROP_WHEAT = 8
    CROP_CARROT = 9
    CROP_TOMATO = 10
    CROP_STRAWBERRY = 11
    CROP_MELON = 12
    FOOD_EGG = 13
    FOOD_MILK = 14
    PRODUCT_FERTILIZER = 15
    FOOD_FLOUR = 16
    FOOD_BREAD = 17

    # Animals (Bits 24..31)
    ANIMAL_CHICKEN = 24
    ANIMAL_COW = 25
    ANIMAL_SHEEP = 26

    # Export Commodities (Bits 32..47)
    EXPORT_WOOL = 32
    EXPORT_BULK_GRAIN = 33
    EXPORT_CHEESE = 34
    EXPORT_PRESERVES = 35

    # Macro Demands (Bits 48..63)
    DOMESTIC_MAJORITY = 48
    EXPORT_MAJORITY = 49
    HARVEST_SURGE = 50


class CultureMarketplace64:
    CAPACITY_BITS = 64

    def __init__(self, raw: int = 0):
        self.raw = int(raw) & ((1 << self.CAPACITY_BITS) - 1)

    @classmethod
    def new_standard_regime(cls) -> "CultureMarketplace64":
        market = cls()
        market.set_demand(MarketItem.SEED_WHEAT, True)
        market.set_demand(MarketItem.SEED_CARROT, True)
        market.set_demand(MarketItem.CROP_WHEAT, True)
        market.set_demand(MarketItem.CROP_CARROT, True)
        market.set_demand(MarketItem.FOOD_EGG, True)
        market.set_demand(MarketItem.FOOD_MILK, True)
        market.set_demand(MarketItem.ANIMAL_CHICKEN, True)
        market.set_demand(MarketItem.ANIMAL_COW, True)
        market.set_demand(MarketItem.EXPORT_WOOL, True)
        market.set_demand(MarketItem.EXPORT_BULK_GRAIN, True)
        market.set_demand(MarketItem.DOMESTIC_MAJORITY, True)
        market.set_demand(MarketItem.EXPORT_MAJORITY, True)
        return market

    def set_demand(self, item: MarketItem, active: bool):
        if active:
            self.raw |= 1 << int(item)
        else:
            self.raw &= ~(1 << int(item))

    def is_in_demand(self, item: MarketItem) -> bool:
        return bool(self.raw & (1 << int(item)))

    def to_hex(self) -> str:
        return f"0x{self.raw:016X}"


# ============================================================================
# Verification
# ============================================================================

if __name__ == "__main__":
    print("=== Python: Farmer #1 128-bit Shed & 64-bit Marketplace ===")

    shed = FarmerShed128.new_farmer_one_default()
    shed.enable_currency(CryptoCurrency.DOGECOIN)
    shed.enable_currency(CryptoCurrency.BITCOIN)

    print(f"Shed Hex: {shed.to_hex()}")
    print(f"Accepts DOGE: {shed.accepts_currency(CryptoCurrency.DOGECOIN)}")
    print(f"Accepts MRR : {shed.accepts_currency(CryptoCurrency.MIRROR_COIN)}")
    print(f"Accepts SOL : {shed.accepts_currency(CryptoCurrency.SOLANA)}")

    market = CultureMarketplace64.new_standard_regime()
    print(f"\nMarketplace Hex: {market.to_hex()}")
    print(f"Wheat Seed Demand : {market.is_in_demand(MarketItem.SEED_WHEAT)}")
    print(f"Egg Food Demand    : {market.is_in_demand(MarketItem.FOOD_EGG)}")
    print(f"Export Wool Demand : {market.is_in_demand(MarketItem.EXPORT_WOOL)}")
    print(f"Strawberry Demand  : {market.is_in_demand(MarketItem.CROP_STRAWBERRY)}")
