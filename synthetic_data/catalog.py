"""
Product Catalog - Pure Dynamic Generation

Provides:
    - Dynamic product generation using Vietnamese vocabulary templates
    - Store type and region-aware product selection
    - Price calculation with variance and store modifiers
    - 165M+ unique product name combinations
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Region(Enum):
    NORTH = "north"
    CENTRAL = "central"
    SOUTH = "south"
    ALL = "all"


class StoreType(Enum):
    SUPERMARKET = "supermarket"
    CONVENIENCE = "convenience"
    TRADITIONAL_MARKET = "traditional_market"
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAKERY = "bakery"
    ELECTRONICS = "electronics"
    PHARMACY = "pharmacy"
    HARDWARE = "hardware"
    CLOTHING = "clothing"


@dataclass
class Product:
    """A product with name, pricing, and metadata."""
    name: str
    category: str
    subcategory: str
    base_price_vnd: int
    price_variance: float = 0.15
    unit: str = "cái"
    region_availability: Region = Region.ALL
    seasonal: Optional[List[int]] = None
    store_types: List[StoreType] = field(default_factory=lambda: [StoreType.SUPERMARKET])
    common_qty_range: Tuple[int, int] = (1, 3)

    def get_price(self, currency: str = "VND", store_type: StoreType = StoreType.SUPERMARKET) -> float:
        """Get price with variance and store-type adjustment."""
        variance = random.uniform(1 - self.price_variance, 1 + self.price_variance)

        modifiers = {
            StoreType.SUPERMARKET: 1.0,
            StoreType.CONVENIENCE: 1.15,
            StoreType.TRADITIONAL_MARKET: 0.85,
            StoreType.RESTAURANT: 2.5,
            StoreType.CAFE: 2.0,
            StoreType.BAKERY: 1.1,
            StoreType.ELECTRONICS: 1.0,
            StoreType.PHARMACY: 1.0,
            StoreType.HARDWARE: 1.0,
            StoreType.CLOTHING: 1.0,
        }

        price = self.base_price_vnd * variance * modifiers.get(store_type, 1.0)

        if currency == "VND":
            if price < 10000:
                return int(round(price / 500) * 500)
            elif price < 100000:
                return int(round(price / 1000) * 1000)
            else:
                return int(round(price / 5000) * 5000)
        elif currency == "USD":
            return round(price / 24500, 2)
        elif currency == "EUR":
            return round(price / 27000, 2)
        return price


class ProductCatalog:
    """
    Dynamic Vietnamese product catalog.
    
    Generates products on-the-fly using vocabulary templates for maximum diversity.
    Supports 165M+ unique product name combinations.
    """

    def __init__(self, seed: Optional[int] = None):
        from .vietnamese_vocab import VietnameseVocabGenerator, ProductCategory
        
        self.vocab_gen = VietnameseVocabGenerator(seed=seed)
        self.ProductCategory = ProductCategory
        
        # Map store types to vocabulary categories
        self._store_to_categories = {
            StoreType.SUPERMARKET: [
                ProductCategory.FOOD, ProductCategory.BEVERAGES,
                ProductCategory.PERSONAL_CARE, ProductCategory.HOUSEHOLD,
                ProductCategory.SNACKS
            ],
            StoreType.CONVENIENCE: [
                ProductCategory.BEVERAGES, ProductCategory.FOOD,
                ProductCategory.PERSONAL_CARE, ProductCategory.SNACKS
            ],
            StoreType.TRADITIONAL_MARKET: [
                ProductCategory.FOOD, ProductCategory.HOUSEHOLD
            ],
            StoreType.RESTAURANT: [
                ProductCategory.FOOD, ProductCategory.BEVERAGES,
                ProductCategory.KITCHEN
            ],
            StoreType.CAFE: [
                ProductCategory.BEVERAGES, ProductCategory.FOOD,
                ProductCategory.SNACKS
            ],
            StoreType.BAKERY: [
                ProductCategory.FOOD, ProductCategory.SNACKS
            ],
            StoreType.ELECTRONICS: [
                ProductCategory.ELECTRONICS
            ],
            StoreType.PHARMACY: [
                ProductCategory.PHARMACY, ProductCategory.PERSONAL_CARE
            ],
            StoreType.HARDWARE: [
                ProductCategory.HOUSEHOLD, ProductCategory.ELECTRONICS
            ],
            StoreType.CLOTHING: [
                ProductCategory.CLOTHING
            ],
        }

    def generate_product(self, store_type: StoreType = None) -> Product:
        """Generate a single random product."""
        if store_type and store_type in self._store_to_categories:
            categories = self._store_to_categories[store_type]
            category = random.choice(categories)
        else:
            category = random.choice(list(self.ProductCategory))

        gen_product = self.vocab_gen.generate_product(category)

        return Product(
            name=gen_product.name,
            category=gen_product.category,
            subcategory=gen_product.subcategory,
            base_price_vnd=gen_product.price_vnd,
            unit=gen_product.unit,
            store_types=[store_type] if store_type else [StoreType.SUPERMARKET],
            common_qty_range=(1, random.randint(2, 5)),
        )

    def get_products_for_store(self, store_type: StoreType,
                               region: Region = Region.ALL,
                               month: Optional[int] = None,
                               count: int = 50) -> List[Product]:
        """Generate products for a store type."""
        return [self.generate_product(store_type) for _ in range(count)]

    def get_bundle(self, count: int = 3, store_type: StoreType = None) -> List[Product]:
        """Get a random bundle of products."""
        return [self.generate_product(store_type) for _ in range(count)]

    def get_related_products(self, product: Product, limit: int = 5,
                            store_type: StoreType = None) -> List[Product]:
        """Get related products (same category)."""
        category = None
        for cat in self.ProductCategory:
            if cat.value == product.category:
                category = cat
                break

        products = []
        for _ in range(limit):
            gen_product = self.vocab_gen.generate_product(category)
            products.append(Product(
                name=gen_product.name,
                category=gen_product.category,
                subcategory=gen_product.subcategory,
                base_price_vnd=gen_product.price_vnd,
                unit=gen_product.unit,
                store_types=[store_type] if store_type else [StoreType.SUPERMARKET],
            ))
        return products

    def get_vocabulary_stats(self) -> Dict[str, any]:
        """Get vocabulary statistics."""
        return self.vocab_gen.get_vocabulary_stats()

    def reset(self):
        """Reset uniqueness tracking for fresh generation."""
        self.vocab_gen.reset_uniqueness_tracker()
