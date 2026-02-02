"""
Product Catalog - Pure Dynamic Generation

Provides:
    - Dynamic product generation using Vietnamese vocabulary templates
    - Store type and region-aware product selection
    - Price calculation with variance and store modifiers
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
    """

    def __init__(self, seed: Optional[int] = None):
        from .vietnamese_vocab import VietnameseVocabGenerator, ProductCategory

        self.vocab_gen = VietnameseVocabGenerator(seed=seed)
        self.ProductCategory = ProductCategory

        from .vietnamese_vocab import STORE_PROFILES

        # KEY MAPPING: StoreType (Enum) -> store_profiles.json (Keys)
        self._store_type_map = {
            StoreType.SUPERMARKET: "supermarket",
            StoreType.CONVENIENCE: "convenience_store", # Key mismatch handling
            StoreType.TRADITIONAL_MARKET: "traditional_market", # Assuming key exists or default
            StoreType.RESTAURANT: "restaurant",
            StoreType.CAFE: "cafe",
            StoreType.BAKERY: "bakery",
            StoreType.ELECTRONICS: "electronics", # Mapped if exists, else fallback
            StoreType.PHARMACY: "pharmacy",
            StoreType.HARDWARE: "hardware", # Might need fallback
            StoreType.CLOTHING: "clothing", # Need fallback check
        }

        # Build _store_to_categories dynamically from JSON
        self._store_to_categories = {}

        for store_enum, profile_key in self._store_type_map.items():
            profile = STORE_PROFILES.get(profile_key)
            if profile and "common_categories" in profile:
                # Convert string list "FOOD" -> ProductCategory.FOOD
                categories = []
                for cat_str in profile["common_categories"]:
                    try:
                        # Handle potential naming mismatches if JSON has slightly different names
                        # Enum names are uppercase: FOOD, BEVERAGES...
                        cat_enum = ProductCategory[cat_str]
                        categories.append(cat_enum)
                    except KeyError:
                        print(f"Warning: Category {cat_str} in profile {profile_key} not found in ProductCategory enum")
                        continue

                if categories:
                    self._store_to_categories[store_enum] = categories

        # Fallback for stores not in JSON or missing categories (keep some defaults just in case)
        if StoreType.SUPERMARKET not in self._store_to_categories:
             self._store_to_categories[StoreType.SUPERMARKET] = [
                 ProductCategory.FOOD, ProductCategory.BEVERAGES, ProductCategory.HOUSEHOLD
             ]

    def generate_product(self, store_type: StoreType = None, text_type: str = None) -> Product:
        """Generate a single random product.
        
        Args:
            store_type: Type of store for category filtering
            text_type: Text generation mode. If None, randomly selects:
                - 40% "pure_random" (random characters)
                - 30% "pseudo_vietnamese" (fake Vietnamese-like words)
                - 30% "real" (real vocabulary)
        """
        # Randomly select text type if not specified
        if text_type is None:
            rand = random.random()
            if rand < 0.4:
                text_type = "pure_random"
            elif rand < 0.7:  # 0.4 to 0.7 = 30%
                text_type = "pseudo_vietnamese"
            else:  # 0.7 to 1.0 = 30%
                text_type = "real"
        
        if store_type and store_type in self._store_to_categories:
            categories = self._store_to_categories[store_type]
            category = random.choice(categories)
        else:
            category = random.choice(list(self.ProductCategory))

        gen_product = self.vocab_gen.generate_product(category, text_type=text_type)

        return Product(
            name=gen_product.name,
            category=gen_product.category,
            subcategory=gen_product.subcategory,
            base_price_vnd=gen_product.price_vnd,
            unit=gen_product.unit,
            store_types=[store_type] if store_type else [StoreType.SUPERMARKET],
            common_qty_range=(1, random.randint(2, 5)),
        )

    def get_seasonal_products(self, month: int, limit: int = 5) -> List[Product]:
        """Get seasonal specific products."""
        from .vietnamese_vocab import SEASONAL_DATA

        products = []
        if not SEASONAL_DATA:
            return products

        # Find active seasons for this month
        active_seasons = []
        for season_key, data in SEASONAL_DATA.items():
            if month in data.get("months", []):
                active_seasons.append(data)

        if not active_seasons:
            return products

        # Select products from active seasons
        for _ in range(limit):
            season = random.choice(active_seasons)
            if "products" in season and season["products"]:
                prod_name = random.choice(season["products"])
                products.append(Product(
                    name=prod_name,
                    category="SEASONAL",
                    subcategory="seasonal",
                    base_price_vnd=random.randint(50000, 500000), # Generic price range
                    unit="hộp" # Generic unit
                ))

        return products

    def get_products_for_store(self, store_type: StoreType,
                               region: Region = Region.ALL,
                               month: Optional[int] = None,
                               count: int = 50,
                               text_type: str = None) -> List[Product]:
        """Generate products for a store type.
        
        Args:
            text_type: If None, each product randomly selects text generation mode
        """
        products = [self.generate_product(store_type, text_type=text_type) for _ in range(count)]

        # Inject seasonal products if month provided
        if month:
            seasonal = self.get_seasonal_products(month, limit=max(1, count // 10))
            products.extend(seasonal)

        return products

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

    def generate_random_products(self, count: int = 50) -> List[Product]:
        """Generate random products across all categories with mixed text types."""
        return [self.generate_product(store_type=None, text_type=None) for _ in range(count)]

    def get_vocabulary_stats(self) -> Dict[str, any]:
        """Get vocabulary statistics."""
        return self.vocab_gen.get_vocabulary_stats()

    def reset(self):
        """Reset uniqueness tracking for fresh generation."""
        self.vocab_gen.reset_uniqueness_tracker()
