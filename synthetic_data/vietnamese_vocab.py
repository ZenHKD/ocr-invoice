"""
Vietnamese Vocabulary Generator for Synthetic Data.

Provides diverse Vietnamese product names using:
    - Extensive Vietnamese brand name database (loaded from JSON)
    - Product modifiers (sizes, colors, variants, quality descriptors)
    - Category-specific vocabulary
    - Template-based dynamic name generation
    - Vietnamese number words and quantities
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum


# Get the vocabulary directory path
VOCAB_DIR = Path(__file__).parent / "vocabulary"


def load_vocab_json(filename: str) -> dict:
    """Load vocabulary from a JSON file."""
    filepath = VOCAB_DIR / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class ProductCategory(Enum):
    """Main product categories for vocabulary organization."""
    # Category values MUST match the static catalog in catalog.py for filtering to work
    FOOD = "Thực phẩm"
    BEVERAGES = "Đồ uống"
    PERSONAL_CARE = "Chăm sóc"
    HOUSEHOLD = "Gia dụng"
    ELECTRONICS = "Điện tử"
    CLOTHING = "Quần áo"
    PHARMACY = "Thuốc"
    STATIONERY = "Văn phòng phẩm"
    KITCHEN = "Ẩm thực"
    BABY = "Chăm sóc"  # Baby products under personal care
    SNACKS = "Bánh kẹo"  # Adding snacks category
    BOOKS = "Sách"
    PETS = "Thú cưng"


# =============================================================================
# VOCABULARY LOADING FROM JSON FILES
# =============================================================================

# Load vocabulary from JSON files
VIETNAMESE_BRANDS = load_vocab_json("brands.json")
_products_data = load_vocab_json("products.json")
MODIFIERS = load_vocab_json("modifiers.json")
_common_data = load_vocab_json("common.json")
_abbrev_data = load_vocab_json("abbreviations.json")
_diacritics_data = load_vocab_json("diacritics.json")

# Load new vocabulary files (previously unused)
STORE_PROFILES = load_vocab_json("store_profiles.json")
REGIONS = load_vocab_json("regions.json")
SEASONAL_DATA = load_vocab_json("seasonal.json")

# Build PRODUCT_BASES dict from JSON, mapping ProductCategory enum to lists
PRODUCT_BASES = {}
for cat in ProductCategory:
    cat_name = cat.name  # e.g., "FOOD", "BEVERAGES"
    if cat_name in _products_data and _products_data[cat_name]:
        PRODUCT_BASES[cat] = _products_data[cat_name]
    else:
        PRODUCT_BASES[cat] = []

# Load common vocabulary
UNITS = _common_data.get("units", [])
ADJECTIVES = _common_data.get("adjectives", [])
STORE_PREFIXES = _common_data.get("store_prefixes", [])
STORE_NAMES_VIETNAMESE = _common_data.get("store_names", [])
_numbers_data = _common_data.get("vietnamese_numbers", {})

# Vietnamese number words for quantity expressions
VIETNAMESE_NUMBERS = {
    "cardinal": {int(k): v for k, v in _numbers_data.get("cardinal", {}).items()},
    "ordinal": {int(k): v for k, v in _numbers_data.get("ordinal", {}).items()},
}

# Build ABBREVIATIONS dict from JSON
ABBREVIATIONS = {}
for section in ["location", "invoice_terms", "product_terms"]:
    if section in _abbrev_data:
        ABBREVIATIONS.update(_abbrev_data[section])

# Build DIACRITICS_MAP from JSON
DIACRITICS_MAP = {}
if "lowercase" in _diacritics_data:
    DIACRITICS_MAP.update(_diacritics_data["lowercase"])
if "uppercase" in _diacritics_data:
    DIACRITICS_MAP.update(_diacritics_data["uppercase"])


def remove_vietnamese_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics from text.

    Converts "Cà phê sữa đá" -> "Ca phe sua da"
    Common on old dot matrix printers and some POS systems.
    """
    result = []
    for char in text:
        result.append(DIACRITICS_MAP.get(char, char))
    return ''.join(result)


def apply_random_abbreviation(text: str) -> str:
    """
    Randomly apply abbreviations to Vietnamese text.

    ~30% chance to abbreviate known terms.
    """
    for full_term, abbreviations in ABBREVIATIONS.items():
        if full_term in text and random.random() < 0.3:
            text = text.replace(full_term, random.choice(abbreviations))
    return text


def apply_text_variations(text: str, no_accent_probability: float = 0.2) -> str:
    """
    Apply random text variations to simulate real-world receipts.

    - Random abbreviations
    - Occasional uppercase
    - Sometimes remove diacritics (old printers)
    """
    # Maybe apply abbreviations
    if random.random() < 0.4:
        text = apply_random_abbreviation(text)

    # Maybe remove diacritics (old printers/displays)
    if random.random() < no_accent_probability:
        text = remove_vietnamese_diacritics(text)
        # Old printers often use uppercase
        if random.random() < 0.5:
            text = text.upper()
    return text


# =============================================================================
# VIETNAMESE SYLLABLE STRUCTURE CONSTANTS
# =============================================================================
VIETNAMESE_INITIALS = ["b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"]
VIETNAMESE_VOWELS = ["a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y", "ai", "ay", "ao", "au", "âu", "ây", "eo", "êu", "ia", "iê", "iu", "oa", "oă", "oe", "oi", "ôi", "ơi", "ua", "uâ", "ui", "uy", "ưa", "ưi", "ưu"]
VIETNAMESE_FINALS = ["", "c", "ch", "m", "n", "ng", "nh", "p", "t", "u", "i", "o"]
VIETNAMESE_TONES = ["", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]  # Combining diacritics: grave, acute, hook above, tilde, dot below


@dataclass
class GeneratedProduct:
    """A dynamically generated product."""
    name: str
    category: str
    subcategory: str
    price_vnd: int
    unit: str
    brand: Optional[str] = None
    modifiers: List[str] = None

    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


# =============================================================================
# CONSTANTS
# =============================================================================
vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"
consonants = "bcdđghklmnpqrstvx"
digits = "0123456789"
symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
vietnamese_vocab = vowels + vowels.upper() + consonants + consonants.upper() + digits + symbols
VOCAB = "".join(sorted(list(set(vietnamese_vocab))))


class VietnameseVocabGenerator:
    """
    Generate diverse Vietnamese product names using templates and vocabulary databases.

    This generator can produce 100,000+ unique product name combinations by mixing:
    - Brand names (Vietnamese + International)
    - Product base names by category
    - Size/quantity modifiers
    - Quality descriptors
    - Flavor variants
    - Packaging types

    Vocabulary is loaded from JSON files in the vocabulary/ directory for easy maintenance.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self._generated_names = set()  # Track unique names

    def random_brand(self, prefer_local: bool = False) -> str:
        """Get a random brand name."""
        if prefer_local or random.random() < 0.6:
            return random.choice(VIETNAMESE_BRANDS.get("local", ["Brand"]))
        return random.choice(VIETNAMESE_BRANDS.get("international", ["Brand"]))

    def random_product_base(self, category: ProductCategory = None) -> Tuple[str, ProductCategory]:
        """Get a random product base name."""
        if category is None:
            category = random.choice(list(ProductCategory))

        bases = PRODUCT_BASES.get(category, PRODUCT_BASES.get(ProductCategory.FOOD, ["Sản phẩm"]))
        if not bases:
            bases = ["Sản phẩm"]
        return random.choice(bases), category

    def random_modifier(self, modifier_type: str = None) -> str:
        """Get a random product modifier."""
        if modifier_type is None:
            modifier_type = random.choice(list(MODIFIERS.keys()))

        modifiers = MODIFIERS.get(modifier_type, MODIFIERS.get("size", ["1kg"]))
        if not modifiers:
            modifiers = ["1kg"]
        return random.choice(modifiers)

    def generate_product_name(self,
                               category: ProductCategory = None,
                               include_brand: bool = True,
                               include_modifiers: bool = True,
                               num_modifiers: int = None) -> str:
        """
        Generate a random Vietnamese product name.

        Template patterns:
        1. [Brand] [Product] [Size]           -> "Vinamilk Sữa tươi 1L"
        2. [Product] [Brand] [Quality]        -> "Mì Hảo Hảo tôm chua cay"
        3. [Product] [Flavor] [Size]          -> "Snack vị tôm gói lớn"
        4. [Brand] [Product] [Quality] [Size] -> "Nestlé Cà phê cao cấp 500g"
        5. [Product] [Quality]                -> "Gạo thượng hạng"
        """
        base, cat = self.random_product_base(category)

        parts = []

        # Template selection
        template = random.randint(1, 5)

        if template == 1 and include_brand:
            # [Brand] [Product] [Size]
            parts.append(self.random_brand())
            parts.append(base)
            if include_modifiers:
                parts.append(self.random_modifier("size"))

        elif template == 2 and include_brand:
            # [Product] [Brand] [Flavor/Quality]
            parts.append(base)
            parts.append(self.random_brand())
            if include_modifiers:
                mod_type = random.choice(["flavor", "quality"])
                parts.append(self.random_modifier(mod_type))

        elif template == 3:
            # [Product] [Flavor] [Size]
            parts.append(base)
            if include_modifiers:
                if random.random() < 0.7:
                    parts.append(self.random_modifier("flavor"))
                parts.append(self.random_modifier("size"))

        elif template == 4 and include_brand:
            # [Brand] [Product] [Quality] [Size]
            parts.append(self.random_brand())
            parts.append(base)
            if include_modifiers:
                if random.random() < 0.6:
                    parts.append(self.random_modifier("quality"))
                parts.append(self.random_modifier("size"))

        else:
            # [Product] [Quality/Variant]
            parts.append(base)
            if include_modifiers:
                mod_type = random.choice(["quality", "variant", "flavor"])
                parts.append(self.random_modifier(mod_type))

        name = " ".join(parts)
        return name

    def generate_unique_product_name(self, max_attempts: int = 10, **kwargs) -> str:
        """Generate a unique product name not seen before (for real corpus)."""
        for _ in range(max_attempts):
            name = self.generate_product_name(**kwargs)
            if name not in self._generated_names:
                self._generated_names.add(name)
                return name

        # If we couldn't find unique, add a suffix
        base_name = self.generate_product_name(**kwargs)
        suffix = f" #{random.randint(1, 9999)}"
        unique_name = base_name + suffix
        self._generated_names.add(unique_name)
        return unique_name

    def generate_pure_random(self, min_words: int = 2, max_words: int = 4) -> str:
        """Type 1: Generate completely random characters including special chars (40% mix)."""
        chars = list(VOCAB)
        words = []
        for _ in range(random.randint(min_words, max_words)):
            word_len = random.randint(2, 10)
            word = "".join(random.choice(chars) for _ in range(word_len))
            words.append(word)
        return " ".join(words)

    def generate_pseudo_vietnamese(self, num_syllables: int = None) -> str:
        """Type 2: Generate Vietnamese-structured but meaningless words (30% mix)."""
        if num_syllables is None:
            num_syllables = random.randint(2, 4)

        syllables = []
        for _ in range(num_syllables):
            # Build syllable: [initial] + [vowel] + [final]
            initial = random.choice(VIETNAMESE_INITIALS)
            vowel = random.choice(VIETNAMESE_VOWELS)
            final = random.choice(VIETNAMESE_FINALS)

            # Validation rules
            # Rule 1: Some initials don't combine with some finals (simplified check)
            if initial in ["ch", "nh"] and final in ["p", "t", "c"]:
                final = ""  # These don't combine

            # Add random tone to vowel (last vowel letter gets the tone)
            tone = random.choice(VIETNAMESE_TONES)
            if tone and vowel:
                # Apply tone to last vowel character using unicodedata normalization
                import unicodedata
                vowel_chars = list(vowel)
                # NFC normalization combines base char + combining diacritic
                vowel_chars[-1] = unicodedata.normalize('NFC', vowel_chars[-1] + tone)
                vowel = "".join(vowel_chars)

            syllable = initial + vowel + final

            # Capitalize first syllable sometimes
            if len(syllables) == 0 and random.random() < 0.7:
                syllable = syllable.capitalize()

            syllables.append(syllable)

        return " ".join(syllables)

    def generate_product(self, category: ProductCategory = None, text_type: str = "real") -> GeneratedProduct:
        """
        Generate a complete product.

        Args:
            category: Product category (optional)
            text_type: "pure_random" | "pseudo_vietnamese" | "real"
        """
        if text_type == "pure_random":
            name = self.generate_pure_random()
        elif text_type == "pseudo_vietnamese":
            name = self.generate_pseudo_vietnamese()
        else:
            # Default to real corpus
            name = self.generate_unique_product_name(category=category)

        if category is None:
            category = random.choice(list(ProductCategory))

        # Price ranges by category (VND)
        price_ranges = {
            ProductCategory.FOOD: (5_000, 500_000),
            ProductCategory.BEVERAGES: (5_000, 200_000),
            ProductCategory.PERSONAL_CARE: (15_000, 500_000),
            ProductCategory.HOUSEHOLD: (10_000, 300_000),
            ProductCategory.ELECTRONICS: (20_000, 2_000_000),
            ProductCategory.PHARMACY: (10_000, 800_000),
            ProductCategory.STATIONERY: (2_000, 100_000),
            ProductCategory.BABY: (50_000, 1_000_000),
            ProductCategory.CLOTHING: (50_000, 2_000_000),
            ProductCategory.KITCHEN: (20_000, 500_000),
        }

        min_price, max_price = price_ranges.get(category, (10_000, 200_000))

        # Generate realistic price (rounded to nice values)
        price = random.randint(min_price, max_price)
        if price < 10_000:
            price = round(price / 500) * 500
        elif price < 100_000:
            price = round(price / 1_000) * 1_000
        else:
            price = round(price / 5_000) * 5_000

        unit = random.choice(UNITS) if UNITS else "cái"

        return GeneratedProduct(
            name=name,
            category=category.value,
            subcategory=category.name.lower(),
            price_vnd=price,
            unit=unit,
        )

    def generate_products(self, count: int,
                          category: ProductCategory = None) -> List[GeneratedProduct]:
        """Generate multiple unique products."""
        products = []
        for _ in range(count):
            product = self.generate_product(category)
            products.append(product)
        return products

    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about available vocabulary."""
        return {
            "local_brands": len(VIETNAMESE_BRANDS.get("local", [])),
            "international_brands": len(VIETNAMESE_BRANDS.get("international", [])),
            "product_categories": len(ProductCategory),
            "product_bases": sum(len(v) for v in PRODUCT_BASES.values()),
            "size_modifiers": len(MODIFIERS.get("size", [])),
            "quality_modifiers": len(MODIFIERS.get("quality", [])),
            "flavor_modifiers": len(MODIFIERS.get("flavor", [])),
            "variant_modifiers": len(MODIFIERS.get("variant", [])),
            "units": len(UNITS),
            "estimated_unique_combinations": self._estimate_combinations(),
        }

    def _estimate_combinations(self) -> int:
        """Estimate number of unique product name combinations possible."""
        brands = len(VIETNAMESE_BRANDS.get("local", [])) + len(VIETNAMESE_BRANDS.get("international", []))
        bases = sum(len(v) for v in PRODUCT_BASES.values())
        modifiers = sum(len(v) for v in MODIFIERS.values())

        # Conservative estimate: brand × base × 2 modifiers
        return brands * bases * modifiers * modifiers // 4

    def reset_uniqueness_tracker(self):
        """Reset the unique names tracker."""
        self._generated_names.clear()

    def random_store_name(self) -> str:
        """Generate a random Vietnamese store name."""
        template = random.randint(1, 4)

        store_names = STORE_NAMES_VIETNAMESE if STORE_NAMES_VIETNAMESE else ["Cửa hàng"]
        store_prefixes = STORE_PREFIXES if STORE_PREFIXES else ["Siêu thị"]

        if template == 1:
            # [Prefix] [Vietnamese Name]
            prefix = random.choice(store_prefixes)
            name = random.choice(store_names)
            return f"{prefix} {name}"
        elif template == 2:
            # [Vietnamese Name] Mart/Store
            name = random.choice(store_names)
            suffix = random.choice(["Mart", "Store", "Shop", "Market"])
            return f"{name} {suffix}"
        elif template == 3:
            # [Brand] + location
            name = random.choice(store_names)
            location = random.choice(["Quận 1", "Quận 3", "Quận 7", "Bình Thạnh",
                                      "Cầu Giấy", "Đống Đa", "Hai Bà Trưng"])
            return f"{name} - {location}"
        else:
            # Simple name
            return random.choice(store_names)


# Test function
if __name__ == "__main__":
    gen = VietnameseVocabGenerator(seed=42)

    print("=== Vietnamese Vocabulary Generator ===\n")

    # Show stats
    stats = gen.get_vocabulary_stats()
    print("Vocabulary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    print("\n=== Sample Generated Products ===\n")

    # Generate samples for each category
    for category in list(ProductCategory)[:5]:
        print(f"\n{category.name}:")
        for _ in range(3):
            product = gen.generate_product(category)
            print(f"  - {product.name} ({product.price_vnd:,}₫/{product.unit})")

    print("\n=== Sample Store Names ===\n")
    for _ in range(5):
        print(f"  - {gen.random_store_name()}")

    # Test uniqueness
    print("\n=== Uniqueness Test ===\n")
    gen.reset_uniqueness_tracker()
    products = gen.generate_products(1000)
    unique_names = len(set(p.name for p in products))
    print(f"Generated 1000 products, {unique_names} unique names ({unique_names/10:.1f}%)")
