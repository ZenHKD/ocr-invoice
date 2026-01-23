"""
Purchase Behavior Patterns for realistic shopping simulations.

Provides:
    - Customer personas with distinct shopping patterns
    - Time-of-day purchase variations
    - Seasonal shopping behaviors
    - Bundle and cross-selling patterns
    - Regional preferences
    - Store-type specific behaviors
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import datetime

from .catalog import Product, ProductCatalog, StoreType, Region


class CustomerType(Enum):
    STUDENT = "student"
    OFFICE_WORKER = "office_worker"
    HOMEMAKER = "homemaker"
    ELDERLY = "elderly"
    TOURIST = "tourist"
    RESTAURANT_BUYER = "restaurant_buyer"
    SMALL_BUSINESS = "small_business"
    FAMILY = "family"
    SINGLE_YOUNG = "single_young"


class TimeOfDay(Enum):
    EARLY_MORNING = "early_morning"   # 5-7am
    MORNING = "morning"               # 7-11am
    LUNCH = "lunch"                   # 11am-2pm
    AFTERNOON = "afternoon"           # 2-5pm
    EVENING = "evening"               # 5-8pm
    NIGHT = "night"                   # 8-11pm
    LATE_NIGHT = "late_night"         # 11pm-5am


class PurchaseOccasion(Enum):
    DAILY_ESSENTIALS = "daily_essentials"
    WEEKLY_GROCERIES = "weekly_groceries"
    PARTY_PREP = "party_prep"
    QUICK_SNACK = "quick_snack"
    MEAL_PURCHASE = "meal_purchase"
    GIFT_BUYING = "gift_buying"
    BULK_BUYING = "bulk_buying"
    IMPULSE = "impulse"
    SPECIAL_OCCASION = "special_occasion"


@dataclass
class CustomerProfile:
    """Profile defining customer shopping behavior."""
    customer_type: CustomerType
    budget_range: Tuple[int, int]  # VND
    typical_items: Tuple[int, int]  # min, max items per transaction
    preferred_stores: List[StoreType]
    preferred_categories: List[str]
    avoided_categories: List[str]
    purchase_occasions: List[PurchaseOccasion]
    typical_times: List[TimeOfDay]
    price_sensitivity: float = 0.5  # 0 = price insensitive, 1 = very price sensitive
    brand_loyalty: float = 0.5  # 0 = switches brands, 1 = loyal
    impulse_tendency: float = 0.3  # probability of adding impulse items


@dataclass
class PurchaseContext:
    """Context for a specific purchase transaction."""
    customer_profile: CustomerProfile
    store_type: StoreType
    occasion: PurchaseOccasion
    time_of_day: TimeOfDay
    date: datetime.date
    region: Region = Region.SOUTH
    is_holiday: bool = False
    is_weekend: bool = False


class PurchaseBehavior:
    """Generate realistic purchase patterns."""

    def __init__(self, catalog: ProductCatalog = None):
        # All catalogs are now fully dynamic
        self.catalog = catalog if catalog else ProductCatalog()
        self.customer_profiles = self._define_profiles()

    def _define_profiles(self) -> Dict[CustomerType, CustomerProfile]:
        """Define customer profiles with shopping behaviors."""
        return {
            CustomerType.STUDENT: CustomerProfile(
                customer_type=CustomerType.STUDENT,
                budget_range=(30_000, 150_000),
                typical_items=(1, 5),
                preferred_stores=[StoreType.CONVENIENCE, StoreType.CAFE, StoreType.TRADITIONAL_MARKET],
                preferred_categories=["Đồ uống", "Bánh kẹo", "Ẩm thực"],
                avoided_categories=["Thuốc", "Gia dụng"],
                purchase_occasions=[PurchaseOccasion.QUICK_SNACK, PurchaseOccasion.MEAL_PURCHASE, PurchaseOccasion.IMPULSE],
                typical_times=[TimeOfDay.MORNING, TimeOfDay.LUNCH, TimeOfDay.EVENING, TimeOfDay.LATE_NIGHT],
                price_sensitivity=0.8,
                brand_loyalty=0.3,
                impulse_tendency=0.5,
            ),

            CustomerType.OFFICE_WORKER: CustomerProfile(
                customer_type=CustomerType.OFFICE_WORKER,
                budget_range=(50_000, 500_000),
                typical_items=(1, 8),
                preferred_stores=[StoreType.CONVENIENCE, StoreType.CAFE, StoreType.SUPERMARKET, StoreType.RESTAURANT],
                preferred_categories=["Đồ uống", "Ẩm thực", "Chăm sóc"],
                avoided_categories=[],
                purchase_occasions=[PurchaseOccasion.MEAL_PURCHASE, PurchaseOccasion.QUICK_SNACK, PurchaseOccasion.DAILY_ESSENTIALS],
                typical_times=[TimeOfDay.MORNING, TimeOfDay.LUNCH, TimeOfDay.EVENING],
                price_sensitivity=0.4,
                brand_loyalty=0.6,
                impulse_tendency=0.4,
            ),

            CustomerType.HOMEMAKER: CustomerProfile(
                customer_type=CustomerType.HOMEMAKER,
                budget_range=(200_000, 2_000_000),
                typical_items=(5, 20),
                preferred_stores=[StoreType.SUPERMARKET, StoreType.TRADITIONAL_MARKET],
                preferred_categories=["Thực phẩm", "Gia dụng", "Chăm sóc"],
                avoided_categories=["Điện tử"],
                purchase_occasions=[PurchaseOccasion.WEEKLY_GROCERIES, PurchaseOccasion.DAILY_ESSENTIALS, PurchaseOccasion.SPECIAL_OCCASION],
                typical_times=[TimeOfDay.EARLY_MORNING, TimeOfDay.MORNING, TimeOfDay.AFTERNOON],
                price_sensitivity=0.7,
                brand_loyalty=0.7,
                impulse_tendency=0.3,
            ),

            CustomerType.ELDERLY: CustomerProfile(
                customer_type=CustomerType.ELDERLY,
                budget_range=(100_000, 500_000),
                typical_items=(2, 10),
                preferred_stores=[StoreType.TRADITIONAL_MARKET, StoreType.PHARMACY, StoreType.SUPERMARKET],
                preferred_categories=["Thực phẩm", "Thuốc", "Gia dụng"],
                avoided_categories=["Điện tử", "Bánh kẹo"],
                purchase_occasions=[PurchaseOccasion.DAILY_ESSENTIALS, PurchaseOccasion.WEEKLY_GROCERIES],
                typical_times=[TimeOfDay.EARLY_MORNING, TimeOfDay.MORNING],
                price_sensitivity=0.8,
                brand_loyalty=0.9,
                impulse_tendency=0.1,
            ),

            CustomerType.TOURIST: CustomerProfile(
                customer_type=CustomerType.TOURIST,
                budget_range=(100_000, 1_000_000),
                typical_items=(2, 8),
                preferred_stores=[StoreType.CONVENIENCE, StoreType.RESTAURANT, StoreType.CAFE],
                preferred_categories=["Đồ uống", "Ẩm thực", "Bánh kẹo"],
                avoided_categories=["Gia dụng", "Phần cứng"],
                purchase_occasions=[PurchaseOccasion.MEAL_PURCHASE, PurchaseOccasion.GIFT_BUYING, PurchaseOccasion.IMPULSE],
                typical_times=[TimeOfDay.MORNING, TimeOfDay.LUNCH, TimeOfDay.AFTERNOON, TimeOfDay.EVENING],
                price_sensitivity=0.3,
                brand_loyalty=0.1,
                impulse_tendency=0.7,
            ),

            CustomerType.RESTAURANT_BUYER: CustomerProfile(
                customer_type=CustomerType.RESTAURANT_BUYER,
                budget_range=(500_000, 10_000_000),
                typical_items=(10, 50),
                preferred_stores=[StoreType.TRADITIONAL_MARKET, StoreType.SUPERMARKET],
                preferred_categories=["Thực phẩm", "Đồ uống"],
                avoided_categories=["Chăm sóc", "Điện tử", "Thuốc"],
                purchase_occasions=[PurchaseOccasion.BULK_BUYING, PurchaseOccasion.DAILY_ESSENTIALS],
                typical_times=[TimeOfDay.EARLY_MORNING, TimeOfDay.MORNING],
                price_sensitivity=0.6,
                brand_loyalty=0.5,
                impulse_tendency=0.1,
            ),

            CustomerType.SMALL_BUSINESS: CustomerProfile(
                customer_type=CustomerType.SMALL_BUSINESS,
                budget_range=(300_000, 5_000_000),
                typical_items=(5, 30),
                preferred_stores=[StoreType.SUPERMARKET, StoreType.HARDWARE, StoreType.ELECTRONICS],
                preferred_categories=["Gia dụng", "Điện tử", "Phần cứng"],
                avoided_categories=[],
                purchase_occasions=[PurchaseOccasion.BULK_BUYING, PurchaseOccasion.DAILY_ESSENTIALS],
                typical_times=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON],
                price_sensitivity=0.7,
                brand_loyalty=0.4,
                impulse_tendency=0.2,
            ),

            CustomerType.FAMILY: CustomerProfile(
                customer_type=CustomerType.FAMILY,
                budget_range=(500_000, 3_000_000),
                typical_items=(8, 25),
                preferred_stores=[StoreType.SUPERMARKET, StoreType.BAKERY],
                preferred_categories=["Thực phẩm", "Gia dụng", "Chăm sóc", "Bánh kẹo"],
                avoided_categories=[],
                purchase_occasions=[PurchaseOccasion.WEEKLY_GROCERIES, PurchaseOccasion.SPECIAL_OCCASION, PurchaseOccasion.PARTY_PREP],
                typical_times=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING],
                price_sensitivity=0.5,
                brand_loyalty=0.6,
                impulse_tendency=0.5,  # Kids add impulse items
            ),

            CustomerType.SINGLE_YOUNG: CustomerProfile(
                customer_type=CustomerType.SINGLE_YOUNG,
                budget_range=(50_000, 400_000),
                typical_items=(2, 8),
                preferred_stores=[StoreType.CONVENIENCE, StoreType.CAFE, StoreType.RESTAURANT],
                preferred_categories=["Đồ uống", "Bánh kẹo", "Ẩm thực", "Chăm sóc"],
                avoided_categories=["Phần cứng"],
                purchase_occasions=[PurchaseOccasion.QUICK_SNACK, PurchaseOccasion.MEAL_PURCHASE, PurchaseOccasion.IMPULSE],
                typical_times=[TimeOfDay.LUNCH, TimeOfDay.EVENING, TimeOfDay.NIGHT],
                price_sensitivity=0.5,
                brand_loyalty=0.4,
                impulse_tendency=0.6,
            ),
        }

    def generate_purchase(self, context: PurchaseContext = None) -> Dict:
        """Generate a realistic purchase based on context."""
        if context is None:
            context = self._generate_random_context()

        profile = context.customer_profile
        store_type = context.store_type

        # Get available products for this store (dynamically generated)
        available_products = self.catalog.get_products_for_store(
            store_type,
            region=context.region,
            month=context.date.month if context.date else None,
            count=100  # Generate enough products to select from
        )

        if not available_products:
            # Fallback: generate generic products
            available_products = [self.catalog.generate_product() for _ in range(50)]

        # Filter by preferred categories
        preferred_products = [p for p in available_products
                             if p.category in profile.preferred_categories]
        other_products = [p for p in available_products
                         if p.category not in profile.avoided_categories]

        # Determine number of items
        min_items, max_items = profile.typical_items

        # Adjust for occasion
        if context.occasion == PurchaseOccasion.BULK_BUYING:
            min_items = max(min_items, 10)
            max_items = max(max_items, 30)
        elif context.occasion == PurchaseOccasion.QUICK_SNACK:
            min_items = 1
            max_items = min(max_items, 3)
        elif context.occasion == PurchaseOccasion.WEEKLY_GROCERIES:
            min_items = max(min_items, 5)
            max_items = max(max_items, 15)

        # Weekend/holiday adjustments
        if context.is_weekend or context.is_holiday:
            max_items = int(max_items * 1.3)

        num_items = random.randint(min_items, max_items)

        # Select products
        selected_items = []
        running_total = 0
        budget_min, budget_max = profile.budget_range

        for _ in range(num_items):
            # Stop if budget exceeded
            if running_total >= budget_max:
                break

            # Prefer preferred products
            if preferred_products and random.random() < 0.7:
                product = random.choice(preferred_products)
            elif other_products:
                product = random.choice(other_products)
            else:
                continue

            # Determine quantity
            qty_min, qty_max = product.common_qty_range

            # Adjust for bulk buying
            if context.occasion == PurchaseOccasion.BULK_BUYING:
                qty_min = max(qty_min, 3)
                qty_max = max(qty_max, qty_min + 5)

            qty = random.randint(qty_min, qty_max)

            # Get price
            price = product.get_price(currency="VND", store_type=store_type)

            # Price sensitivity check
            if profile.price_sensitivity > random.random():
                # May reduce quantity for expensive items
                if price > 100_000:
                    qty = max(1, qty // 2)

            total = price * qty

            if running_total + total <= budget_max * 1.2:  # Allow slight overrun
                selected_items.append({
                    "product": product,
                    "desc": product.name,
                    "qty": qty,
                    "unit": price,
                    "total": total,
                    "category": product.category,
                    "subcategory": product.subcategory,
                })
                running_total += total

        # Add impulse items
        if random.random() < profile.impulse_tendency:
            impulse_categories = ["Bánh kẹo", "Đồ uống"]
            impulse_products = [p for p in available_products
                               if p.category in impulse_categories]

            if impulse_products:
                impulse = random.choice(impulse_products)
                price = impulse.get_price(currency="VND", store_type=store_type)
                selected_items.append({
                    "product": impulse,
                    "desc": impulse.name,
                    "qty": 1,
                    "unit": price,
                    "total": price,
                    "category": impulse.category,
                    "subcategory": impulse.subcategory,
                    "is_impulse": True,
                })
                running_total += price

        # Calculate totals
        subtotal = sum(item["total"] for item in selected_items)

        # VAT handling (depends on store type)
        if store_type in [StoreType.SUPERMARKET, StoreType.CONVENIENCE]:
            vat_rate = random.choice([0, 8, 10])
        elif store_type == StoreType.RESTAURANT:
            vat_rate = 10
        elif store_type == StoreType.TRADITIONAL_MARKET:
            vat_rate = 0
        else:
            vat_rate = random.choice([0, 10])

        vat = int(subtotal * vat_rate / 100)
        grand_total = subtotal + vat

        # Select payment method
        payment_methods = {
            CustomerType.STUDENT: ["Tiền mặt", "MoMo", "ZaloPay"],
            CustomerType.OFFICE_WORKER: ["Thẻ tín dụng", "MoMo", "Chuyển khoản", "Tiền mặt"],
            CustomerType.HOMEMAKER: ["Tiền mặt", "Thẻ tín dụng"],
            CustomerType.ELDERLY: ["Tiền mặt"],
            CustomerType.TOURIST: ["Thẻ tín dụng", "Tiền mặt", "USD"],
            CustomerType.RESTAURANT_BUYER: ["Tiền mặt", "Chuyển khoản"],
            CustomerType.SMALL_BUSINESS: ["Chuyển khoản", "Tiền mặt"],
            CustomerType.FAMILY: ["Tiền mặt", "Thẻ tín dụng", "MoMo"],
            CustomerType.SINGLE_YOUNG: ["MoMo", "ZaloPay", "Thẻ tín dụng", "Tiền mặt"],
        }

        payment = random.choice(payment_methods.get(profile.customer_type, ["Tiền mặt"]))

        # Clean up items for output (remove product object)
        items_output = [{
            "desc": item["desc"],
            "qty": item["qty"],
            "unit": item["unit"],
            "total": item["total"],
        } for item in selected_items]

        return {
            "items": items_output,
            "subtotal": subtotal,
            "vat_rate": vat_rate,
            "vat": vat,
            "grand_total": grand_total,
            "payment_method": payment,
            "customer_type": context.customer_profile.customer_type.value,
            "store_type": context.store_type.value,
            "occasion": context.occasion.value,
            "time_of_day": context.time_of_day.value,
            "is_weekend": context.is_weekend,
            "is_holiday": context.is_holiday,
            "region": context.region.value,
            "currency": "VND",
        }

    def _generate_random_context(self) -> PurchaseContext:
        """Generate a random but realistic purchase context."""
        # Pick customer type with realistic distribution
        customer_weights = {
            CustomerType.HOMEMAKER: 0.2,
            CustomerType.OFFICE_WORKER: 0.2,
            CustomerType.STUDENT: 0.15,
            CustomerType.FAMILY: 0.15,
            CustomerType.SINGLE_YOUNG: 0.1,
            CustomerType.ELDERLY: 0.08,
            CustomerType.TOURIST: 0.05,
            CustomerType.RESTAURANT_BUYER: 0.04,
            CustomerType.SMALL_BUSINESS: 0.03,
        }

        customer_type = random.choices(
            list(customer_weights.keys()),
            weights=list(customer_weights.values()),
            k=1
        )[0]

        profile = self.customer_profiles[customer_type]

        # Pick store from customer preferences
        store_type = random.choice(profile.preferred_stores)

        # Pick occasion
        occasion = random.choice(profile.purchase_occasions)

        # Pick time
        time_of_day = random.choice(profile.typical_times)

        # Generate date
        today = datetime.date.today()
        days_back = random.randint(0, 365)
        date = today - datetime.timedelta(days=days_back)

        is_weekend = date.weekday() >= 5

        # Vietnamese holidays (simplified)
        holidays = [
            (1, 1),   # New Year
            (4, 30),  # Reunification Day
            (5, 1),   # Labor Day
            (9, 2),   # National Day
        ]
        is_holiday = (date.month, date.day) in holidays

        # Region distribution
        region_weights = {
            Region.SOUTH: 0.45,   # Ho Chi Minh area
            Region.NORTH: 0.35,   # Hanoi area
            Region.CENTRAL: 0.20,
        }
        region = random.choices(
            list(region_weights.keys()),
            weights=list(region_weights.values()),
            k=1
        )[0]

        return PurchaseContext(
            customer_profile=profile,
            store_type=store_type,
            occasion=occasion,
            time_of_day=time_of_day,
            date=date,
            region=region,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
        )

    def generate_batch(self, count: int,
                       store_filter: StoreType = None,
                       customer_filter: CustomerType = None) -> List[Dict]:
        """Generate multiple purchases."""
        purchases = []

        for _ in range(count):
            context = self._generate_random_context()

            if store_filter:
                context.store_type = store_filter

            if customer_filter:
                context.customer_profile = self.customer_profiles[customer_filter]

            purchase = self.generate_purchase(context)
            purchases.append(purchase)

        return purchases
