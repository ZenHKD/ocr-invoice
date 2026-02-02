"""
Synthetic Invoice Data Generator - Production Quality

A modular system for generating realistic Vietnamese invoices/receipts
with high variance for ML training data.

All products are dynamically generated using Vietnamese vocabulary templates
for maximum diversity.

Modules:
    - catalog: Dynamic product catalog with Vietnamese vocabulary
    - vietnamese_vocab: Vietnamese vocabulary generator (brands, products, modifiers)
    - layouts: Different invoice/receipt layout templates
    - defects: Visual defects simulation (folds, stains, shadows, etc.)
    - behaviors: Realistic purchasing behavior patterns
    - edge_cases: Unusual cases (partial scans, rotated, multi-receipt)
    - generator: Main orchestrator
"""

from .generator import SyntheticInvoiceGenerator
from .catalog import ProductCatalog
from .layouts import LayoutFactory
from .defects import DefectApplicator
from .behaviors import PurchaseBehavior
from .edge_cases import EdgeCaseGenerator
from .vietnamese_vocab import VietnameseVocabGenerator

__all__ = [
    "SyntheticInvoiceGenerator",
    "ProductCatalog",
    "VietnameseVocabGenerator",
    "LayoutFactory",
    "DefectApplicator",
    "PurchaseBehavior",
    "EdgeCaseGenerator",
]
