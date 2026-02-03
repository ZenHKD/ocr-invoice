"""
Main Synthetic Invoice Generator - Orchestrates all components.

Features:
    - Configurable generation scenarios
    - Realistic data distribution
    - Multiple output formats
    - Batch generation with progress tracking
    - Quality assurance checks
"""

import os
import random
import json
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path
from PIL import Image
from faker import Faker

from .catalog import ProductCatalog, StoreType, Region
from .layouts import LayoutFactory, LayoutType
from .defects import DefectApplicator, DefectConfig, create_defect_preset, DefectType
from .behaviors import PurchaseBehavior, PurchaseContext, CustomerType
from .edge_cases import EdgeCaseGenerator, EdgeCaseConfig, EdgeCaseType


class GenerationScenario(Enum):
    """Predefined generation scenarios."""
    TRAINING_BALANCED = "training_balanced"      # Balanced mix for training
    TRAINING_HARD = "training_hard"              # More difficult cases
    VALIDATION = "validation"                     # Clean for validation
    EDGE_CASES_FOCUS = "edge_cases_focus"        # Heavy edge cases
    RETAIL_FOCUS = "retail_focus"                # Focus on retail receipts
    RESTAURANT_FOCUS = "restaurant_focus"        # Focus on restaurant bills
    FORMAL_INVOICES = "formal_invoices"          # VAT invoices
    MIXED_RANDOM = "mixed_random"                # 40-30-30 nutritious meal
    PURE_RANDOM_FOCUS = "pure_random_focus"      # 100% Type 1
    PSEUDO_FOCUS = "pseudo_focus"                # 100% Type 2


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    output_dir: str = "data/raw"
    num_samples: int = 100
    scenario: GenerationScenario = GenerationScenario.TRAINING_BALANCED

    # Distribution controls
    realistic_ratio: float = 0.80       # Normal invoices
    edge_case_ratio: float = 0.10       # Edge cases
    blank_ratio: float = 0.05           # Blank pages
    unreadable_ratio: float = 0.05      # Unreadable

    # Quality settings
    defect_preset: str = "used_receipt"
    min_jpeg_quality: int = 40
    max_jpeg_quality: int = 95

    # Layout preferences (weights)
    layout_weights: Dict[LayoutType, float] = field(default_factory=dict)

    # Store type preferences
    store_weights: Dict[StoreType, float] = field(default_factory=dict)

    # Currency distribution
    currency_weights: Dict[str, float] = field(default_factory=lambda: {
        "VND": 0.85,
        "USD": 0.10,
        "EUR": 0.05,
    })

    # Region distribution
    region_weights: Dict[Region, float] = field(default_factory=lambda: {
        Region.SOUTH: 0.45,
        Region.NORTH: 0.35,
        Region.CENTRAL: 0.20,
    })

    # Text type distribution
    text_type_ratios: Dict[str, float] = field(default_factory=lambda: {
        "real": 1.0  # Default: 100% real corpus
    })

    def __post_init__(self):
        if not self.layout_weights:
            self.layout_weights = {
                LayoutType.SUPERMARKET_THERMAL: 0.15,
                LayoutType.FORMAL_VAT: 0.08,
                LayoutType.HANDWRITTEN: 0.08,
                LayoutType.CAFE_MINIMAL: 0.10,
                LayoutType.RESTAURANT_BILL: 0.10,
                LayoutType.MODERN_POS: 0.09,
                LayoutType.DELIVERY_RECEIPT: 0.10,
                LayoutType.HOTEL_BILL: 0.08,
                LayoutType.UTILITY_BILL: 0.08,
                LayoutType.ECOMMERCE_RECEIPT: 0.08,
                LayoutType.TAXI_RECEIPT: 0.06,
            }
        if not self.store_weights:
            self.store_weights = {
                StoreType.SUPERMARKET: 0.25,
                StoreType.CONVENIENCE: 0.20,
                StoreType.TRADITIONAL_MARKET: 0.15,
                StoreType.RESTAURANT: 0.15,
                StoreType.CAFE: 0.10,
                StoreType.BAKERY: 0.05,
                StoreType.PHARMACY: 0.05,
                StoreType.HARDWARE: 0.03,
                StoreType.ELECTRONICS: 0.02,
            }


def get_scenario_config(scenario: GenerationScenario) -> GenerationConfig:
    """Get configuration for a specific scenario."""
    configs = {
        GenerationScenario.TRAINING_BALANCED: GenerationConfig(
            realistic_ratio=0.75,
            edge_case_ratio=0.15,
            blank_ratio=0.05,
            unreadable_ratio=0.05,
            defect_preset="used_receipt",
            text_type_ratios={
                "pure_random": 0.40,
                "pseudo_vietnamese": 0.30,
                "real": 0.30,
            }
        ),

        GenerationScenario.TRAINING_HARD: GenerationConfig(
            realistic_ratio=0.50,
            edge_case_ratio=0.35,
            blank_ratio=0.08,
            unreadable_ratio=0.07,
            defect_preset="damaged",
            text_type_ratios={
                "pure_random": 0.40,
                "pseudo_vietnamese": 0.30,
                "real": 0.30,
            }
        ),

        GenerationScenario.VALIDATION: GenerationConfig(
            realistic_ratio=0.95,
            edge_case_ratio=0.05,
            blank_ratio=0.00,
            unreadable_ratio=0.00,
            defect_preset="good_scan",
            text_type_ratios={
                "pure_random": 0.40,
                "pseudo_vietnamese": 0.30,
                "real": 0.30,
            }
        ),

        GenerationScenario.EDGE_CASES_FOCUS: GenerationConfig(
            realistic_ratio=0.30,
            edge_case_ratio=0.60,
            blank_ratio=0.05,
            unreadable_ratio=0.05,
            defect_preset="extreme",
            text_type_ratios={
                "pure_random": 0.40,
                "pseudo_vietnamese": 0.30,
                "real": 0.30,
            }
        ),

        GenerationScenario.RETAIL_FOCUS: GenerationConfig(
            realistic_ratio=0.85,
            edge_case_ratio=0.10,
            layout_weights={
                LayoutType.SUPERMARKET_THERMAL: 0.7,
                LayoutType.CAFE_MINIMAL: 0.2,
                LayoutType.HANDWRITTEN: 0.1,
            },
            store_weights={
                StoreType.SUPERMARKET: 0.5,
                StoreType.CONVENIENCE: 0.3,
                StoreType.TRADITIONAL_MARKET: 0.2,
            },
        ),

        GenerationScenario.RESTAURANT_FOCUS: GenerationConfig(
            realistic_ratio=0.85,
            edge_case_ratio=0.10,
            layout_weights={
                LayoutType.CAFE_MINIMAL: 0.4,
                LayoutType.HANDWRITTEN: 0.3,
                LayoutType.SUPERMARKET_THERMAL: 0.3,
            },
            store_weights={
                StoreType.RESTAURANT: 0.5,
                StoreType.CAFE: 0.3,
                StoreType.BAKERY: 0.2,
            },
        ),

        GenerationScenario.FORMAL_INVOICES: GenerationConfig(
            realistic_ratio=0.90,
            edge_case_ratio=0.10,
            defect_preset="good_scan",
            layout_weights={
                LayoutType.FORMAL_VAT: 0.8,
                LayoutType.SUPERMARKET_THERMAL: 0.2,
            },
        ),

        GenerationScenario.MIXED_RANDOM: GenerationConfig(
            realistic_ratio=0.85,
            edge_case_ratio=0.10,
            blank_ratio=0.025,
            unreadable_ratio=0.025,
            defect_preset="used_receipt",
            text_type_ratios={
                "pure_random": 0.40,
                "pseudo_vietnamese": 0.30,
                "real": 0.30,
            }
        ),

        GenerationScenario.PURE_RANDOM_FOCUS: GenerationConfig(
            realistic_ratio=0.90,
            text_type_ratios={"pure_random": 1.0}
        ),

        GenerationScenario.PSEUDO_FOCUS: GenerationConfig(
            realistic_ratio=0.90,
            text_type_ratios={"pseudo_vietnamese": 1.0}
        ),
    }

    return configs.get(scenario, GenerationConfig())


class SyntheticInvoiceGenerator:
    """Main generator class orchestrating all components."""

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()

        # Initialize components
        self.catalog = ProductCatalog()
        self.behavior = PurchaseBehavior(self.catalog)
        self.defect_applicator = DefectApplicator(
            create_defect_preset(self.config.defect_preset)
        )
        self.edge_case_generator = EdgeCaseGenerator()
        self.faker = Faker("vi_VN")

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_batch(self, num_samples: int = None,
                       progress_callback=None) -> List[Dict]:
        """Generate a batch of synthetic invoices."""
        if num_samples is None:
            num_samples = self.config.num_samples

        results = []

        for i in range(num_samples):
            try:
                # Determine what type to generate
                dice = random.random()

                if dice < self.config.blank_ratio:
                    result = self._generate_blank(i)
                elif dice < self.config.blank_ratio + self.config.unreadable_ratio:
                    result = self._generate_unreadable(i)
                elif dice < (self.config.blank_ratio + self.config.unreadable_ratio +
                            self.config.edge_case_ratio):
                    result = self._generate_with_edge_case(i)
                else:
                    result = self._generate_realistic(i)

                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, num_samples, result)

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        return results

    def _generate_realistic(self, sample_id: int) -> Dict:
        """Generate a realistic invoice."""
        # Generate purchase data
        purchase_data = self.behavior.generate_purchase(
            text_type_ratios=self.config.text_type_ratios
        )

        # Add store information
        store_type = self._select_store_type()
        store_data = self._generate_store_data(store_type)

        # Combine data
        data = {**store_data, **purchase_data}

        # Select and render layout
        layout = LayoutFactory.create_random(self.config.layout_weights)
        img = layout.render(data)

        # Get OCR ground truth with bounding boxes
        ocr_annotations = layout.get_ocr_annotations()

        # Apply defects
        img, applied_defects, defect_transforms = self.defect_applicator.apply_random_defects(img)

        # Apply defect-induced coordinate transforms
        ocr_annotations = self._apply_defect_transforms(ocr_annotations, defect_transforms)

        # Save
        return self._save_sample(
            sample_id, img, data,
            sample_type="realistic",
            layout_type=layout.config.layout_type.value,
            defects=[d.value for d in applied_defects],
            ocr_annotations=ocr_annotations
        )

    def _generate_with_edge_case(self, sample_id: int) -> Dict:
        """Generate an invoice with edge case applied."""
        # First generate a normal invoice
        purchase_data = self.behavior.generate_purchase(
            text_type_ratios=self.config.text_type_ratios
        )
        store_type = self._select_store_type()
        store_data = self._generate_store_data(store_type)
        data = {**store_data, **purchase_data}

        layout = LayoutFactory.create_random(self.config.layout_weights)
        img = layout.render(data)

        # Get OCR ground truth with bounding boxes before edge case transforms
        ocr_annotations = layout.get_ocr_annotations()

        # Apply edge case
        img, edge_case, edge_metadata = self.edge_case_generator.apply_edge_case(img)

        # Apply coordinate transform to bounding boxes based on edge case metadata
        ocr_annotations = self._transform_annotations(ocr_annotations, edge_metadata)

        # Maybe apply some defects too
        if random.random() < 0.5:
            img, applied_defects, defect_transforms = self.defect_applicator.apply_random_defects(img)
            # Apply defect-induced coordinate transforms
            ocr_annotations = self._apply_defect_transforms(ocr_annotations, defect_transforms)
        else:
            applied_defects = []

        # Save
        return self._save_sample(
            sample_id, img, data,
            sample_type="edge_case",
            layout_type=layout.config.layout_type.value,
            defects=[d.value for d in applied_defects],
            edge_case=edge_case.value,
            edge_metadata=edge_metadata,
            ocr_annotations=ocr_annotations
        )

    def _transform_annotations(self, annotations: List[Dict], edge_metadata: Dict) -> List[Dict]:
        """Apply coordinate transforms to polygon annotations based on edge case metadata."""
        if not annotations or not edge_metadata:
            return annotations

        from .geometry import rotate_point, apply_perspective_transform

        transformed = []

        # Handle offset transform (textured_background, photo_of_receipt, etc.)
        offset_x = edge_metadata.get("offset_x", 0)
        offset_y = edge_metadata.get("offset_y", 0)

        # Handle rotation
        rotation_angle = edge_metadata.get("rotation", 0)

        # Handle perspective transform matrix
        perspective_matrix = edge_metadata.get("perspective_matrix", None)

        # Get original image size for rotation center
        orig_width = edge_metadata.get("orig_width", 0)
        orig_height = edge_metadata.get("orig_height", 0)

        for ann in annotations:
            text = ann.get("text", "")
            polygon = ann.get("polygon", [])
            bbox = ann.get("bbox", [])

            # If no polygon, try to create from bbox
            if not polygon and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            if len(polygon) != 4:
                transformed.append(ann)
                continue

            new_polygon = []
            for point in polygon:
                x, y = point[0], point[1]

                # Apply rotation if present
                if rotation_angle != 0 and orig_width > 0 and orig_height > 0:
                    # Rotate around center of original image
                    cx, cy = orig_width / 2, orig_height / 2
                    x, y = rotate_point((x, y), (cx, cy), -rotation_angle)

                    # Compute new center after rotation expansion
                    new_cx = edge_metadata.get("new_width", orig_width) / 2
                    new_cy = edge_metadata.get("new_height", orig_height) / 2

                    # Subtract the 'cx' added by rotate_point and add 'new_cx'
                    x = x - cx + new_cx
                    y = y - cy + new_cy

                # Apply perspective transform if present
                if perspective_matrix is not None:
                    x, y = apply_perspective_transform((x, y), perspective_matrix)

                # Apply offset (always last)
                x += offset_x
                y += offset_y

                new_polygon.append([int(x), int(y)])

            # Compute new bounding box from polygon
            xs = [p[0] for p in new_polygon]
            ys = [p[1] for p in new_polygon]
            new_bbox = [min(xs), min(ys), max(xs), max(ys)]

            transformed.append({
                "text": text,
                "polygon": new_polygon,
                "bbox": new_bbox  # Axis-aligned bounding box for backward compatibility
            })

        return transformed

    def _apply_defect_transforms(self, annotations: List[Dict], transforms: List[Dict]) -> List[Dict]:
        """Apply geometric transforms from defects to annotations."""
        if not annotations or not transforms:
            return annotations

        from .geometry import apply_perspective_transform

        transformed_anns = annotations

        for transform in transforms:
            if transform.get("type") == "matrix":
                matrix = transform["matrix"]
                new_anns = []

                for ann in transformed_anns:
                    polygon = ann.get("polygon", [])

                    # Ensure polygon exists
                    if not polygon and ann.get("bbox") and len(ann["bbox"]) == 4:
                        x1, y1, x2, y2 = ann["bbox"]
                        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                    if len(polygon) != 4:
                        new_anns.append(ann)
                        continue

                    new_poly = []
                    for pt in polygon:
                        x, y = apply_perspective_transform((pt[0], pt[1]), matrix)
                        new_poly.append([int(x), int(y)])

                    # Recalculate bbox from new polygon
                    xs = [p[0] for p in new_poly]
                    ys = [p[1] for p in new_poly]
                    if xs and ys:
                        new_bbox = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        new_bbox = ann.get("bbox")

                    new_ann = ann.copy()
                    new_ann["polygon"] = new_poly
                    new_ann["bbox"] = new_bbox
                    new_anns.append(new_ann)

                transformed_anns = new_anns

        return transformed_anns

    def _generate_blank(self, sample_id: int) -> Dict:
        """Generate a blank page."""
        img, metadata = self.edge_case_generator.generate_blank_page()

        # Apply some noise
        img, defects, _ = self.defect_applicator.apply_random_defects(img)

        return self._save_sample(
            sample_id, img, {},
            sample_type="blank",
            defects=[d.value for d in defects],
            note="Blank scan error"
        )

    def _generate_unreadable(self, sample_id: int) -> Dict:
        """Generate an unreadable/corrupted image."""
        img, metadata = self.edge_case_generator.generate_unreadable()

        return self._save_sample(
            sample_id, img, {},
            sample_type="unreadable",
            note="OCR fail",
            corruption_type=metadata.get("corruption", "unknown")
        )

    def _select_store_type(self) -> StoreType:
        """Select store type based on weights."""
        types = list(self.config.store_weights.keys())
        weights = list(self.config.store_weights.values())
        return random.choices(types, weights=weights, k=1)[0]

    def _generate_store_data(self, store_type: StoreType) -> Dict:
        """Generate store information."""
        from .vietnamese_vocab import STORE_PROFILES, REGIONS

        # Mappings
        store_key_map = {
            StoreType.SUPERMARKET: "supermarket",
            StoreType.CONVENIENCE: "convenience_store",
            StoreType.TRADITIONAL_MARKET: "traditional_market",
            StoreType.RESTAURANT: "restaurant",
            StoreType.CAFE: "cafe",
            StoreType.BAKERY: "bakery",
            StoreType.ELECTRONICS: "electronics",
            StoreType.PHARMACY: "pharmacy",
            StoreType.HARDWARE: "hardware",
            StoreType.CLOTHING: "clothing",
        }

        store_key = store_key_map.get(store_type)
        profile = STORE_PROFILES.get(store_key, {})

        # 1. Store Name Generation
        # Try to get from REGIONS first (for variety), then fallback to hardcoded list (if needed)
        store_names = []

        # Simple hardcoded fallback lists just in case
        fallback_names = {
            StoreType.SUPERMARKET: ["Big C", "Co.opmart", "Lotte Mart", "Vinmart", "Mega Market", "Aeon"],
            StoreType.CONVENIENCE: ["Circle K", "7-Eleven", "Family Mart", "B's Mart", "Vinmart+", "GS25"],
            StoreType.TRADITIONAL_MARKET: ["Chợ", "Tạp hóa", "Cửa hàng"],
            StoreType.RESTAURANT: ["Quán cơm", "Nhà hàng", "Quán ăn", "Bếp", "Kitchen"],
            StoreType.CAFE: ["The Coffee House", "Highlands Coffee", "Phúc Long", "Cộng Cà Phê", "Starbucks"],
        }

        region_keys = list(REGIONS.keys())
        if region_keys:
            r_key = random.choice(region_keys)
            if "store_names" in REGIONS[r_key]:
                 # These are often generic "Names" like "Ha Noi", "Sai Gon", need prefix
                 prefixes = ["Cửa hàng", "Siêu thị", "Shop"]
                 store_names.extend([f"{random.choice(prefixes)} {n}" for n in REGIONS[r_key]["store_names"]])

        # Fallback to defaults
        default_names = fallback_names.get(store_type, ["Cửa hàng"])
        store_names.extend(default_names)

        base_name = random.choice(store_names)

        # Add branch number or location sometimes
        if random.random() < 0.4:
            store_name = f"{base_name} - Chi nhánh {random.randint(1, 50)}"
        elif random.random() < 0.3:
            province = random.choice([
                "Quận 1", "Quận 3", "Quận 7", "Bình Thạnh", "Tân Bình",
                "Hoàn Kiếm", "Cầu Giấy", "Đống Đa", "Ba Đình",
            ])
            store_name = f"{base_name} {province}"
        else:
            store_name = base_name

        # Generate address
        streets = [
            "Nguyễn Huệ", "Lê Lợi", "Đồng Khởi", "Nguyễn Trãi", "Cách Mạng Tháng 8",
            "Điện Biên Phủ", "Võ Văn Tần", "Phan Xích Long", "Nguyễn Đình Chiểu",
            "Lý Tự Trọng", "Hai Bà Trưng", "Trần Hưng Đạo", "Ngô Quyền",
        ]

        provinces = [
            "TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng",
            "Bình Dương", "Đồng Nai", "Bắc Ninh", "Thanh Hóa", "Nghệ An",
        ]

        address = f"{random.randint(1, 500)} {random.choice(streets)}, {random.choice(provinces)}"

        # Generate contact info
        phone_prefixes = ["028", "024", "0236", "0292", "0225"]
        phone = f"{random.choice(phone_prefixes)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"

        # Tax code
        tax_code = f"{random.randint(10, 99)}{random.randint(10000, 99999)}{random.randint(100, 999)}"

        # Invoice number
        date = datetime.date.today() - datetime.timedelta(days=random.randint(0, 365))
        inv_num = f"HD{date.strftime('%y%m')}{random.randint(10000, 99999)}"

        # Date format
        date_formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y.%m.%d",
            "%d/%m/%Y %H:%M",
            "%d %b %Y",
            "%d-%m-%Y %H:%M:%S",
        ]
        date_str = date.strftime(random.choice(date_formats))

        # Load payment methods from profile if available
        payment_method = "Tiền mặt"
        if profile and "payment_methods" in profile:
            payment_method = random.choice(profile["payment_methods"])

        # Helper: Get footer message from profile
        footer_msg = "Cảm ơn quý khách!"
        if profile and "footer_messages" in profile:
             footer_msg = random.choice(profile["footer_messages"])

        return {
            "store_name": store_name,
            "store_type": store_type.value,
            "address": address,
            "phone": phone,
            "tax_code": tax_code,
            "invoice_number": inv_num,
            "date": date_str,
            "barcode": "".join(str(random.randint(0, 9)) for _ in range(13)),
            "payment_method": payment_method,
            "footer_message": footer_msg, # Add this field to be used by layout
            "features": profile.get("features", []) # Pass features for layout logic
        }

    def _save_sample(self, sample_id: int, img: Image.Image,
                     data: Dict, **extra_metadata) -> Dict:
        """Save image and metadata."""
        # Generate unique ID
        unique_id = f"{sample_id:06d}_{uuid.uuid4().hex[:8]}"

        # File paths
        img_path = os.path.join(self.config.output_dir, f"invoice_{unique_id}.jpg")
        json_path = os.path.join(self.config.output_dir, f"invoice_{unique_id}.json")

        # Save image with random quality
        quality = random.randint(self.config.min_jpeg_quality, self.config.max_jpeg_quality)
        img.convert("RGB").save(img_path, "JPEG", quality=quality)

        # Get OCR annotations (with bounding boxes) from extra_metadata
        ocr_annotations = extra_metadata.pop("ocr_annotations", [])
        sample_type = extra_metadata.get("sample_type", "unknown")
        layout_type = extra_metadata.get("layout_type", "unknown")

        # Prepare metadata for OCR training with bounding boxes
        metadata = {
            "id": unique_id,
            "image_path": img_path,
            "image_size": list(img.size),
            "sample_type": sample_type,
            "layout_type": layout_type,
            # OCR annotations with bounding boxes: [{"text": str, "bbox": [x1,y1,x2,y2]}]
            "ocr_annotations": ocr_annotations,
            # Legacy fields for backward compatibility
            "ocr_text": [a["text"] for a in ocr_annotations] if ocr_annotations else [],
            "ocr_text_full": "\n".join(a["text"] for a in ocr_annotations) if ocr_annotations else "",
        }

        # Save metadata
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata


    @classmethod
    def from_scenario(cls, scenario: GenerationScenario,
                      output_dir: str = "data/raw",
                      num_samples: int = 100) -> "SyntheticInvoiceGenerator":
        """Create generator from a predefined scenario."""
        config = get_scenario_config(scenario)
        config.output_dir = output_dir
        config.num_samples = num_samples
        return cls(config)


def generate_dataset(output_dir: str = "data/raw",
                     num_samples: int = 100,
                     scenario: str = "training_balanced",
                     verbose: bool = True) -> List[Dict]:
    """Convenience function to generate a dataset."""
    scenario_enum = GenerationScenario(scenario)
    generator = SyntheticInvoiceGenerator.from_scenario(
        scenario_enum, output_dir, num_samples
    )

    def progress(current, total, result):
        if verbose:
            sample_type = result.get("sample_type", "unknown")
            print(f"[{current}/{total}] Generated {sample_type}: {result.get('id', 'N/A')}")

    results = generator.generate_batch(progress_callback=progress if verbose else None)

    if verbose:
        # Print summary
        type_counts = {}
        layout_counts = {}
        for r in results:
            t = r.get("sample_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

            layout = r.get("layout_type", "unknown")
            if layout != "unknown":
                layout_counts[layout] = layout_counts.get(layout, 0) + 1

        print(f"\n=== Generation Complete ===")
        print(f"Total samples: {len(results)}")

        print(f"\nSample Types:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t}: {count} ({100 * count / len(results):.1f}%)")

        if layout_counts:
            print(f"\nLayout Types:")
            for layout, count in sorted(layout_counts.items()):
                print(f"  {layout}: {count} ({100 * count / len(results):.1f}%)")

    return results


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic invoice dataset")
    parser.add_argument("-o", "--output", default="data/raw", help="Output directory")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of samples")
    parser.add_argument("-s", "--scenario", default="training_balanced",
                       choices=[s.value for s in GenerationScenario],
                       help="Generation scenario")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        num_samples=args.num,
        scenario=args.scenario,
        verbose=not args.quiet
    )
