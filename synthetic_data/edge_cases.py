"""
Edge Cases Generator for unusual/difficult invoice scenarios.

Provides:
    - Partial scans (cropped, cut-off)
    - Multi-receipt composites (multiple receipts on one scan)
    - Extreme rotations and angles
    - Blank pages with artifacts
    - Unreadable/corrupted images
    - Overlapping documents
    - Mixed orientation receipts
    - Receipt on textured backgrounds
    - Photos of receipts (not scans)
    - Receipts with foreign languages mixed in
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import cv2


class EdgeCaseType(Enum):
    PARTIAL_TOP = "partial_top"
    PARTIAL_BOTTOM = "partial_bottom"
    PARTIAL_LEFT = "partial_left"
    PARTIAL_RIGHT = "partial_right"
    MULTI_RECEIPT = "multi_receipt"
    EXTREME_ROTATION = "extreme_rotation"
    UPSIDE_DOWN = "upside_down"
    BLANK_PAGE = "blank_page"
    UNREADABLE = "unreadable"
    OVERLAPPING = "overlapping"
    PHOTO_OF_RECEIPT = "photo_of_receipt"
    TEXTURED_BACKGROUND = "textured_background"
    CRUMPLED = "crumpled"
    FOLDED_CORNER = "folded_corner"
    VERY_SMALL = "very_small"
    VERY_LARGE = "very_large"
    LOW_CONTRAST = "low_contrast"
    INVERTED = "inverted"
    # New Phase 2 realistic effects
    HAND_HOLDING = "hand_holding"  # Fingers obscuring edges
    REALISTIC_BACKGROUND = "realistic_background"  # Cafe table, desk, etc.
    TPS_WARP = "tps_warp"  # Thin Plate Spline 3D warping
    CYLINDRICAL_CURL = "cylindrical_curl"  # Freshly printed curl effect



@dataclass
class EdgeCaseConfig:
    """Configuration for edge case generation."""
    edge_case_ratio: float = 0.2  # 20% edge cases by default
    case_probabilities: Dict[EdgeCaseType, float] = None

    def __post_init__(self):
        if self.case_probabilities is None:
            self.case_probabilities = {
                EdgeCaseType.PARTIAL_TOP: 0.15,
                EdgeCaseType.PARTIAL_BOTTOM: 0.15,
                EdgeCaseType.PARTIAL_LEFT: 0.1,
                EdgeCaseType.PARTIAL_RIGHT: 0.1,
                EdgeCaseType.MULTI_RECEIPT: 0.1,
                EdgeCaseType.EXTREME_ROTATION: 0.1,
                EdgeCaseType.UPSIDE_DOWN: 0.05,
                EdgeCaseType.BLANK_PAGE: 0.05,
                EdgeCaseType.UNREADABLE: 0.05,
                EdgeCaseType.OVERLAPPING: 0.05,
                EdgeCaseType.PHOTO_OF_RECEIPT: 0.08,
                EdgeCaseType.TEXTURED_BACKGROUND: 0.08,
                EdgeCaseType.CRUMPLED: 0.08,
                EdgeCaseType.FOLDED_CORNER: 0.1,
                EdgeCaseType.VERY_SMALL: 0.03,
                EdgeCaseType.VERY_LARGE: 0.03,
                EdgeCaseType.LOW_CONTRAST: 0.1,
                EdgeCaseType.INVERTED: 0.02,
                # New Phase 2 realistic effects
                EdgeCaseType.HAND_HOLDING: 0.15,  # Common in user photos
                EdgeCaseType.REALISTIC_BACKGROUND: 0.12,  # Cafe tables, etc.
                EdgeCaseType.TPS_WARP: 0.1,  # 3D warping
                EdgeCaseType.CYLINDRICAL_CURL: 0.1,  # Freshly printed
            }


class EdgeCaseGenerator:
    """Generate edge case invoice images."""

    def __init__(self, config: EdgeCaseConfig = None):
        self.config = config or EdgeCaseConfig()

    def should_generate_edge_case(self) -> bool:
        """Determine if an edge case should be generated."""
        return random.random() < self.config.edge_case_ratio

    def select_edge_case(self) -> EdgeCaseType:
        """Select which edge case to generate."""
        cases = list(self.config.case_probabilities.keys())
        weights = list(self.config.case_probabilities.values())
        return random.choices(cases, weights=weights, k=1)[0]

    def apply_edge_case(self, img: Image.Image,
                        edge_case: EdgeCaseType = None) -> Tuple[Image.Image, EdgeCaseType, Dict]:
        """Apply an edge case transformation."""
        if edge_case is None:
            edge_case = self.select_edge_case()

        method = getattr(self, f"_apply_{edge_case.value}", None)

        if method:
            result, metadata = method(img)
            return result, edge_case, metadata

        return img, edge_case, {}

    def generate_blank_page(self) -> Tuple[Image.Image, Dict]:
        """Generate a blank page."""
        return self._apply_blank_page(None)

    def generate_unreadable(self) -> Tuple[Image.Image, Dict]:
        """Generate an unreadable/corrupted image."""
        return self._apply_unreadable(None)

    # ============= PARTIAL SCANS =============

    def _apply_partial_top(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Cut off the top portion of the receipt."""
        if img is None:
            return self._generate_placeholder(), {"crop": "top", "crop_ratio": 0}

        w, h = img.size
        crop_ratio = random.uniform(0.15, 0.4)
        crop_height = int(h * crop_ratio)

        cropped = img.crop((0, crop_height, w, h))

        # Add white/gray top edge to simulate scan edge
        result = Image.new("RGB", (w, h - crop_height + 20), (245, 245, 245))
        result.paste(cropped, (0, 20))

        return result, {"crop": "top", "crop_ratio": crop_ratio, "offset_y": 20 - crop_height}

    def _apply_partial_bottom(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Cut off the bottom portion."""
        if img is None:
            return self._generate_placeholder(), {"crop": "bottom", "crop_ratio": 0}

        w, h = img.size
        crop_ratio = random.uniform(0.15, 0.4)
        crop_height = int(h * crop_ratio)

        cropped = img.crop((0, 0, w, h - crop_height))

        # Add jagged edge at bottom
        result_h = h - crop_height + 20
        result = Image.new("RGB", (w, result_h), (245, 245, 245))
        result.paste(cropped, (0, 0))

        # Add torn edge effect
        draw = ImageDraw.Draw(result)
        y = h - crop_height
        for x in range(0, w, 5):
            tear_y = y + random.randint(-5, 10)
            draw.rectangle((x, tear_y, x + 5, result_h), fill=(255, 255, 255))

        return result, {"crop": "bottom", "crop_ratio": crop_ratio}

    def _apply_partial_left(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Cut off the left portion."""
        if img is None:
            return self._generate_placeholder(), {"crop": "left", "crop_ratio": 0}

        w, h = img.size
        crop_ratio = random.uniform(0.1, 0.3)
        crop_width = int(w * crop_ratio)

        cropped = img.crop((crop_width, 0, w, h))

        return cropped, {"crop": "left", "crop_ratio": crop_ratio, "offset_x": -crop_width}

    def _apply_partial_right(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Cut off the right portion."""
        if img is None:
            return self._generate_placeholder(), {"crop": "right", "crop_ratio": 0}

        w, h = img.size
        crop_ratio = random.uniform(0.1, 0.3)
        crop_width = int(w * crop_ratio)

        cropped = img.crop((0, 0, w - crop_width, h))

        return cropped, {"crop": "right", "crop_ratio": crop_ratio}

    # ============= MULTI-RECEIPT =============

    def _apply_multi_receipt(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Create a composite with multiple receipts."""
        if img is None:
            return self._generate_placeholder(), {"num_receipts": 0}

        # Create a larger canvas
        w, h = img.size
        canvas_w = int(w * 1.8)
        canvas_h = int(h * 1.5)

        canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

        # Place main receipt
        x1 = random.randint(0, 50)
        y1 = random.randint(0, 50)

        # Rotate slightly
        angle1 = random.uniform(-5, 5)
        img_rotated = img.rotate(angle1, expand=True, fillcolor=(240, 240, 240))
        main_new_w, main_new_h = img_rotated.size
        canvas.paste(img_rotated, (x1, y1))

        base_metadata = {
            "rotation": angle1,
            "offset_x": x1,
            "offset_y": y1,
            "orig_width": w,
            "orig_height": h,
            "new_width": main_new_w,
            "new_height": main_new_h,
            "num_receipts": 1
        }

        # Create a second "receipt" (smaller version or cropped)
        if random.random() < 0.7:
            # Second receipt - smaller scale
            scale = random.uniform(0.6, 0.9)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img2 = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            angle2 = random.uniform(-10, 10)
            img2_rotated = img2.rotate(angle2, expand=True, fillcolor=(240, 240, 240))

            # Position to the side or overlapping
            x2 = random.randint(w // 2, canvas_w - new_w)
            y2 = random.randint(0, canvas_h - new_h)

            canvas.paste(img2_rotated, (x2, y2))
            
            base_metadata["num_receipts"] = 2
            base_metadata["overlap"] = True
            return canvas, base_metadata

        return canvas, base_metadata

    # ============= ROTATIONS =============

    def _apply_extreme_rotation(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply extreme rotation (beyond normal skew)."""
        if img is None:
            return self._generate_placeholder(), {"rotation": 0}

        orig_w, orig_h = img.size
        
        angle = random.choice([
            random.uniform(-45, -20),
            random.uniform(20, 45),
            random.uniform(-90, -70),
            random.uniform(70, 90),
        ])

        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        new_w, new_h = rotated.size

        return rotated, {
            "rotation": angle,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "new_width": new_w,
            "new_height": new_h
        }

    def _apply_upside_down(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Flip receipt upside down."""
        if img is None:
            return self._generate_placeholder(), {"rotation": 180}

        orig_w, orig_h = img.size
        
        # Add small random deviation from perfect 180
        angle = 180 + random.uniform(-3, 3)
        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        new_w, new_h = rotated.size

        return rotated, {
            "rotation": angle,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "new_width": new_w,
            "new_height": new_h
        }

    # ============= BLANK & UNREADABLE =============

    def _apply_blank_page(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Generate a blank page with some artifacts."""
        w = random.randint(400, 1200)
        h = random.randint(600, 1600)

        # Paper color
        color = random.choice([
            (255, 255, 255),
            (250, 248, 240),
            (248, 248, 248),
            (255, 250, 245),
        ])

        page = Image.new("RGB", (w, h), color)
        draw = ImageDraw.Draw(page)

        # Add some artifacts
        if random.random() < 0.3:
            # Faint lines (like notebook paper)
            line_color = (220, 220, 230)
            for y in range(50, h, 30):
                draw.line((20, y, w - 20, y), fill=line_color, width=1)

        if random.random() < 0.2:
            # Scanner edge shadow
            for x in range(20):
                alpha = int(30 * (1 - x / 20))
                draw.line((x, 0, x, h), fill=(200 - alpha, 200 - alpha, 200 - alpha))

        if random.random() < 0.4:
            # Dust specks
            for _ in range(random.randint(3, 15)):
                x = random.randint(0, w)
                y = random.randint(0, h)
                size = random.randint(1, 3)
                draw.ellipse((x, y, x + size, y + size), fill=(80, 80, 80))

        return page, {"type": "blank"}

    def _apply_unreadable(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Generate an unreadable/corrupted image."""
        w = random.randint(400, 1000)
        h = random.randint(600, 1200)

        # Random noise pattern
        np_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Add some structure
        corruption_type = random.choice(["noise", "glitch", "overexposed", "black"])

        if corruption_type == "noise":
            # Heavy noise
            pass  # Already noisy

        elif corruption_type == "glitch":
            # Horizontal glitch lines
            for _ in range(random.randint(10, 50)):
                y = random.randint(0, h - 10)
                shift = random.randint(-50, 50)
                np_img[y:y + 5, :] = np.roll(np_img[y:y + 5, :], shift, axis=1)

        elif corruption_type == "overexposed":
            # Almost completely white
            np_img = np.full((h, w, 3), 250, dtype=np.uint8)
            noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
            np_img = np.clip(np_img.astype(int) - noise, 0, 255).astype(np.uint8)

        elif corruption_type == "black":
            # Almost completely black
            np_img = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)

        result = Image.fromarray(np_img)

        return result, {"type": "unreadable", "corruption": corruption_type}

    # ============= OVERLAPPING =============

    def _apply_overlapping(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Create overlapping documents effect."""
        if img is None:
            return self._generate_placeholder(), {}

        w, h = img.size

        # Create canvas larger than the image
        canvas_w = int(w * 1.5)
        canvas_h = int(h * 1.3)

        # Background - could be another document
        canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))

        # Add a fake background document
        draw = ImageDraw.Draw(canvas)
        for y in range(20, canvas_h, 25):
            # Simulate lines of text
            line_len = random.randint(100, canvas_w - 100)
            draw.rectangle((50, y, 50 + line_len, y + 8), fill=(180, 180, 180))

        # Paste the main receipt on top with offset
        x_offset = random.randint(20, 100)
        y_offset = random.randint(20, 100)

        # Slight rotation
        angle = random.uniform(-3, 3)
        img_rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        new_w, new_h = img_rotated.size

        canvas.paste(img_rotated, (x_offset, y_offset))

        return canvas, {
            "overlapping": True, 
            "offset_x": x_offset, 
            "offset_y": y_offset, 
            "rotation": angle,
            "orig_width": w,
            "orig_height": h,
            "new_width": new_w,
            "new_height": new_h
        }

    # ============= PHOTO OF RECEIPT =============

    def _apply_photo_of_receipt(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Simulate a photo taken of a receipt (not a scan)."""
        if img is None:
            return self._generate_placeholder(), {}

        # Add perspective distortion
        img = img.convert("RGB")
        cv_img = np.array(img)
        h, w = cv_img.shape[:2]

        # Perspective transform
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Random perspective
        margin = 30
        dst_pts = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin * 2)],
            [w - random.randint(0, margin * 2), h - random.randint(0, margin)],
            [random.randint(0, margin * 2), h - random.randint(0, margin)]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(cv_img, M, (w, h))

        # Add vignette effect (darker edges)
        rows, cols = result.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.5)
        Y = cv2.getGaussianKernel(rows, rows * 0.5)
        kernel = Y * X.T
        mask = kernel / kernel.max()

        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask

        # Add some blur (camera focus)
        if random.random() < 0.5:
            result = cv2.GaussianBlur(result, (3, 3), 0)

        # Adjust brightness (flash or low light)
        brightness = random.uniform(0.8, 1.2)
        result = cv2.convertScaleAbs(result, alpha=brightness, beta=random.randint(-20, 20))

        # Add slight color cast (indoor lighting)
        if random.random() < 0.4:
            # Warm light
            result[:, :, 0] = np.clip(result[:, :, 0] * 1.1, 0, 255)  # Blue down
            result[:, :, 2] = np.clip(result[:, :, 2] * 0.95, 0, 255)  # Red up

        return Image.fromarray(result), {
            "photo": True,
            "perspective_matrix": M.tolist(),  # Convert numpy array to list for JSON serialization
            "orig_width": w,
            "orig_height": h
        }

    # ============= TEXTURED BACKGROUND =============

    def _apply_textured_background(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Place receipt on a textured background."""
        if img is None:
            return self._generate_placeholder(), {}

        w, h = img.size

        # Create canvas with texture
        canvas_w = w + random.randint(50, 150)
        canvas_h = h + random.randint(50, 150)

        # Background type
        bg_type = random.choice(["wood", "fabric", "desk", "marble"])

        if bg_type == "wood":
            # Wood-like pattern
            canvas = Image.new("RGB", (canvas_w, canvas_h), (139, 90, 43))
            draw = ImageDraw.Draw(canvas)
            for i in range(0, canvas_h, 10):
                color = (139 + random.randint(-20, 20),
                        90 + random.randint(-15, 15),
                        43 + random.randint(-10, 10))
                draw.line((0, i, canvas_w, i), fill=color, width=random.randint(5, 15))

        elif bg_type == "fabric":
            # Fabric pattern
            base = random.choice([(100, 100, 150), (150, 100, 100), (100, 150, 100)])
            canvas = Image.new("RGB", (canvas_w, canvas_h), base)
            np_canvas = np.array(canvas)
            noise = np.random.randint(-15, 15, np_canvas.shape, dtype=np.int16)
            np_canvas = np.clip(np_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(np_canvas)

        elif bg_type == "desk":
            # Office desk (gray/beige)
            canvas = Image.new("RGB", (canvas_w, canvas_h),
                             random.choice([(180, 170, 160), (160, 160, 170), (200, 195, 180)]))

        else:  # marble
            canvas = Image.new("RGB", (canvas_w, canvas_h), (230, 230, 235))
            draw = ImageDraw.Draw(canvas)
            for _ in range(20):
                x1 = random.randint(0, canvas_w)
                y1 = random.randint(0, canvas_h)
                x2 = x1 + random.randint(50, 200)
                y2 = y1 + random.randint(2, 5)
                draw.line((x1, y1, x2, y2), fill=(200, 200, 210), width=1)

        # Add shadow under receipt
        shadow_offset = 5
        shadow = Image.new("RGBA", (w + 10, h + 10), (0, 0, 0, 50))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        canvas.paste(shadow, (25 + shadow_offset, 25 + shadow_offset), shadow)

        # Paste receipt
        x = random.randint(20, canvas_w - w - 20) if canvas_w > w + 40 else 20
        y = random.randint(20, canvas_h - h - 20) if canvas_h > h + 40 else 20

        canvas.paste(img, (x, y))

        # Return offset for bbox adjustment
        return canvas, {"background": bg_type, "offset_x": x, "offset_y": y}

    # ============= CRUMPLED =============

    def _apply_crumpled(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Simulate a crumpled receipt."""
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGB")
        cv_img = np.array(img)
        h, w = cv_img.shape[:2]

        # Apply wave distortion
        for i in range(3):
            amplitude = random.randint(2, 8)
            frequency = random.uniform(0.02, 0.05)

            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)

            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x + amplitude * math.sin(frequency * y)
                    map_y[y, x] = y + amplitude * math.sin(frequency * x)

            cv_img = cv2.remap(cv_img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Add shadow patterns for creases
        overlay = Image.fromarray(cv_img)
        draw = ImageDraw.Draw(overlay, "RGBA")

        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = x1 + random.randint(-100, 100)
            y2 = y1 + random.randint(-100, 100)
            draw.line((x1, y1, x2, y2), fill=(100, 100, 100, 30), width=random.randint(5, 15))

        return overlay.convert("RGB"), {"crumpled": True}

    # ============= FOLDED CORNER =============

    def _apply_folded_corner(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Add a folded corner effect."""
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGBA")
        w, h = img.size

        # Choose corner
        corner = random.choice(["top_right", "top_left", "bottom_right", "bottom_left"])
        fold_size = random.randint(30, 80)

        draw = ImageDraw.Draw(img)

        if corner == "top_right":
            triangle = [(w, 0), (w - fold_size, 0), (w, fold_size)]
            draw.polygon(triangle, fill=(220, 220, 210, 255))
            # Shadow
            draw.line((w - fold_size, 0, w, fold_size), fill=(150, 150, 150), width=2)

        elif corner == "top_left":
            triangle = [(0, 0), (fold_size, 0), (0, fold_size)]
            draw.polygon(triangle, fill=(220, 220, 210, 255))
            draw.line((fold_size, 0, 0, fold_size), fill=(150, 150, 150), width=2)

        elif corner == "bottom_right":
            triangle = [(w, h), (w - fold_size, h), (w, h - fold_size)]
            draw.polygon(triangle, fill=(220, 220, 210, 255))
            draw.line((w - fold_size, h, w, h - fold_size), fill=(150, 150, 150), width=2)

        else:  # bottom_left
            triangle = [(0, h), (fold_size, h), (0, h - fold_size)]
            draw.polygon(triangle, fill=(220, 220, 210, 255))
            draw.line((fold_size, h, 0, h - fold_size), fill=(150, 150, 150), width=2)

        return img.convert("RGB"), {"folded_corner": corner}

    # ============= SIZE VARIATIONS =============

    def _apply_very_small(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Generate a very small resolution image."""
        if img is None:
            return self._generate_placeholder(), {}

        w, h = img.size
        scale = random.uniform(0.2, 0.4)

        new_w = max(100, int(w * scale))
        new_h = max(150, int(h * scale))

        # Use low quality resize
        result = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        return result, {"scale": scale}

    def _apply_very_large(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Generate a very large resolution image."""
        if img is None:
            return self._generate_placeholder(), {}

        w, h = img.size
        scale = random.uniform(2.0, 4.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        result = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return result, {"scale": scale}

    # ============= CONTRAST/COLOR =============

    def _apply_low_contrast(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Create a very low contrast image."""
        if img is None:
            return self._generate_placeholder(), {}

        enhancer = ImageEnhance.Contrast(img)
        contrast = random.uniform(0.2, 0.5)
        result = enhancer.enhance(contrast)

        # Also adjust brightness
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(random.uniform(0.9, 1.2))

        return result, {"contrast": contrast}

    def _apply_inverted(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """Invert colors (negative)."""
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGB")
        np_img = np.array(img)
        inverted = 255 - np_img

        return Image.fromarray(inverted), {"inverted": True}

    # =========================================================================
    # NEW PHASE 2 REALISTIC EFFECTS
    # =========================================================================

    def _apply_hand_holding(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """
        Simulate a hand holding the receipt.
        
        Common in user photos where fingers/thumbs obscure edges.
        VLM must learn to separate finger pixels from text.
        """
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGBA")
        w, h = img.size

        # Create finger overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Skin color variations
        skin_colors = [
            (255, 224, 189),  # Light skin
            (241, 194, 125),  # Medium light
            (198, 134, 66),   # Medium
            (141, 85, 36),    # Medium dark
            (87, 57, 28),     # Dark
        ]
        skin_color = random.choice(skin_colors)

        # How many fingers (1-4)
        num_fingers = random.randint(1, 4)
        
        # Which edge (fingers usually from left/right/bottom)
        edge = random.choice(["left", "right", "bottom"])
        
        for i in range(num_fingers):
            finger_width = random.randint(25, 45)
            finger_length = random.randint(60, 120)
            
            # Add slight color variation per finger
            color_variation = random.randint(-15, 15)
            finger_color = tuple(max(0, min(255, c + color_variation)) for c in skin_color)
            
            if edge == "left":
                x = random.randint(-finger_width // 2, 10)
                y = random.randint(h // 4, 3 * h // 4) + i * random.randint(-30, 30)
                # Horizontal finger from left
                finger_box = (x, y, x + finger_length, y + finger_width)
                draw.ellipse(finger_box, fill=(*finger_color, 255))
                # Fingernail
                nail_color = (255, 220, 220)
                nail_box = (x + finger_length - 15, y + 5, x + finger_length - 2, y + finger_width - 5)
                draw.ellipse(nail_box, fill=(*nail_color, 220))
                
            elif edge == "right":
                x = w - random.randint(10, 50)
                y = random.randint(h // 4, 3 * h // 4) + i * random.randint(-30, 30)
                # Horizontal finger from right
                finger_box = (x - finger_length, y, x, y + finger_width)
                draw.ellipse(finger_box, fill=(*finger_color, 255))
                # Fingernail
                nail_color = (255, 220, 220)
                nail_box = (x - finger_length + 2, y + 5, x - finger_length + 15, y + finger_width - 5)
                draw.ellipse(nail_box, fill=(*nail_color, 220))
                
            else:  # bottom - thumb usually
                x = random.randint(10, w - 60) + i * random.randint(20, 50)
                y = h - random.randint(10, 40)
                # Vertical thumb from bottom
                thumb_width = random.randint(40, 60)
                thumb_length = random.randint(80, 140)
                finger_box = (x, y - thumb_length + 20, x + thumb_width, y + 20)
                draw.ellipse(finger_box, fill=(*finger_color, 255))
        
        # Add slight blur for realism
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1))
        
        result = Image.alpha_composite(img, overlay)
        return result.convert("RGB"), {"hand_holding": True, "num_fingers": num_fingers, "edge": edge}

    def _apply_realistic_background(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """
        Place receipt on realistic background.
        
        Simulates photos taken on cafe tables, car dashboards, bedsheets, etc.
        These are common scenarios for receipt capture apps.
        """
        if img is None:
            return self._generate_placeholder(), {}

        w, h = img.size
        
        # Canvas size (receipt + margins)
        margin = random.randint(40, 100)
        canvas_w = w + margin * 2
        canvas_h = h + margin * 2

        # Background type with realistic textures
        bg_type = random.choice([
            "cafe_table", "dark_wood", "glass_surface", 
            "car_interior", "bedsheet", "kitchen_counter",
            "office_desk", "marble", "leather"
        ])

        if bg_type == "cafe_table":
            # Wood grain texture
            base_color = random.choice([(139, 90, 43), (160, 120, 80), (101, 67, 33)])
            canvas = Image.new("RGB", (canvas_w, canvas_h), base_color)
            np_canvas = np.array(canvas, dtype=np.float32)
            # Add grain
            for i in range(canvas_h):
                variation = random.uniform(0.9, 1.1)
                np_canvas[i, :] = np_canvas[i, :] * variation
            canvas = Image.fromarray(np.clip(np_canvas, 0, 255).astype(np.uint8))

        elif bg_type == "dark_wood":
            canvas = Image.new("RGB", (canvas_w, canvas_h), (45, 30, 20))
            draw = ImageDraw.Draw(canvas)
            for i in range(0, canvas_h, random.randint(3, 8)):
                color = (45 + random.randint(-10, 10), 
                        30 + random.randint(-10, 10), 
                        20 + random.randint(-5, 5))
                draw.line((0, i, canvas_w, i), fill=color, width=random.randint(2, 5))

        elif bg_type == "glass_surface":
            # Translucent blue-ish
            canvas = Image.new("RGB", (canvas_w, canvas_h), (220, 235, 245))
            draw = ImageDraw.Draw(canvas)
            # Reflections
            for _ in range(random.randint(5, 15)):
                x1 = random.randint(0, canvas_w)
                y1 = random.randint(0, canvas_h)
                draw.line((x1, y1, x1 + random.randint(50, 200), y1 + random.randint(-20, 20)),
                         fill=(255, 255, 255), width=random.randint(1, 3))

        elif bg_type == "car_interior":
            # Dark gray/black leather look
            canvas = Image.new("RGB", (canvas_w, canvas_h), (50, 50, 55))
            np_canvas = np.array(canvas)
            noise = np.random.randint(-10, 10, np_canvas.shape, dtype=np.int16)
            np_canvas = np.clip(np_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(np_canvas)

        elif bg_type == "bedsheet":
            # Light fabric texture
            colors = [(255, 250, 245), (245, 240, 235), (240, 245, 250)]
            canvas = Image.new("RGB", (canvas_w, canvas_h), random.choice(colors))
            np_canvas = np.array(canvas)
            noise = np.random.randint(-8, 8, np_canvas.shape, dtype=np.int16)
            np_canvas = np.clip(np_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(np_canvas)
            # Add subtle wrinkles
            draw = ImageDraw.Draw(canvas)
            for _ in range(random.randint(3, 8)):
                x1 = random.randint(0, canvas_w)
                y1 = random.randint(0, canvas_h)
                draw.line((x1, y1, x1 + random.randint(-100, 100), y1 + random.randint(50, 150)),
                         fill=(220, 215, 210), width=random.randint(1, 3))

        elif bg_type == "kitchen_counter":
            # Granite/speckled look
            canvas = Image.new("RGB", (canvas_w, canvas_h), (180, 175, 170))
            np_canvas = np.array(canvas)
            noise = np.random.randint(-25, 25, np_canvas.shape, dtype=np.int16)
            np_canvas = np.clip(np_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(np_canvas)

        elif bg_type == "office_desk":
            canvas = Image.new("RGB", (canvas_w, canvas_h), 
                             random.choice([(200, 195, 185), (180, 170, 160), (220, 215, 200)]))

        elif bg_type == "marble":
            canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 245))
            draw = ImageDraw.Draw(canvas)
            # Veins
            for _ in range(random.randint(5, 12)):
                points = []
                x, y = random.randint(0, canvas_w), random.randint(0, canvas_h)
                for _ in range(random.randint(5, 15)):
                    x += random.randint(-30, 30)
                    y += random.randint(-30, 30)
                    points.append((x, y))
                if len(points) >= 2:
                    draw.line(points, fill=(180, 180, 190), width=random.randint(1, 2))

        else:  # leather
            canvas = Image.new("RGB", (canvas_w, canvas_h), (60, 40, 30))
            np_canvas = np.array(canvas)
            noise = np.random.randint(-10, 10, np_canvas.shape, dtype=np.int16)
            np_canvas = np.clip(np_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(np_canvas)

        # Add shadow under receipt
        shadow = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 60))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
        shadow_offset = random.randint(5, 15)
        canvas.paste(shadow, (margin - 10 + shadow_offset, margin - 10 + shadow_offset), shadow)

        # Paste receipt with slight rotation
        angle = random.uniform(-5, 5)
        img_rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        x = margin + random.randint(-10, 10)
        y = margin + random.randint(-10, 10)
        canvas.paste(img_rotated, (x, y))

        return canvas, {"realistic_background": bg_type, "offset_x": x, "offset_y": y, "rotation": angle}

    def _apply_tps_warp(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """
        Apply Thin Plate Spline (TPS) warping for realistic 3D curvature.
        
        Creates more natural-looking deformations than simple wave distortion.
        Simulates paper bending/curving when held or placed on uneven surfaces.
        """
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGB")
        np_img = np.array(img)
        h, w = np_img.shape[:2]

        # Define control points (source and destination)
        num_control_points = random.randint(4, 8)
        
        # Create grid of source points
        src_pts = []
        dst_pts = []
        
        # Add corner points with slight displacement
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        for cx, cy in corners:
            src_pts.append([cx, cy])
            # Random displacement
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            dst_pts.append([cx + dx, cy + dy])
        
        # Add random interior control points
        for _ in range(num_control_points - 4):
            x = random.randint(w // 4, 3 * w // 4)
            y = random.randint(h // 4, 3 * h // 4)
            src_pts.append([x, y])
            # Larger displacement for interior points (bulging effect)
            dx = random.randint(-25, 25)
            dy = random.randint(-25, 25)
            dst_pts.append([x + dx, y + dy])
        
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        # Use OpenCV's ThinPlateSplineShapeTransformer if available
        # Fallback to a simpler polynomial warp
        try:
            # Create coordinate maps using polynomial approximation
            # (TPS is complex, so we use a simplified approximation)
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    # Find weighted displacement based on control points
                    total_weight = 0
                    dx_sum = 0
                    dy_sum = 0
                    
                    for i, (sx, sy) in enumerate(src_pts):
                        # Distance from this pixel to control point
                        dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2) + 1
                        weight = 1.0 / (dist * dist)
                        
                        # Displacement of this control point
                        ddx = dst_pts[i][0] - sx
                        ddy = dst_pts[i][1] - sy
                        
                        dx_sum += weight * ddx
                        dy_sum += weight * ddy
                        total_weight += weight
                    
                    # Average displacement
                    if total_weight > 0:
                        map_x[y, x] = x + (dx_sum / total_weight) * 0.5
                        map_y[y, x] = y + (dy_sum / total_weight) * 0.5
                    else:
                        map_x[y, x] = x
                        map_y[y, x] = y
            
            result = cv2.remap(np_img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            
            return Image.fromarray(result), {"tps_warp": True, "control_points": len(src_pts)}
            
        except Exception as e:
            # Fallback: just return with slight perspective
            return img, {"tps_warp": False, "error": str(e)}

    def _apply_cylindrical_curl(self, img: Image.Image) -> Tuple[Image.Image, Dict]:
        """
        Apply cylindrical curl effect - like a freshly printed thermal receipt.
        
        Thermal paper naturally curls when printed due to heat differential.
        Creates a characteristic cylinder-like curvature.
        """
        if img is None:
            return self._generate_placeholder(), {}

        img = img.convert("RGB")
        np_img = np.array(img)
        h, w = np_img.shape[:2]

        # Curl direction (usually along the length of thermal receipts)
        curl_direction = random.choice(["horizontal", "vertical"])
        curl_intensity = random.uniform(0.1, 0.3)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        if curl_direction == "horizontal":
            # Receipt curls left-right (like a scroll)
            for y in range(h):
                for x in range(w):
                    # Cylindrical mapping - x displacement based on y position
                    # Center of curl
                    center_y = h / 2
                    # Distance from center (normalized)
                    dist_from_center = (y - center_y) / center_y
                    
                    # Curl creates horizontal compression/expansion
                    curl_factor = curl_intensity * (1 - dist_from_center ** 2)
                    
                    # X mapping (slight squeeze toward center)
                    map_x[y, x] = x + (w/2 - x) * curl_factor * 0.2
                    map_y[y, x] = y
                    
        else:  # vertical curl
            # Receipt curls top-bottom
            for y in range(h):
                for x in range(w):
                    center_x = w / 2
                    dist_from_center = (x - center_x) / center_x
                    
                    curl_factor = curl_intensity * (1 - dist_from_center ** 2)
                    
                    map_x[y, x] = x
                    map_y[y, x] = y + (h/2 - y) * curl_factor * 0.2

        result = cv2.remap(np_img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Add subtle shadow gradient to simulate 3D depth
        result_img = Image.fromarray(result)
        draw = ImageDraw.Draw(result_img, "RGBA")
        
        if curl_direction == "horizontal":
            # Top and bottom edges darker
            for i in range(20):
                alpha = int(30 * (1 - i / 20))
                draw.line((0, i, w, i), fill=(0, 0, 0, alpha))
                draw.line((0, h - 1 - i, w, h - 1 - i), fill=(0, 0, 0, alpha))
        else:
            # Left and right edges darker
            for i in range(20):
                alpha = int(30 * (1 - i / 20))
                draw.line((i, 0, i, h), fill=(0, 0, 0, alpha))
                draw.line((w - 1 - i, 0, w - 1 - i, h), fill=(0, 0, 0, alpha))

        return result_img.convert("RGB"), {"cylindrical_curl": curl_direction, "intensity": curl_intensity}

    # ============= HELPER =============

    def _generate_placeholder(self) -> Image.Image:
        """Generate a placeholder image."""
        w = random.randint(300, 600)
        h = random.randint(400, 800)
        return Image.new("RGB", (w, h), (200, 200, 200))

