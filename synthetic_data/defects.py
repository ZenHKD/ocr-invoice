"""
Visual Defects Simulator for realistic document scanning artifacts.

Provides:
    - Folds and creases
    - Stains (coffee, water, oil)
    - Tears and edge damage
    - Scanner shadows and lighting
    - Faded/overexposed regions
    - Skew and perspective distortion
    - Compression artifacts
    - Handwritten annotations/marks
    - Staple holes and paper clips shadows
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
import cv2


class DefectType(Enum):
    FOLD = "fold"
    CREASE = "crease"
    COFFEE_STAIN = "coffee_stain"
    WATER_STAIN = "water_stain"
    OIL_STAIN = "oil_stain"
    TEAR = "tear"
    EDGE_DAMAGE = "edge_damage"
    SCANNER_SHADOW = "scanner_shadow"
    LIGHTING_UNEVEN = "lighting_uneven"
    FADE = "fade"
    OVEREXPOSE = "overexpose"
    SKEW = "skew"
    PERSPECTIVE = "perspective"
    BLUR_LOCAL = "blur_local"
    NOISE = "noise"
    COMPRESSION = "compression"
    HANDWRITING = "handwriting"
    STAPLE_HOLE = "staple_hole"
    PAPER_CLIP = "paper_clip"
    FINGERPRINT = "fingerprint"
    DUST_SPECKS = "dust_specks"
    # New realistic defects (Phase 1 & 3)
    THERMAL_STREAK = "thermal_streak"  # Vertical white streaks from dead pixels
    INK_FADE_HORIZONTAL = "ink_fade_horizontal"  # Ink fading to one side
    RED_STAMP = "red_stamp"  # Official red company/tax stamp
    HANDWRITING_OVERLAP = "handwriting_overlap"  # Pen marks over printed text



@dataclass
class DefectConfig:
    """Configuration for applying defects."""
    min_defects: int = 0
    max_defects: int = 5
    intensity: str = "medium"  # light, medium, heavy, extreme
    defect_probabilities: dict = None

    def __post_init__(self):
        if self.defect_probabilities is None:
            self.defect_probabilities = {
                DefectType.FOLD: 0.15,
                DefectType.CREASE: 0.2,
                DefectType.COFFEE_STAIN: 0.1,
                DefectType.WATER_STAIN: 0.08,
                DefectType.OIL_STAIN: 0.05,
                DefectType.TEAR: 0.08,
                DefectType.EDGE_DAMAGE: 0.15,
                DefectType.SCANNER_SHADOW: 0.25,
                DefectType.LIGHTING_UNEVEN: 0.3,
                DefectType.FADE: 0.2,
                DefectType.OVEREXPOSE: 0.1,
                DefectType.SKEW: 0.4,
                DefectType.PERSPECTIVE: 0.15,
                DefectType.BLUR_LOCAL: 0.2,
                DefectType.NOISE: 0.5,
                DefectType.COMPRESSION: 0.4,
                DefectType.HANDWRITING: 0.15,
                DefectType.STAPLE_HOLE: 0.12,
                DefectType.PAPER_CLIP: 0.08,
                DefectType.FINGERPRINT: 0.1,
                DefectType.DUST_SPECKS: 0.3,
                # New realistic defects
                DefectType.THERMAL_STREAK: 0.25,  # Common in thermal printers
                DefectType.INK_FADE_HORIZONTAL: 0.2,  # Ink running low
                DefectType.RED_STAMP: 0.15,  # VAT invoices often have stamps
                DefectType.HANDWRITING_OVERLAP: 0.2,  # Staff annotations
            }


class DefectApplicator:
    """Apply various visual defects to invoice images."""

    def __init__(self, config: DefectConfig = None):
        self.config = config or DefectConfig()

    def apply_random_defects(self, img: Image.Image) -> Tuple[Image.Image, List[DefectType], List[Dict]]:
        """Apply random defects based on configuration."""
        applied_defects = []
        transforms = []
        num_defects = random.randint(self.config.min_defects, self.config.max_defects)

        # Select defects based on probabilities
        available_defects = []
        for defect_type, prob in self.config.defect_probabilities.items():
            if random.random() < prob:
                available_defects.append(defect_type)

        # Limit to num_defects
        random.shuffle(available_defects)
        defects_to_apply = available_defects[:num_defects]

        for defect in defects_to_apply:
            img, metadata = self._apply_defect(img, defect)
            applied_defects.append(defect)
            if metadata:
                transforms.append(metadata)

        return img, applied_defects, transforms

    def apply_specific_defects(self, img: Image.Image,
                                defects: List[DefectType]) -> Tuple[Image.Image, List[Dict]]:
        """Apply specific defects."""
        transforms = []
        for defect in defects:
            img, metadata = self._apply_defect(img, defect)
            if metadata:
                transforms.append(metadata)
        return img, transforms

    def _apply_defect(self, img: Image.Image, defect: DefectType) -> Tuple[Image.Image, Optional[Dict]]:
        """Apply a single defect."""
        method = getattr(self, f"_apply_{defect.value}", None)
        if method:
            result = method(img)
            # Handle methods that return metadata
            if isinstance(result, tuple):
                return result
            return result, None
        return img, None

    def _apply_fold(self, img: Image.Image) -> Image.Image:
        """Apply a fold/crease line."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Fold can be horizontal or vertical
        if random.random() < 0.5:
            # Horizontal fold
            y = random.randint(h // 4, 3 * h // 4)
            fold_width = random.randint(3, 8)

            for offset in range(fold_width):
                alpha = int(50 * (1 - offset / fold_width))
                color = (128 - alpha, 128 - alpha, 128 - alpha)
                draw.line((0, y + offset, w, y + offset), fill=color)
        else:
            # Vertical fold
            x = random.randint(w // 4, 3 * w // 4)
            fold_width = random.randint(3, 8)

            for offset in range(fold_width):
                alpha = int(50 * (1 - offset / fold_width))
                color = (128 - alpha, 128 - alpha, 128 - alpha)
                draw.line((x + offset, 0, x + offset, h), fill=color)

        return img

    def _apply_crease(self, img: Image.Image) -> Image.Image:
        """Apply diagonal crease lines."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Random diagonal
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(w // 2, w)
        y2 = random.randint(h // 2, h)

        crease_color = random.choice([
            (200, 200, 200),
            (180, 180, 180),
            (160, 160, 160),
        ])

        for _ in range(random.randint(2, 4)):
            offset = random.randint(-2, 2)
            draw.line((x1 + offset, y1, x2 + offset, y2), fill=crease_color, width=1)

        return img

    def _apply_coffee_stain(self, img: Image.Image) -> Image.Image:
        """Apply coffee ring stain."""
        img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Coffee ring parameters
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(30, 80)

        # Multiple concentric circles for ring effect
        for r in range(radius - 10, radius + 5):
            alpha = random.randint(15, 40)
            color = (139, 90, 43, alpha)  # Coffee brown
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)

        # Fill center lightly
        inner_alpha = random.randint(5, 20)
        draw.ellipse(
            (cx - radius + 15, cy - radius + 15, cx + radius - 15, cy + radius - 15),
            fill=(139, 90, 43, inner_alpha)
        )

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _apply_water_stain(self, img: Image.Image) -> Image.Image:
        """Apply water damage stain."""
        img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Irregular water stain shape
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)

        # Create irregular shape with multiple ellipses
        for _ in range(random.randint(3, 7)):
            offset_x = random.randint(-30, 30)
            offset_y = random.randint(-30, 30)
            rx = random.randint(20, 60)
            ry = random.randint(20, 60)
            alpha = random.randint(10, 30)

            draw.ellipse(
                (cx + offset_x - rx, cy + offset_y - ry,
                 cx + offset_x + rx, cy + offset_y + ry),
                fill=(150, 150, 130, alpha)
            )

        img = Image.alpha_composite(img, overlay)

        # Slightly warp the area
        result = img.convert("RGB")
        return result

    def _apply_oil_stain(self, img: Image.Image) -> Image.Image:
        """Apply oily/greasy stain."""
        img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(15, 40)

        # Translucent spot
        for r in range(radius, 0, -3):
            alpha = int(15 * (radius - r) / radius)
            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=(200, 200, 180, alpha)
            )

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _apply_tear(self, img: Image.Image) -> Image.Image:
        """Apply torn edge effect."""
        img = img.convert("RGB")
        cv_img = np.array(img)

        h, w = cv_img.shape[:2]

        # Create a tear mask
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Random edge to tear
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge == "top":
            for x in range(w):
                tear_depth = random.randint(0, 20)
                mask[:tear_depth, x] = 0
        elif edge == "bottom":
            for x in range(w):
                tear_depth = random.randint(0, 20)
                mask[h - tear_depth:, x] = 0
        elif edge == "left":
            for y in range(h):
                tear_depth = random.randint(0, 15)
                mask[y, :tear_depth] = 0
        else:  # right
            for y in range(h):
                tear_depth = random.randint(0, 15)
                mask[y, w - tear_depth:] = 0

        # Apply white background where torn
        cv_img[mask == 0] = [255, 255, 255]

        return Image.fromarray(cv_img)

    def _apply_edge_damage(self, img: Image.Image) -> Image.Image:
        """Apply worn/damaged edges."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        edge_color = (240, 238, 230)

        # Corner damage
        corners = [
            (0, 0, random.randint(10, 30), random.randint(10, 30)),
            (w - random.randint(10, 30), 0, w, random.randint(10, 30)),
            (0, h - random.randint(10, 30), random.randint(10, 30), h),
            (w - random.randint(10, 30), h - random.randint(10, 30), w, h),
        ]

        for i, corner in enumerate(corners):
            if random.random() < 0.5:
                draw.rectangle(corner, fill=edge_color)

        return img

    def _apply_scanner_shadow(self, img: Image.Image) -> Image.Image:
        """Apply scanner edge shadows."""
        img = img.convert("RGB")
        w, h = img.size

        # Create gradient overlay
        np_img = np.array(img, dtype=np.float32)

        # Left shadow
        if random.random() < 0.5:
            shadow_width = random.randint(20, 60)
            for x in range(shadow_width):
                factor = 0.7 + 0.3 * (x / shadow_width)
                np_img[:, x] = np_img[:, x] * factor

        # Right shadow
        if random.random() < 0.5:
            shadow_width = random.randint(20, 60)
            for x in range(shadow_width):
                factor = 0.7 + 0.3 * (x / shadow_width)
                np_img[:, w - 1 - x] = np_img[:, w - 1 - x] * factor

        # Top shadow
        if random.random() < 0.3:
            shadow_height = random.randint(15, 40)
            for y in range(shadow_height):
                factor = 0.8 + 0.2 * (y / shadow_height)
                np_img[y, :] = np_img[y, :] * factor

        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def _apply_lighting_uneven(self, img: Image.Image) -> Image.Image:
        """Apply uneven lighting/exposure."""
        img = img.convert("RGB")
        w, h = img.size
        np_img = np.array(img, dtype=np.float32)

        # Create a gradient mask
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)

        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2)

        # Normalize and create brightness mask
        mask = 1 - (distances / max_dist) * random.uniform(0.1, 0.3)
        mask = mask[:, :, np.newaxis]

        np_img = np_img * mask
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)

        return Image.fromarray(np_img)

    def _apply_fade(self, img: Image.Image) -> Image.Image:
        """Apply faded/washed out effect."""
        img = img.convert("RGB")

        # Reduce contrast and add brightness
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.5, 0.8))

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(1.1, 1.3))

        return img

    def _apply_overexpose(self, img: Image.Image) -> Image.Image:
        """Apply overexposed region."""
        img = img.convert("RGB")
        w, h = img.size
        np_img = np.array(img, dtype=np.float32)

        # Random overexposed region
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + 50, w)
        y2 = random.randint(y1 + 50, h)

        # Increase brightness in region
        factor = random.uniform(1.3, 1.6)
        np_img[y1:y2, x1:x2] = np_img[y1:y2, x1:x2] * factor

        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def _apply_skew(self, img: Image.Image) -> Image.Image:
        """Apply rotation/skew."""
        img = img.convert("RGB")
        cv_img = np.array(img)
        h, w = cv_img.shape[:2]

        angle = random.uniform(-8, 8)

        # Get rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Apply rotation with white background
        rotated = cv2.warpAffine(cv_img, M, (new_w, new_h),
                                  borderValue=(255, 255, 255))

        # Convert 2x3 affine to 3x3 for consistency
        M_3x3 = np.eye(3)
        M_3x3[:2, :] = M

        return Image.fromarray(rotated), {
            "type": "matrix",
            "matrix": M_3x3.tolist(),
            "new_size": (new_w, new_h)
        }

    def _apply_perspective(self, img: Image.Image) -> Image.Image:
        """Apply perspective distortion."""
        img = img.convert("RGB")
        cv_img = np.array(img)
        h, w = cv_img.shape[:2]

        # Define source points
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Random perspective shift
        shift = random.randint(5, 25)
        dst_pts = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), h - random.randint(0, shift)],
            [random.randint(0, shift), h - random.randint(0, shift)]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(cv_img, M, (w, h),
                                      borderValue=(255, 255, 255))

        return Image.fromarray(result), {
            "type": "matrix",
            "matrix": M.tolist(),
            "new_size": (w, h)
        }

    def _apply_blur_local(self, img: Image.Image) -> Image.Image:
        """Apply localized blur."""
        img = img.convert("RGB")
        w, h = img.size

        # Create a blurred version
        blurred = img.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 5)))

        # Apply blur to random region
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + 50, min(x1 + 200, w))
        y2 = random.randint(y1 + 50, min(y1 + 200, h))

        region = blurred.crop((x1, y1, x2, y2))
        img.paste(region, (x1, y1))

        return img

    def _apply_noise(self, img: Image.Image) -> Image.Image:
        """Apply various noise types."""
        img = img.convert("RGB")
        np_img = np.array(img, dtype=np.float32)

        noise_type = random.choice(["gaussian", "salt_pepper", "speckle"])

        if noise_type == "gaussian":
            noise = np.random.normal(0, random.randint(5, 20), np_img.shape)
            np_img = np_img + noise
        elif noise_type == "salt_pepper":
            prob = random.uniform(0.01, 0.05)
            mask = np.random.random(np_img.shape[:2])
            np_img[mask < prob / 2] = 0
            np_img[mask > 1 - prob / 2] = 255
        else:  # speckle
            noise = np.random.randn(*np_img.shape) * 0.1
            np_img = np_img + np_img * noise

        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def _apply_compression(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts."""
        img = img.convert("RGB")
        cv_img = np.array(img)

        quality = random.randint(15, 50)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        _, encoded = cv2.imencode(".jpg", cv_img, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        return Image.fromarray(decoded)

    def _apply_handwriting(self, img: Image.Image) -> Image.Image:
        """Apply handwritten annotations."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        annotations = [
            "✓", "OK", "X", "?", "!", "→",
            "đã nhận", "trả", "chờ", "xem lại",
            str(random.randint(1, 100)),
        ]

        text = random.choice(annotations)
        x = random.randint(w // 2, w - 50)
        y = random.randint(10, h - 50)

        ink_color = random.choice([
            (0, 0, 150),    # Blue
            (150, 0, 0),    # Red
            (0, 100, 0),    # Green
        ])

        try:
            font = ImageFont.truetype("Comic Sans MS", random.randint(16, 28))
        except:
            font = ImageFont.load_default()

        draw.text((x, y), text, font=font, fill=ink_color)

        return img

    def _apply_staple_hole(self, img: Image.Image) -> Image.Image:
        """Apply staple holes."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        h = img.size[1]

        # Staple position - usually top left
        x = random.randint(10, 40)
        y = random.randint(20, 60)

        # Two holes
        hole_color = (80, 80, 80)
        draw.ellipse((x, y, x + 3, y + 8), fill=hole_color)
        draw.ellipse((x + 15, y, x + 18, y + 8), fill=hole_color)

        return img

    def _apply_paper_clip(self, img: Image.Image) -> Image.Image:
        """Apply paper clip shadow."""
        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Position - usually top
        x = random.randint(w - 80, w - 30)
        y = random.randint(10, 50)

        # Simple clip shadow shape
        clip_color = (100, 100, 100, 80)
        draw.rectangle((x, y, x + 25, y + 60), fill=clip_color)
        draw.rectangle((x + 5, y + 10, x + 20, y + 50), fill=(255, 255, 255, 200))

        return img.convert("RGB")

    def _apply_fingerprint(self, img: Image.Image) -> Image.Image:
        """Apply fingerprint smudge."""
        img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cx = random.randint(50, w - 50)
        cy = random.randint(50, h - 50)

        # Fingerprint is made of concentric ellipses
        for i in range(random.randint(8, 15)):
            rx = random.randint(10, 30) + i * 2
            ry = random.randint(8, 25) + i * 2
            alpha = random.randint(3, 10)

            draw.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry),
                outline=(150, 150, 150, alpha),
                width=1
            )

        # Slight blur
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1))

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _apply_dust_specks(self, img: Image.Image) -> Image.Image:
        """Apply dust specks."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        num_specks = random.randint(5, 30)

        for _ in range(num_specks):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(1, 3)
            color = random.choice([
                (50, 50, 50),
                (100, 100, 100),
                (80, 80, 80),
            ])
            draw.ellipse((x, y, x + size, y + size), fill=color)

        return img

    # =========================================================================
    # NEW REALISTIC DEFECTS (Phase 1 & 3)
    # =========================================================================

    def _apply_thermal_streak(self, img: Image.Image) -> Image.Image:
        """
        Apply vertical white streaks simulating dead pixels in thermal printers.
        
        Common in older Vietnamese receipt printers where print heads have
        dead pixels that create characteristic vertical white lines.
        """
        img = img.convert("RGB")
        np_img = np.array(img, dtype=np.float32)
        h, w = np_img.shape[:2]
        
        # Number of streaks (1-4)
        num_streaks = random.randint(1, 4)
        
        for _ in range(num_streaks):
            # Random x position for vertical streak
            x = random.randint(10, w - 10)
            streak_width = random.randint(1, 3)
            
            # Streak intensity varies - some are pure white, some are faded
            intensity = random.uniform(0.7, 1.0)
            
            # Apply fade along the streak (not uniform)
            for y in range(h):
                # Vary intensity along y-axis
                local_intensity = intensity * random.uniform(0.8, 1.0)
                for dx in range(streak_width):
                    if 0 <= x + dx < w:
                        # Blend toward white
                        np_img[y, x + dx] = np_img[y, x + dx] * (1 - local_intensity) + 255 * local_intensity
        
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def _apply_ink_fade_horizontal(self, img: Image.Image) -> Image.Image:
        """
        Apply horizontal ink fade effect - one side of receipt is lighter.
        
        Simulates thermal printers running low on thermal coating or
        uneven pressure on the print head, common in Vietnamese receipts.
        """
        img = img.convert("RGB")
        np_img = np.array(img, dtype=np.float32)
        h, w = np_img.shape[:2]
        
        # Decide which side fades (left or right)
        fade_from_left = random.choice([True, False])
        
        # Fade intensity (how much lighter the faded side gets)
        max_fade = random.uniform(0.3, 0.6)
        
        # Create gradient mask
        for x in range(w):
            if fade_from_left:
                # Left side is faded
                fade_factor = max_fade * (1 - x / w)
            else:
                # Right side is faded
                fade_factor = max_fade * (x / w)
            
            # Apply fade (blend toward white)
            np_img[:, x] = np_img[:, x] * (1 - fade_factor) + 255 * fade_factor
        
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def _apply_red_stamp(self, img: Image.Image) -> Image.Image:
        """
        Apply official red company/tax stamp overlay.
        
        Vietnamese VAT invoices and restaurant bills often have round or
        square red stamps that partially obscure text - a nightmare for OCR.
        """
        img = img.convert("RGBA")
        w, h = img.size
        
        # Create stamp overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Stamp position (usually bottom half, right side for VAT invoices)
        cx = random.randint(w // 2, w - 80)
        cy = random.randint(h // 2, h - 80)
        
        # Stamp type: round or square
        stamp_type = random.choice(["round", "square"])
        stamp_size = random.randint(60, 120)
        
        # Red color with transparency (looks like ink stamp)
        red_colors = [
            (200, 30, 30),   # Dark red
            (220, 50, 50),   # Medium red
            (180, 20, 20),   # Very dark red
        ]
        stamp_color = random.choice(red_colors)
        alpha = random.randint(80, 150)  # Semi-transparent
        
        if stamp_type == "round":
            # Draw concentric circles for stamp border
            for r in range(3):
                radius = stamp_size - r * 3
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=(*stamp_color, alpha),
                    width=2
                )
            
            # Add some text-like patterns inside (company name simulation)
            inner_radius = stamp_size - 15
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x1 = cx + int(inner_radius * 0.6 * math.cos(rad))
                y1 = cy + int(inner_radius * 0.6 * math.sin(rad))
                x2 = cx + int(inner_radius * 0.8 * math.cos(rad))
                y2 = cy + int(inner_radius * 0.8 * math.sin(rad))
                draw.line((x1, y1, x2, y2), fill=(*stamp_color, alpha // 2), width=2)
            
            # Center star or symbol
            star_size = stamp_size // 4
            draw.regular_polygon((cx, cy, star_size), 5, fill=(*stamp_color, alpha // 2))
            
        else:  # square
            # Draw square stamp
            x1 = cx - stamp_size // 2
            y1 = cy - stamp_size // 2
            x2 = cx + stamp_size // 2
            y2 = cy + stamp_size // 2
            
            # Border
            for offset in range(3):
                draw.rectangle(
                    (x1 + offset, y1 + offset, x2 - offset, y2 - offset),
                    outline=(*stamp_color, alpha),
                    width=2
                )
            
            # Add horizontal lines (text simulation)
            for y in range(y1 + 15, y2 - 10, 12):
                line_alpha = random.randint(alpha // 3, alpha // 2)
                draw.line((x1 + 10, y, x2 - 10, y), fill=(*stamp_color, line_alpha), width=1)
        
        # Slight rotation for realism
        rotation = random.uniform(-15, 15)
        overlay = overlay.rotate(rotation, center=(cx, cy), resample=Image.BICUBIC)
        
        # Apply slight blur (ink bleeding)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _apply_handwriting_overlap(self, img: Image.Image) -> Image.Image:
        """
        Apply handwritten annotations that overlap printed text.
        
        Vietnamese cafe/restaurant staff often use pens to:
        - Circle table numbers
        - Cross out items
        - Write "Mang về" (Takeaway)  
        - Add notes over printed text
        """
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Ink colors (ballpoint pen)
        ink_colors = [
            (0, 0, 180),      # Blue (most common)
            (180, 0, 0),      # Red
            (0, 0, 0),        # Black
            (0, 100, 0),      # Green
        ]
        ink_color = random.choice(ink_colors)
        
        # Choose annotation type
        annotation_type = random.choice([
            "circle", "cross_out", "text_overlay", "arrow", "underline", "checkmark"
        ])
        
        try:
            # Try to load a handwriting-style font
            font_size = random.randint(14, 24)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        if annotation_type == "circle":
            # Circle a region (like table number)
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            rx = random.randint(20, 50)
            ry = random.randint(15, 40)
            
            # Draw imperfect circle (hand-drawn look)
            points = []
            for angle in range(0, 370, 10):
                rad = math.radians(angle)
                jitter_x = random.uniform(-3, 3)
                jitter_y = random.uniform(-3, 3)
                x = cx + (rx + jitter_x) * math.cos(rad)
                y = cy + (ry + jitter_y) * math.sin(rad)
                points.append((x, y))
            
            draw.line(points, fill=ink_color, width=2)
            
        elif annotation_type == "cross_out":
            # Cross out a line (cancelled item)
            y = random.randint(h // 4, 3 * h // 4)
            x1 = random.randint(10, w // 4)
            x2 = random.randint(w // 2, w - 10)
            
            # Wavy cross-out line
            points = []
            for x in range(x1, x2, 5):
                jitter = random.uniform(-2, 2)
                points.append((x, y + jitter))
            
            draw.line(points, fill=ink_color, width=2)
            
            # Sometimes double cross-out
            if random.random() < 0.3:
                points2 = [(p[0], p[1] + 5) for p in points]
                draw.line(points2, fill=ink_color, width=2)
                
        elif annotation_type == "text_overlay":
            # Write Vietnamese text over print
            texts = [
                "Mang về", "Takeaway", "Bàn ", "Table ", 
                "OK", "Đã TT", "Paid", "Chờ", "VIP",
                "Giảm", "Free", "Tặng", "x2", "DONE"
            ]
            text = random.choice(texts)
            if "Bàn" in text or "Table" in text:
                text += str(random.randint(1, 30))
            
            x = random.randint(10, w - 100)
            y = random.randint(10, h - 40)
            
            # Slight rotation for hand-written look
            text_img = Image.new("RGBA", (150, 50), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((5, 5), text, font=font, fill=ink_color)
            
            rotation = random.uniform(-10, 10)
            text_img = text_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
            
            # Paste onto main image
            img.paste(text_img, (x, y), text_img)
            
        elif annotation_type == "arrow":
            # Draw an arrow pointing to something
            x1 = random.randint(w // 2, w - 30)
            y1 = random.randint(30, h - 30)
            x2 = x1 - random.randint(30, 80)
            y2 = y1 + random.randint(-20, 20)
            
            # Arrow line
            draw.line((x1, y1, x2, y2), fill=ink_color, width=2)
            
            # Arrow head
            angle = math.atan2(y2 - y1, x2 - x1)
            arrow_size = 10
            draw.line((x2, y2, 
                       x2 - arrow_size * math.cos(angle - 0.5),
                       y2 - arrow_size * math.sin(angle - 0.5)), 
                      fill=ink_color, width=2)
            draw.line((x2, y2,
                       x2 - arrow_size * math.cos(angle + 0.5),
                       y2 - arrow_size * math.sin(angle + 0.5)),
                      fill=ink_color, width=2)
                      
        elif annotation_type == "underline":
            # Underline important text
            y = random.randint(h // 3, 2 * h // 3)
            x1 = random.randint(10, w // 3)
            x2 = random.randint(w // 2, w - 10)
            
            # Wavy underline
            points = []
            for x in range(x1, x2, 3):
                jitter = random.uniform(-1, 1)
                points.append((x, y + jitter))
            
            draw.line(points, fill=ink_color, width=2)
            
        else:  # checkmark
            # Draw a checkmark
            x = random.randint(w - 80, w - 20)
            y = random.randint(20, h - 40)
            
            # Checkmark shape
            draw.line((x, y + 15, x + 10, y + 25), fill=ink_color, width=3)
            draw.line((x + 10, y + 25, x + 30, y), fill=ink_color, width=3)
        
        return img

def create_defect_preset(preset: str) -> DefectConfig:
    """Create predefined defect configurations."""
    presets = {
        "pristine": DefectConfig(
            min_defects=0,
            max_defects=1,
            intensity="light",
            defect_probabilities={dt: 0.05 for dt in DefectType}
        ),
        "good_scan": DefectConfig(
            min_defects=1,
            max_defects=3,
            intensity="light",
            defect_probabilities={
                DefectType.SKEW: 0.3,
                DefectType.NOISE: 0.2,
                DefectType.SCANNER_SHADOW: 0.2,
                DefectType.COMPRESSION: 0.3,
            }
        ),
        "used_receipt": DefectConfig(
            min_defects=2,
            max_defects=5,
            intensity="medium",
            defect_probabilities={
                DefectType.FOLD: 0.4,
                DefectType.CREASE: 0.3,
                DefectType.EDGE_DAMAGE: 0.3,
                DefectType.FADE: 0.2,
                DefectType.DUST_SPECKS: 0.4,
                DefectType.SKEW: 0.5,
            }
        ),
        "damaged": DefectConfig(
            min_defects=4,
            max_defects=8,
            intensity="heavy",
            defect_probabilities={
                DefectType.FOLD: 0.5,
                DefectType.TEAR: 0.3,
                DefectType.COFFEE_STAIN: 0.3,
                DefectType.WATER_STAIN: 0.2,
                DefectType.FADE: 0.4,
                DefectType.BLUR_LOCAL: 0.3,
                DefectType.NOISE: 0.6,
            }
        ),
        "extreme": DefectConfig(
            min_defects=5,
            max_defects=10,
            intensity="extreme",
            defect_probabilities={dt: 0.4 for dt in DefectType}
        ),
    }

    return presets.get(preset, presets["used_receipt"])
