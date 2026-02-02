"""
Layout Templates for diverse invoice/receipt styles.

Provides:
    - Supermarket receipts (thermal paper style, long narrow)
    - Formal VAT invoices (government format)
    - Restaurant bills (casual, often handwritten elements)
    - Traditional market receipts (minimal, often handwritten)
    - Modern POS receipts (clean, digital look)
    - Cafe receipts (trendy, minimalist)
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
import math
import os
import glob
from pathlib import Path

from synthetic_data import vietnamese_vocab
VIETNAMESE_BRANDS = vietnamese_vocab.VIETNAMESE_BRANDS
STORE_PROFILES = vietnamese_vocab.STORE_PROFILES


class LayoutType(Enum):
    SUPERMARKET_THERMAL = "supermarket_thermal"      # Long narrow thermal receipt
    FORMAL_VAT = "formal_vat"                        # Official VAT invoice
    RESTAURANT_BILL = "restaurant_bill"              # Restaurant-style bill
    TRADITIONAL_MARKET = "traditional_market"        # Handwritten/simple
    MODERN_POS = "modern_pos"                        # Clean POS system
    CAFE_MINIMAL = "cafe_minimal"                    # Trendy cafe receipt
    HANDWRITTEN = "handwritten"                      # Fully handwritten
    DELIVERY_RECEIPT = "delivery_receipt"            # Food delivery apps
    HOTEL_BILL = "hotel_bill"                        # Hotel/hospitality bill
    UTILITY_BILL = "utility_bill"                    # Electricity, water, internet
    ECOMMERCE_RECEIPT = "ecommerce_receipt"          # Online shopping confirmation
    TAXI_RECEIPT = "taxi_receipt"                    # Taxi/ride-hailing receipts


@dataclass
class LayoutConfig:
    """Configuration for a layout."""
    layout_type: LayoutType
    width_range: Tuple[int, int]
    height_range: Tuple[int, int]
    margin: int = 20
    line_spacing: float = 1.2
    font_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)


class FontManager:
    """Manage fonts for rendering Vietnamese text."""

    # Font mapping
    # Assuming fonts are at: synthetic_data/fonts/{category}/{Family}/{...}.ttf
    _font_paths = {}
    _font_cache = {}
    _font_dir = Path("synthetic_data/fonts")

    @classmethod
    def _scan_fonts(cls):
        """Dynamically scan for fonts if not already loaded."""
        if cls._font_paths:
            return

        # Define categories and patterns
        patterns = {
            "sans": ["formal/*/*-Regular.ttf", "formal/*/*-Medium.ttf"],
            "sans_bold": ["formal/*/*-Bold.ttf"],
            "serif": ["formal/*/*-Regular.ttf", "formal/*/*-Medium.ttf"],
            "serif_bold": ["formal/*/*-Bold.ttf"],
            "mono": ["thermal_printer/*/*-Regular.ttf", "dot_matrix/*/*-Regular.ttf"],
            "handwritten": ["handwritten/*/*-Regular.ttf"],
            "thermal": ["thermal_printer/*/*-Regular.ttf", "dot_matrix/*/*-Regular.ttf"],
        }
        
        
        # Absolute path to font dir
        # We need to find the synthetic_data/fonts directory relative to this file
        base_dir = Path(__file__).parent / "fonts"
        
        for category, glob_patterns in patterns.items():
            cls._font_paths[category] = []
            for pattern in glob_patterns:
                # Use glob to find files
                found = list(base_dir.glob(pattern))
                
                # VALIDATE FONTS due to box rendering issues
                valid_fonts = []
                for p in found:
                    if cls._font_supports_vietnamese(str(p)):
                         valid_fonts.append(str(p))
                    else:
                        pass # significantly slows down startup if we print every skip
                
                cls._font_paths[category].extend(valid_fonts)
            
            # If no fonts found (e.g. structure difference), fallback or empty?
            if not cls._font_paths[category]:
                # Try fallback to scanning everything in category dir?
                cat_dir = base_dir / category.split("_")[0] # e.g. formal, handwritten
                if cat_dir.exists():
                     # Also validate fallbacks
                     candidates = list(cat_dir.glob("**/*.ttf"))
                     valid_fallbacks = [str(p) for p in candidates if cls._font_supports_vietnamese(str(p))]
                     cls._font_paths[category].extend(valid_fallbacks)

    @classmethod
    def _font_supports_vietnamese(cls, font_path: str) -> bool:
        """Check if font supports basic Vietnamese characters."""
        try:
            # Load font
            font = ImageFont.truetype(font_path, 20)
            
            # Critical Vietnamese characters to check
            # ế, ộ, ơ, ư, ắ, ậ, đ
            test_chars = ["ế", "ộ", "ơ", "ư", "ắ", "ậ", "đ"]
            
            # Check if font has glyphs. getmask() is a reliable way to check
            # if a char renders to something non-empty, but for 'box' characters
            # we rely on FreeType behavior or just visual check.
            # However, PIL doesn't easily expose "missing glyph" info.
            # A common heuristic is checking getmetrics or similar but complex.
            
            # SIMPLER CHECK:
            # Many fonts that don't support Vietnamese will default to a 'box' which HAS dimensions.
            # But usually it's a specific '.notdef' glyph.
            
            # Let's try rendering a known char vs unknown char?
            # Actually, most open source fonts we use MUST support VN.
            # If the user says they are seeing boxes, it means some fonts in the dir are BAD.
            # We will use a known set of good fonts or try to detect.
            
            # For now, let's just ensure we can load it.
            # IMPROVEMENT: Use the 'cmap' table to check support if valid.
            import fontTools.ttLib
            tt = fontTools.ttLib.TTFont(font_path)
            cmap = tt['cmap']
            tables = cmap.getBestCmap()
            
            if not tables:
                return False
                
            # Check unicode code points for test chars
            for char in test_chars:
                if ord(char) not in tables:
                    return False
                    
            return True
        except Exception:
            return False

    @classmethod
    def get_font(cls, family: str = "sans", size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font instance."""
        # Ensure fonts are scanned
        cls._scan_fonts()
        
        key = f"{family}_{size}_{bold}"
        
        if key in cls._font_cache:
            return cls._font_cache[key]
            
        # Select font path
        paths = cls._font_paths.get(family, [])
        if not paths:
             # Fallback if specific family not found
             if family.endswith("_bold"):
                 fallback = family.replace("_bold", "")
                 paths = cls._font_paths.get(fallback, [])

        if not paths:
            # Ultimate fallback
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size) # specific to system
            except:
                font = ImageFont.load_default()
            cls._font_cache[key] = font
            return font
            
        font_path = random.choice(paths)
        
        try:
            font = ImageFont.truetype(font_path, size)
            cls._font_cache[key] = font
            return font
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            font = ImageFont.load_default()
            cls._font_cache[key] = font
            return font



    @classmethod
    def get_random_font(cls, size: int = 14, style: str = "any") -> ImageFont.FreeTypeFont:
        """Get a random font with Vietnamese support."""
        if style == "any":
            family = random.choice(["sans", "serif", "mono"])
        else:
            family = style
        return cls.get_font(family, size)


class BaseLayout:
    """Base class for invoice layouts."""

    def __init__(self, config: LayoutConfig):
        self.config = config
        self.width = random.randint(*config.width_range)
        self.height = random.randint(*config.height_range)
        self.margin = config.margin
        self.y_cursor = self.margin
        self.img = None
        self.draw = None
        self.rendered_text = []  # Track text + polygons: [{"text": str, "polygon": [[x1,y1],...]}]
        self.currency_format_style = "standard"  # Default

    def _init_canvas(self, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        """Initialize the canvas."""
        self.img = Image.new("RGB", (self.width, self.height), bg_color)
        self.draw = ImageDraw.Draw(self.img)
        self.rendered_text = []  # Reset text tracking
        
        # Randomize currency format style per invoice
        # 50% chance to have 'đ'/'₫' symbol, 50% chance to have no symbol
        if random.random() < 0.5:
            self.currency_format_style = "none"
        else:
            self.currency_format_style = random.choice(["standard", "symbol", "symbol_clean"])
            
    def _format_currency(self, value: float, currency: str = "VND") -> str:
        """Format currency string based on current style."""
        if currency != "VND":
            return f"{value:.2f} {currency}"
            
        val_int = int(value)
        # Use simple comma separator
        val_str = f"{val_int:,}"
        
        if self.currency_format_style == "none":
            return val_str
        elif self.currency_format_style == "standard":
            return f"{val_str}đ"
        elif self.currency_format_style == "symbol":
            return f"{val_str}₫"
        elif self.currency_format_style == "symbol_clean":
            return f"{val_str} ₫"
        else:
            return f"{val_str}đ"

    def _text_size(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Get text dimensions."""
        if font is None:
            return (len(text) * 8, 16)
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            return (len(text) * 8, 16)

    def _draw_text(self, text: str, x: int, y: int, font: ImageFont.FreeTypeFont,
                   color: Tuple[int, int, int] = (0, 0, 0), max_width: Optional[int] = None):
        """Draw text with optional truncation and track polygon annotation."""
        text = text.replace('\n', ' ')
        if max_width:
            while self._text_size(text, font)[0] > max_width and len(text) > 3:
                text = text[:-4] + "..."
        self.draw.text((x, y), text, font=font, fill=color, anchor="lt")
        # Track the rendered text with polygon for OCR ground truth
        # Polygon format: 4 corner points clockwise from top-left
        if text.strip():
            w, h = self._text_size(text, font)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            self.rendered_text.append({
                "text": text.strip(),
                # Polygon: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] clockwise from top-left
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                # Keep bbox for backward compatibility
                "bbox": [x1, y1, x2, y2]
            })

    def _draw_line(self, y: int, style: str = "solid", color: Tuple[int, int, int] = (0, 0, 0)):
        """Draw a horizontal line."""
        x1, x2 = self.margin, self.width - self.margin

        if style == "solid":
            self.draw.line((x1, y, x2, y), fill=color, width=1)
        elif style == "dashed":
            dash_len = 10
            for x in range(x1, x2, dash_len * 2):
                self.draw.line((x, y, min(x + dash_len, x2), y), fill=color, width=1)
        elif style == "dotted":
            for x in range(x1, x2, 5):
                self.draw.point((x, y), fill=color)
        elif style == "double":
            self.draw.line((x1, y - 2, x2, y - 2), fill=color, width=1)
            self.draw.line((x1, y + 2, x2, y + 2), fill=color, width=1)

    def _advance_y(self, amount: int = None, font: ImageFont.FreeTypeFont = None):
        """Advance the y cursor."""
        if amount:
            self.y_cursor += amount
        elif font:
            self.y_cursor += int(self._text_size("A", font)[1] * self.config.line_spacing)
        else:
            self.y_cursor += 20

    def render(self, data: Dict) -> Image.Image:
        """Render the invoice. Override in subclasses."""
        raise NotImplementedError

    def get_ocr_annotations(self) -> List[Dict]:
        """Get all rendered text with polygons for OCR ground truth."""
        return self.rendered_text

    def get_ocr_text(self) -> List[str]:
        """Get just text strings (legacy compatibility)."""
        return [item["text"] for item in self.rendered_text]


class ThermalReceiptLayout(BaseLayout):
    """Supermarket thermal receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.SUPERMARKET_THERMAL,
            width_range=(280, 380),
            height_range=(600, 1400),
            margin=10,
            line_spacing=1.1,
            font_sizes={
                "header": (14, 20),
                "body": (10, 14),
                "footer": (8, 12),
            },
            features={
                "barcode": True,
                "qr_code": random.random() < 0.3,
                "logo_placeholder": random.random() < 0.4,
            }
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render thermal receipt."""
        # Slight yellow/gray tint for thermal paper
        bg = random.choice([
            (255, 255, 255),
            (252, 250, 245),
            (248, 248, 248),
            (255, 253, 245),
        ])
        self._init_canvas(bg)

        # Fonts
        header_size = random.randint(*self.config.font_sizes["header"])
        body_size = random.randint(*self.config.font_sizes["body"])
        footer_size = random.randint(*self.config.font_sizes["footer"])

        header_font = FontManager.get_font("mono", header_size)
        body_font = FontManager.get_font("mono", body_size)
        footer_font = FontManager.get_font("mono", footer_size)

        # Text color - thermal prints can be faded
        text_color = random.choice([
            (0, 0, 0),
            (30, 30, 30),
            (50, 50, 50),
            (20, 10, 10),  # Slightly reddish
        ])

        # === HEADER ===
        # Store name centered
        store_name = data.get("store_name", "CỬA HÀNG")
        tw, th = self._text_size(store_name, header_font)
        self._draw_text(store_name, (self.width - tw) // 2, self.y_cursor, header_font, text_color)
        self._advance_y(font=header_font)

        # Address
        address = data.get("address", "")
        if address:
            # Wrap long address
            words = address.split()
            lines = []
            current = ""
            for word in words:
                test = current + " " + word if current else word
                if self._text_size(test, footer_font)[0] < self.width - 2 * self.margin:
                    current = test
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)

            for line in lines[:2]:  # Max 2 lines
                tw, _ = self._text_size(line, footer_font)
                self._draw_text(line, (self.width - tw) // 2, self.y_cursor, footer_font, text_color)
                self._advance_y(font=footer_font)

        # Phone
        phone = data.get("phone", "")
        if phone:
            tw, _ = self._text_size(phone, footer_font)
            self._draw_text(phone, (self.width - tw) // 2, self.y_cursor, footer_font, text_color)
            self._advance_y(font=footer_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(10)

        # Invoice number and date
        inv_num = data.get("invoice_number", "")
        date = data.get("date", "")

        self._draw_text(f"HD: {inv_num}", self.margin, self.y_cursor, footer_font, text_color)
        date_w, _ = self._text_size(date, footer_font)
        self._draw_text(date, self.width - self.margin - date_w, self.y_cursor, footer_font, text_color)
        self._advance_y(font=footer_font)

        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(8)

        # === ITEMS ===
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")[:20]  # Truncate long names
            qty = item.get("qty", 1)
            unit_price = item.get("unit", 0)
            total = item.get("total", 0)

            # Item name
            self._draw_text(desc, self.margin, self.y_cursor, body_font, text_color)
            self._advance_y(font=body_font)

            # Qty x Price = Total (right-aligned)
            if currency == "VND":
                # Special detailed line for thermal
                u_str = self._format_currency(unit_price, currency)
                t_str = self._format_currency(total, currency)
                line = f"  {qty} x {u_str} = {t_str}"
            else:
                line = f"  {qty} x {unit_price:.2f} = {total:.2f}"

            tw, _ = self._text_size(line, footer_font)
            self._draw_text(line, self.width - self.margin - tw, self.y_cursor, footer_font, text_color)
            self._advance_y(font=footer_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(8)

        # === TOTALS ===
        subtotal = data.get("subtotal", 0)
        vat_rate = data.get("vat_rate", 0)
        vat = data.get("vat", 0)
        grand = data.get("grand_total", 0)

        def draw_total_line(label: str, value: float):
            self._draw_text(label, self.margin, self.y_cursor, body_font, text_color)
            if currency == "VND":
                val_str = self._format_currency(value, currency)
            else:
                val_str = f"{value:.2f}"
            tw, _ = self._text_size(val_str, body_font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, body_font, text_color)
            self._advance_y(font=body_font)

        draw_total_line("Cộng:", subtotal)
        if vat_rate > 0:
            draw_total_line(f"VAT ({vat_rate}%):", vat)

        self._draw_line(self.y_cursor, style="solid", color=text_color)
        self._advance_y(5)

        # Grand total - bigger
        self._draw_text("TỔNG CỘNG:", self.margin, self.y_cursor, header_font, text_color)
        if currency == "VND":
            grand_str = self._format_currency(grand, currency)
        else:
            grand_str = f"{grand:.2f} {currency}"
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, text_color)
        self._advance_y(font=header_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(10)

        # Payment method
        payment = data.get("payment_method")
        if payment:
            self._draw_text(f"Thanh toán: {payment}", self.margin, self.y_cursor, footer_font, text_color)
            self._advance_y(font=footer_font)

        # Footer messages
        footer_msgs = [
            "Cảm ơn quý khách!",
            "Hẹn gặp lại!",
            "Thank you!",
            "Xin cảm ơn!",
        ]

        self._advance_y(15)
        msg = random.choice(footer_msgs)
        tw, _ = self._text_size(msg, body_font)
        self._draw_text(msg, (self.width - tw) // 2, self.y_cursor, body_font, text_color)

        # Barcode placeholder
        if self.config.features.get("barcode") and random.random() < 0.6:
            self._advance_y(25)
            barcode = data.get("barcode", "0000000000000")
            self._draw_barcode(barcode, text_color)

        # Crop to actual content height
        return self._crop_to_content()

    def _draw_barcode(self, code: str, color: Tuple[int, int, int]):
        """Draw a simple barcode representation."""
        x = self.margin + 10
        y = self.y_cursor
        bar_height = 40

        for i, char in enumerate(code):
            digit = int(char)
            bar_width = 2 + (digit % 3)
            if i % 2 == 0:
                self.draw.rectangle((x, y, x + bar_width, y + bar_height), fill=color)
            x += bar_width + 1

        self.y_cursor += bar_height + 5

        # Code text below
        font = FontManager.get_font("mono", 10)
        tw, _ = self._text_size(code, font)
        self._draw_text(code, (self.width - tw) // 2, self.y_cursor, font, color)
        self._advance_y(15)

    def _crop_to_content(self) -> Image.Image:
        """Crop image to actual content."""
        return self.img.crop((0, 0, self.width, min(self.y_cursor + 20, self.height)))


class FormalVATLayout(BaseLayout):
    """Official Vietnamese VAT invoice format."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.FORMAL_VAT,
            width_range=(700, 900),
            height_range=(900, 1200),
            margin=40,
            line_spacing=1.3,
            font_sizes={
                "title": (24, 32),
                "header": (14, 18),
                "body": (11, 14),
                "footer": (9, 11),
            },
            features={
                "red_header": True,
                "serial_number": True,
                "government_format": True,
            }
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render formal VAT invoice."""
        self._init_canvas((255, 255, 255))

        # Fonts
        title_font = FontManager.get_font("serif", random.randint(*self.config.font_sizes["title"]), bold=True)
        header_font = FontManager.get_font("serif", random.randint(*self.config.font_sizes["header"]))
        body_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["footer"]))

        red = (180, 0, 0)
        black = (0, 0, 0)
        gray = (100, 100, 100)

        # === HEADER ===
        # Title
        title = "HÓA ĐƠN GIÁ TRỊ GIA TĂNG"
        tw, _ = self._text_size(title, title_font)
        self._draw_text(title, (self.width - tw) // 2, self.y_cursor, title_font, red)
        self._advance_y(font=title_font)

        subtitle = "(VAT INVOICE)"
        tw, _ = self._text_size(subtitle, body_font)
        self._draw_text(subtitle, (self.width - tw) // 2, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)
        self._advance_y(15)

        # Serial and number
        serial = f"Ký hiệu: {random.choice(['AA', 'AB', 'BA', 'BB'])}/{random.randint(20, 25)}E"
        inv_num = data.get("invoice_number", f"HD{random.randint(100000, 999999)}")

        self._draw_text(serial, self.margin, self.y_cursor, body_font, black)
        tw, _ = self._text_size(f"Số: {inv_num}", body_font)
        self._draw_text(f"Số: {inv_num}", self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Date
        date = data.get("date", "")
        tw, _ = self._text_size(f"Ngày: {date}", body_font)
        self._draw_text(f"Ngày: {date}", (self.width - tw) // 2, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._advance_y(20)

        # Seller info
        self._draw_text("Đơn vị bán hàng:", self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        seller_info = [
            f"Tên: {data.get('store_name', '')}",
            f"Địa chỉ: {data.get('address', '')}",
            f"MST: {data.get('tax_code', '')}",
            f"Điện thoại: {data.get('phone', '')}",
        ]

        for info in seller_info:
            self._draw_text(info, self.margin + 20, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        self._advance_y(15)

        # Buyer info (random or empty)
        self._draw_text("Đơn vị mua hàng:", self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        if random.random() < 0.6:
            from faker import Faker
            fake = Faker("vi_VN")
            buyer_info = [
                f"Tên: {fake.company()}",
                f"Địa chỉ: {fake.address()[:50]}",
                f"MST: {random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            ]
            for info in buyer_info:
                self._draw_text(info, self.margin + 20, self.y_cursor, body_font, black)
                self._advance_y(font=body_font)

        self._advance_y(20)

        # === TABLE ===
        self._draw_table(data, body_font, small_font, black)

        # === TOTALS ===
        self._advance_y(20)
        currency = data.get("currency", "VND")
        subtotal = data.get("subtotal", 0)
        vat = data.get("vat", 0)
        grand = data.get("grand_total", 0)
        vat_rate = data.get("vat_rate", 10)

        # Amount in words
        self._draw_text("Số tiền viết bằng chữ:", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        if currency == "VND":
            words = self._number_to_vietnamese(int(grand))
            self._draw_text(f"    {words}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(30)

        # Signatures
        sig_y = self.y_cursor
        col_width = (self.width - 2 * self.margin) // 3

        signatures = ["Người mua hàng", "Người bán hàng", "Thủ trưởng đơn vị"]
        for i, sig in enumerate(signatures):
            x = self.margin + i * col_width
            tw, _ = self._text_size(sig, body_font)
            self._draw_text(sig, x + (col_width - tw) // 2, sig_y, body_font, black)

            sub = "(Ký, ghi rõ họ tên)"
            tw, _ = self._text_size(sub, small_font)
            self._draw_text(sub, x + (col_width - tw) // 2, sig_y + 20, small_font, gray)

        return self.img

    def _draw_table(self, data: Dict, body_font, small_font, color):
        """Draw the items table."""
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        # Column widths
        cols = [40, 200, 60, 80, 80, 100]  # STT, Name, Unit, Qty, Price, Total
        headers = ["STT", "Tên hàng hóa, dịch vụ", "ĐVT", "SL", "Đơn giá", "Thành tiền"]

        x = self.margin

        # Header row
        self._draw_line(self.y_cursor, style="solid", color=color)
        self.y_cursor += 3

        for i, (header, width) in enumerate(zip(headers, cols)):
            self._draw_text(header, x + 5, self.y_cursor, small_font, color, max_width=width - 10)
            x += width

        self._advance_y(font=body_font)
        self._draw_line(self.y_cursor, style="solid", color=color)
        self.y_cursor += 3

        # Items
        for idx, item in enumerate(items, 1):
            x = self.margin
            row_data = [
                str(idx),
                item.get("desc", "")[:25],
                random.choice(["cái", "hộp", "kg", "chai", "gói"]),
                str(item.get("qty", 1)),
                self._format_currency(item.get('unit', 0), currency) if currency == "VND" else f"{item.get('unit', 0):.2f}",
                self._format_currency(item.get('total', 0), currency) if currency == "VND" else f"{item.get('total', 0):.2f}",
            ]

            for value, width in zip(row_data, cols):
                self._draw_text(value, x + 5, self.y_cursor, small_font, color, max_width=width - 10)
                x += width

            self._advance_y(font=small_font)

        self._draw_line(self.y_cursor, style="solid", color=color)
        self.y_cursor += 5

        # Totals in table
        subtotal = data.get("subtotal", 0)
        vat = data.get("vat", 0)
        vat_rate = data.get("vat_rate", 10)
        grand = data.get("grand_total", 0)

        total_x = self.margin + sum(cols[:4])

        def draw_total_row(label, value):
            self._draw_text(label, total_x, self.y_cursor, body_font, color)
            if currency == "VND":
                val_str = self._format_currency(value, currency)
            else:
                val_str = f"{value:.2f}"
            tw, _ = self._text_size(val_str, body_font)
            self._draw_text(val_str, self.margin + sum(cols) - tw - 5, self.y_cursor, body_font, color)
            self._advance_y(font=body_font)

        draw_total_row("Cộng tiền hàng:", subtotal)
        draw_total_row(f"Thuế GTGT ({vat_rate}%):", vat)
        draw_total_row("Tổng cộng tiền thanh toán:", grand)

    def _number_to_vietnamese(self, n: int) -> str:
        """Convert number to Vietnamese words (simplified)."""
        if n == 0:
            return "Không đồng"

        units = ["", "nghìn", "triệu", "tỷ"]
        digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

        # Simplified conversion
        if n < 1000:
            return f"{digits[n // 100]} trăm {digits[(n // 10) % 10]} mươi {digits[n % 10]} đồng"

        parts = []
        unit_idx = 0
        while n > 0:
            chunk = n % 1000
            if chunk > 0:
                parts.insert(0, f"{chunk} {units[unit_idx]}")
            n //= 1000
            unit_idx += 1

        return " ".join(parts) + " đồng"


class HandwrittenLayout(BaseLayout):
    """Simulated handwritten receipt/note."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.HANDWRITTEN,
            width_range=(400, 600),
            height_range=(500, 800),
            margin=30,
            line_spacing=1.8,
            font_sizes={
                "body": (16, 24),
            },
            features={
                "lined_paper": random.random() < 0.5,
                "crossed_out": random.random() < 0.2,
            }
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render handwritten receipt."""
        # Paper color
        bg = random.choice([
            (255, 255, 240),  # Cream
            (255, 250, 230),  # Light yellow
            (245, 245, 245),  # Light gray
            (255, 255, 255),  # White
        ])
        self._init_canvas(bg)

        # Draw lines if lined paper
        if self.config.features.get("lined_paper"):
            line_color = (200, 200, 220)
            for y in range(50, self.height, 30):
                self.draw.line((20, y, self.width - 20, y), fill=line_color, width=1)

        font = FontManager.get_font("handwritten", random.randint(*self.config.font_sizes["body"]))
        ink_color = random.choice([
            (0, 0, 100),      # Blue ink
            (0, 0, 0),        # Black
            (50, 50, 50),     # Dark gray
        ])

        # Randomize text positions slightly for handwritten effect
        def jitter():
            return random.randint(-3, 3)

        # Header
        store = data.get("store_name", "Cửa hàng")
        self._draw_text(store, self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)
        self._advance_y(font=font)

        # Date
        date = data.get("date", "")
        self._draw_text(f"Ngày: {date}", self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)
        self._advance_y(font=font)

        self._advance_y(10)

        # Items with casual formatting
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")[:15]
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            if currency == "VND":
                t_str = self._format_currency(total, currency)
                line = f"{desc} x{qty} = {t_str}"
            else:
                line = f"{desc} x{qty} = {total:.2f}"

            self._draw_text(line, self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)
            self._advance_y(font=font)

        # Underline before total
        self._advance_y(5)
        self.draw.line(
            (self.margin, self.y_cursor, self.width - self.margin, self.y_cursor),
            fill=ink_color, width=2
        )
        self._advance_y(10)

        # Grand total
        grand = data.get("grand_total", 0)
        if currency == "VND":
            total_text = f"Tổng: {self._format_currency(grand, currency)}"
        else:
            total_text = f"Tổng: {grand:.2f} {currency}"

        self._draw_text(total_text, self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)

        return self.img


class CafeMinimalLayout(BaseLayout):
    """Modern minimalist cafe receipt."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.CAFE_MINIMAL,
            width_range=(300, 400),
            height_range=(400, 700),
            margin=25,
            line_spacing=1.4,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render minimal cafe receipt."""
        self._init_canvas((255, 255, 255))

        # Clean modern fonts
        title_font = FontManager.get_font("sans", 22)
        body_font = FontManager.get_font("sans", 13)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (120, 120, 120)

        # Centered store name
        store = data.get("store_name", "COFFEE SHOP")
        tw, _ = self._text_size(store.upper(), title_font)
        self._draw_text(store.upper(), (self.width - tw) // 2, self.y_cursor, title_font, black)
        self._advance_y(font=title_font)
        self._advance_y(20)

        # Simple line
        self.draw.line(
            (self.width // 4, self.y_cursor, 3 * self.width // 4, self.y_cursor),
            fill=gray, width=1
        )
        self._advance_y(20)

        # Items
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            # Item name left
            self._draw_text(f"{qty}x {desc}", self.margin, self.y_cursor, body_font, black)

            # Price right
            if currency == "VND":
                price_str = self._format_currency(total, currency)
            else:
                price_str = f"{total:.2f}"
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)

            self._advance_y(font=body_font)

        self._advance_y(15)
        self.draw.line(
            (self.margin, self.y_cursor, self.width - self.margin, self.y_cursor),
            fill=gray, width=1
        )
        self._advance_y(15)

        # Total
        grand = data.get("grand_total", 0)
        self._draw_text("TOTAL", self.margin, self.y_cursor, title_font, black)
        self._draw_text("TOTAL", self.margin, self.y_cursor, title_font, black)
        if currency == "VND":
            total_str = self._format_currency(grand, currency)
        else:
            total_str = f"{grand:.2f}"
        tw, _ = self._text_size(total_str, title_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, title_font, black)

        self._advance_y(font=title_font)
        self._advance_y(30)

        # Thank you centered
        thanks = "Thank you • Cảm ơn"
        tw, _ = self._text_size(thanks, small_font)
        self._draw_text(thanks, (self.width - tw) // 2, self.y_cursor, small_font, gray)

        return self.img


class RestaurantBillLayout(BaseLayout):
    """Restaurant dine-in bill layout with table number and service charge."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.RESTAURANT_BILL,
            width_range=(350, 450),
            height_range=(500, 900),
            margin=20,
            line_spacing=1.3,
            font_sizes={
                "header": (18, 24),
                "body": (12, 16),
                "footer": (10, 12),
            },
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render restaurant bill."""
        # Paper color variations
        bg = random.choice([
            (255, 255, 255),
            (255, 252, 245),  # Cream
            (248, 248, 245),  # Light gray
        ])
        self._init_canvas(bg)

        # Fonts
        header_font = FontManager.get_font("serif", random.randint(*self.config.font_sizes["header"]), bold=True)
        body_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["footer"]))

        black = (0, 0, 0)
        gray = (100, 100, 100)
        red = (180, 50, 50)

        # === HEADER ===
        store_name = random.choice(VIETNAMESE_BRANDS.get("cafes_restaurants", ["NHÀ HÀNG"]))
        if "NHÀ HÀNG" in store_name:
             pass 
        elif random.random() < 0.3:
             prefix = random.choice(["Nhà hàng", "Quán", "Bếp", "Tiệm ăn"])
             store_name = f"{prefix} {store_name}"
        tw, _ = self._text_size(store_name.upper(), header_font)
        self._draw_text(store_name.upper(), (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Address
        address = data.get("address", "")
        if address:
            tw, _ = self._text_size(address[:40], small_font)
            self._draw_text(address[:40], (self.width - tw) // 2, self.y_cursor, small_font, gray)
            self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(15)

        # Table number and waiter (restaurant-specific)
        table_num = random.randint(1, 30)
        waiter_names = ["Minh", "Lan", "Hùng", "Mai", "Tuấn", "Hoa", "Nam", "Linh"]
        waiter = random.choice(waiter_names)

        self._draw_text(f"Bàn: {table_num}", self.margin, self.y_cursor, body_font, red)
        tw, _ = self._text_size(f"NV: {waiter}", body_font)
        self._draw_text(f"NV: {waiter}", self.width - self.margin - tw, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)

        # Date and time
        date = data.get("date", "")
        self._draw_text(f"Ngày: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(10)

        # === ITEMS ===
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")[:25]
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            # Item line
            self._draw_text(f"{qty}x {desc}", self.margin, self.y_cursor, body_font, black)
            if currency == "VND":
                price_str = self._format_currency(total, currency)
            else:
                price_str = f"{total:.2f}"
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # === TOTALS ===
        subtotal = data.get("subtotal", 0)
        vat = data.get("vat", 0)
        grand = data.get("grand_total", 0)

        # Service charge (restaurant-specific)
        service_rate = random.choice([0, 5, 10])
        service_charge = int(subtotal * service_rate / 100)

        def draw_total_line(label, value, font=body_font, color=black):
            self._draw_text(label, self.margin, self.y_cursor, font, color)
            if currency == "VND":
                val_str = self._format_currency(value, currency)
            else:
                val_str = f"{value:.2f}"
            tw, _ = self._text_size(val_str, font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, font, color)
            self._advance_y(font=font)

        draw_total_line("Tạm tính:", subtotal, small_font, gray)
        if service_rate > 0:
            draw_total_line(f"Phí dịch vụ ({service_rate}%):", service_charge, small_font, gray)
        if vat > 0:
            draw_total_line("VAT:", vat, small_font, gray)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(10)

        # Grand total
        total_with_service = grand + service_charge
        self._draw_text("TỔNG CỘNG:", self.margin, self.y_cursor, header_font, black)
        self._draw_text("TỔNG CỘNG:", self.margin, self.y_cursor, header_font, black)
        if currency == "VND":
            grand_str = self._format_currency(total_with_service, currency)
        else:
            grand_str = f"{total_with_service:.2f}"
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Footer
        self._advance_y(20)
        footer_msgs = ["Chúc quý khách ngon miệng!", "Cảm ơn quý khách!", "Hẹn gặp lại!"]
        msg = random.choice(footer_msgs)
        tw, _ = self._text_size(msg, small_font)
        self._draw_text(msg, (self.width - tw) // 2, self.y_cursor, small_font, gray)

        return self.img


class ModernPOSLayout(BaseLayout):
    """Modern digital POS receipt layout (Square/Shopify style)."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.MODERN_POS,
            width_range=(320, 400),
            height_range=(450, 750),
            margin=25,
            line_spacing=1.4,
            font_sizes={
                "header": (16, 22),
                "body": (11, 14),
                "footer": (9, 11),
            },
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render modern POS receipt."""
        self._init_canvas((255, 255, 255))

        # Clean modern fonts
        header_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["header"]), bold=True)
        body_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["footer"]))

        black = (40, 40, 40)
        gray = (140, 140, 140)
        accent = random.choice([(0, 122, 255), (52, 199, 89), (255, 149, 0)])  # Blue/Green/Orange

        # === HEADER - Centered, clean ===
        store_name = data.get("store_name", "STORE")
        tw, _ = self._text_size(store_name.upper(), header_font)
        self._draw_text(store_name.upper(), (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Tagline
        taglines = ["Cảm ơn bạn đã mua hàng", "Thank you for your purchase", ""]
        tagline = random.choice(taglines)
        if tagline:
            tw, _ = self._text_size(tagline, small_font)
            self._draw_text(tagline, (self.width - tw) // 2, self.y_cursor, small_font, gray)
            self._advance_y(font=small_font)

        self._advance_y(15)

        # Order info in a clean box style
        inv_num = data.get("invoice_number", f"#{random.randint(1000, 9999)}")
        date = data.get("date", "")

        self._draw_text(f"Order {inv_num}", self.margin, self.y_cursor, body_font, black)
        tw, _ = self._text_size(date, small_font)
        self._draw_text(date, self.width - self.margin - tw, self.y_cursor, small_font, gray)
        self._advance_y(font=body_font)

        self._advance_y(10)
        # Thin line separator
        self.draw.line((self.margin, self.y_cursor, self.width - self.margin, self.y_cursor), fill=gray, width=1)
        self._advance_y(15)

        # === ITEMS - Clean list style ===
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            # Item description
            item_text = f"{desc}"
            if qty > 1:
                item_text = f"{qty} × {desc}"
            self._draw_text(item_text, self.margin, self.y_cursor, body_font, black)

            # Price right-aligned
            if currency == "VND":
                price_str = self._format_currency(total, currency)
            else:
                price_str = f"${total:.2f}"
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._advance_y(3)

        self._advance_y(10)
        self.draw.line((self.margin, self.y_cursor, self.width - self.margin, self.y_cursor), fill=gray, width=1)
        self._advance_y(15)

        # === TOTALS ===
        subtotal = data.get("subtotal", 0)
        vat = data.get("vat", 0)
        grand = data.get("grand_total", 0)

        def draw_summary_line(label, value, bold=False):
            font = header_font if bold else body_font
            color = black if bold else gray
            self._draw_text(label, self.margin, self.y_cursor, font, color)
            if currency == "VND":
                val_str = self._format_currency(value, currency)
                # Ensure spacing isn't affected, though logic below calculates width
            else:
                val_str = f"${value:.2f}"
            tw, _ = self._text_size(val_str, font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, font, black)
            self._advance_y(font=font)

        draw_summary_line("Subtotal", subtotal)
        if vat > 0:
            draw_summary_line("Tax", vat)

        self._advance_y(5)
        draw_summary_line("Total", grand, bold=True)

        # Payment method
        self._advance_y(15)
        payment = data.get("payment_method", random.choice(["Card", "Cash", "MoMo", "ZaloPay"]))
        self._draw_text(f"Paid with {payment}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        # QR Code area (placeholder)
        if random.random() < 0.4:
            self._advance_y(20)
            qr_size = 60
            qr_x = (self.width - qr_size) // 2
            self.draw.rectangle(
                (qr_x, self.y_cursor, qr_x + qr_size, self.y_cursor + qr_size),
                outline=gray, width=1
            )
            # QR pattern placeholder
            for _ in range(20):
                x = qr_x + random.randint(5, qr_size - 5)
                y = self.y_cursor + random.randint(5, qr_size - 5)
                self.draw.rectangle((x, y, x + 3, y + 3), fill=black)
            self._advance_y(qr_size + 10)

        return self.img


class DeliveryReceiptLayout(BaseLayout):
    """Food delivery app receipt layout (GrabFood, ShopeeFood style)."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.DELIVERY_RECEIPT,
            width_range=(350, 420),
            height_range=(600, 1000),
            margin=20,
            line_spacing=1.3,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render delivery receipt."""
        self._init_canvas((255, 255, 255))

        header_font = FontManager.get_font("sans", 20, bold=True)
        body_font = FontManager.get_font("sans", 13)
        small_font = FontManager.get_font("sans", 11)

        black = (30, 30, 30)
        gray = (120, 120, 120)
        # Delivery app brand colors
        brand_color = random.choice([
            (0, 171, 102),   # Grab green
            (238, 77, 45),   # ShopeeFood orange
            (255, 0, 80),    # GoFood red
        ])

        # === APP HEADER ===
        app_names = ["GrabFood", "ShopeeFood", "GoFood", "Baemin", "Loship"]
        # Use existing list or vocab if available, but delivery apps are specific
        # We can keep this list as it's specific to the layout type logic (colors)
        app_name = random.choice(app_names)
        tw, _ = self._text_size(app_name, header_font)
        self._draw_text(app_name, (self.width - tw) // 2, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        self._advance_y(5)
        order_label = "Đơn hàng của bạn"
        tw, _ = self._text_size(order_label, small_font)
        self._draw_text(order_label, (self.width - tw) // 2, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._advance_y(15)

        # Order ID
        order_id = f"#{random.randint(100000, 999999)}"
        self._draw_text(f"Mã đơn: {order_id}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Date/time
        date = data.get("date", "")
        self._draw_text(f"Thời gian: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # === RESTAURANT INFO ===
        store_name = data.get("store_name", "Nhà hàng")
        self._draw_text("Đặt từ:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(store_name, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        address = data.get("address", "")
        if address:
            self._draw_text(address[:45], self.margin, self.y_cursor, small_font, gray)
            self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # === ITEMS ===
        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "")[:30]
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            self._draw_text(f"{qty}x {desc}", self.margin, self.y_cursor, body_font, black)
            if currency == "VND":
                price_str = f"{int(total):,}đ"
            else:
                price_str = f"${total:.2f}"
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._advance_y(3)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # === DELIVERY FEES (delivery-specific) ===
        subtotal = data.get("subtotal", 0)
        grand = data.get("grand_total", 0)

        delivery_fee = random.choice([15000, 20000, 25000, 30000, 0])
        platform_fee = random.choice([0, 2000, 3000])
        discount = random.choice([0, 0, 0, -10000, -20000, -30000])

        def draw_fee_line(label, value, color=black):
            self._draw_text(label, self.margin, self.y_cursor, small_font, gray)
            if currency == "VND":
                 val_str = self._format_currency(value, currency)
            else:
                val_str = f"${value:.2f}"
            tw, _ = self._text_size(val_str, small_font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, small_font, color)
            self._advance_y(font=small_font)

        draw_fee_line("Tạm tính", subtotal)
        if delivery_fee > 0:
            draw_fee_line("Phí giao hàng", delivery_fee)
        if platform_fee > 0:
            draw_fee_line("Phí nền tảng", platform_fee)
        if discount < 0:
            draw_fee_line("Giảm giá", discount, brand_color)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Total
        total = subtotal + delivery_fee + platform_fee + discount
        self._draw_text("Tổng cộng", self.margin, self.y_cursor, header_font, black)
        if currency == "VND":
            total_str = self._format_currency(total, currency)
        else:
            total_str = f"${total:.2f}"
        tw, _ = self._text_size(total_str, header_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        # Payment
        self._advance_y(15)
        payments = ["Thanh toán khi nhận hàng", "Ví MoMo", "Ví ZaloPay", "Thẻ tín dụng", "GrabPay"]
        payment = random.choice(payments)
        self._draw_text(f"Thanh toán: {payment}", self.margin, self.y_cursor, small_font, gray)

        return self.img


class HotelBillLayout(BaseLayout):
    """Hotel/hospitality bill with multi-day charges."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.HOTEL_BILL,
            width_range=(600, 750),
            height_range=(800, 1100),
            margin=35,
            line_spacing=1.4,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render hotel bill."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("serif", 24, bold=True)
        header_font = FontManager.get_font("serif", 16)
        body_font = FontManager.get_font("sans", 12)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (100, 100, 100)
        gold = (180, 140, 50)

        # === HOTEL HEADER ===
        hotel_names = VIETNAMESE_BRANDS.get("hotels", ["Khách sạn Hoàng Gia"])
        hotel_name = random.choice(hotel_names)
        tw, _ = self._text_size(hotel_name, title_font)
        self._draw_text(hotel_name, (self.width - tw) // 2, self.y_cursor, title_font, gold)
        self._advance_y(font=title_font)

        # Stars
        stars = "★" * random.randint(3, 5)
        tw, _ = self._text_size(stars, header_font)
        self._draw_text(stars, (self.width - tw) // 2, self.y_cursor, header_font, gold)
        self._advance_y(font=header_font)

        address = data.get("address", "123 Đông Khởi, Quận 1, TP.HCM")
        tw, _ = self._text_size(address[:50], small_font)
        self._draw_text(address[:50], (self.width - tw) // 2, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(15)
        self._draw_line(self.y_cursor, style="double", color=gold)
        self._advance_y(20)

        # === BILL DETAILS ===
        self._draw_text("HÓA ĐƠN THANH TOÁN", self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(10)

        # Guest info
        guest_names = ["Nguyễn Văn A", "Trần Thị B", "Lê Minh C", "Phạm Hoàng D"]
        guest = random.choice(guest_names)
        room_types = ["Deluxe", "Superior", "Suite", "Presidential"]
        room_type = random.choice(room_types)
        room_num = random.randint(101, 899)
        nights = random.randint(1, 5)

        self._draw_text(f"Khách hàng: {guest}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(f"Phòng: {room_num} ({room_type})", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(f"Số đêm: {nights}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        date = data.get("date", "")
        self._draw_text(f"Ngày thanh toán: {date}", self.margin, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)

        self._advance_y(15)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(15)

        # === CHARGES TABLE ===
        currency = data.get("currency", "VND")

        # Hotel-specific charges
        room_rate = random.choice([1500000, 2000000, 2500000, 3500000, 5000000])
        room_total = room_rate * nights

        charges = [
            (f"Tiền phòng ({nights} đêm x {room_rate:,}đ)", room_total),
            ("Minibar", random.choice([0, 0, 150000, 250000])),
            ("Dịch vụ giặt ủi", random.choice([0, 0, 80000, 150000])),
            ("Điện thoại", random.choice([0, 0, 50000])),
            ("Bữa sáng", random.choice([0, 0, 200000 * nights])),
            ("Spa & Wellness", random.choice([0, 0, 0, 500000])),
        ]

        # Filter out zero charges
        charges = [(desc, amt) for desc, amt in charges if amt > 0]

        for desc, amount in charges:
            self._draw_text(desc, self.margin, self.y_cursor, body_font, black)
            if currency == "VND":
                amt_str = self._format_currency(amount, currency)
            else:
                amt_str = f"{amount:.2f}"
            tw, _ = self._text_size(amt_str, body_font)
            self._draw_text(amt_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Totals
        subtotal = sum(amt for _, amt in charges)
        service_charge = int(subtotal * 0.05)  # 5% service
        vat = int((subtotal + service_charge) * 0.1)  # 10% VAT
        grand_total = subtotal + service_charge + vat

        self._draw_text("Tạm tính:", self.margin, self.y_cursor, body_font, gray)
        # Assuming formatting consistency is desired, we use helper but note original might have lacked symbol
        # But user wants mixed, so helper is good.
        if currency == "VND":
             s_str = self._format_currency(subtotal, currency)
        else:
             s_str = f"{int(subtotal):,}" # Fallback
             
        tw, _ = self._text_size(s_str, body_font)
        self._draw_text(s_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text("Phí dịch vụ (5%):", self.margin, self.y_cursor, body_font, gray)
        if currency == "VND": v_str = self._format_currency(service_charge, currency) 
        else: v_str = f"{int(service_charge):,}"
        tw, _ = self._text_size(v_str, body_font)
        self._draw_text(v_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text("VAT (10%):", self.margin, self.y_cursor, body_font, gray)
        if currency == "VND": va_str = self._format_currency(vat, currency)
        else: va_str = f"{int(vat):,}"
        tw, _ = self._text_size(va_str, body_font)
        self._draw_text(va_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="double", color=gold)
        self._advance_y(10)

        # Grand total
        self._draw_text("TỔNG CỘNG:", self.margin, self.y_cursor, header_font, black)
        if currency == "VND":
            grand_str = self._format_currency(grand_total, currency)
        else:
            grand_str = f"${grand_total:.2f}"
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, gold)
        self._advance_y(font=header_font)

        # Signature area
        self._advance_y(40)
        col_width = (self.width - 2 * self.margin) // 2

        self._draw_text("Khách hàng", self.margin, self.y_cursor, small_font, gray)
        self._draw_text("Thu ngân", self.margin + col_width, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._draw_text("(Ký tên)", self.margin, self.y_cursor, small_font, gray)
        self._draw_text("(Ký tên, đóng dấu)", self.margin + col_width, self.y_cursor, small_font, gray)

        return self.img


class UtilityBillLayout(BaseLayout):
    """Utility bill layout (electricity, water, internet)."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.UTILITY_BILL,
            width_range=(600, 750),
            height_range=(700, 1000),
            margin=30,
            line_spacing=1.4,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render utility bill."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 22, bold=True)
        header_font = FontManager.get_font("sans", 14, bold=True)
        body_font = FontManager.get_font("sans", 12)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (100, 100, 100)
        blue = (0, 80, 160)

        # Utility type
        # Use dynamic providers if available
        providers = VIETNAMESE_BRANDS.get("utility_providers", [])
        if providers:
             provider = random.choice(providers)
             # Determine type based on name keywords
             if "Điện" in provider or "EVN" in provider:
                 short_name = "EVN"
                 bill_title = "HÓA ĐƠN TIỀN ĐIỆN"
                 color = (255, 100, 0) # Electricity orange/red often
             elif "Nước" in provider or "SAWACO" in provider:
                 short_name = "WTR"
                 bill_title = "HÓA ĐƠN TIỀN NƯỚC"
                 color = (0, 100, 200)
             elif "Telecom" in provider or "VNPT" in provider or "Viettel" in provider or "FPT" in provider:
                  short_name = "TEL"
                  bill_title = "HÓA ĐƠN VIỄN THÔNG"
                  color = (0, 80, 160)
             else:
                  short_name = "UTL"
                  bill_title = "HÓA ĐƠN DỊCH VỤ"
                  color = blue
             company = provider
        else:
            utility_types = [
                ("Điện lực Việt Nam", "EVN", "HÓA ĐƠN TIỀN ĐIỆN"),
                ("Cấp nước Sài Gòn", "SAWACO", "HÓA ĐƠN TIỀN NƯỚC"),
                ("VNPT", "VNPT", "HÓA ĐƠN VIỄN THÔNG"),
                ("Viettel", "Viettel", "HÓA ĐƠN DỊCH VỤ"),
                ("FPT Telecom", "FPT", "HÓA ĐƠN INTERNET"),
            ]
            company, short_name, bill_title = random.choice(utility_types)
            color = blue

        # Header
        tw, _ = self._text_size(company, title_font)
        self._draw_text(company, (self.width - tw) // 2, self.y_cursor, title_font, blue)
        self._advance_y(font=title_font)

        tw, _ = self._text_size(bill_title, header_font)
        self._draw_text(bill_title, (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(15)

        self._draw_line(self.y_cursor, style="solid", color=blue)
        self._advance_y(15)

        # Customer info
        customer_names = ["Nguyễn Văn A", "Trần Thị B", "Lê Minh C", "Phạm D"]
        customer = random.choice(customer_names)
        customer_id = f"{short_name}{random.randint(10000000, 99999999)}"

        self._draw_text(f"Khách hàng: {customer}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(f"Mã KH: {customer_id}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        address = data.get("address", "123 Nguyễn Huệ, Quận 1, TP.HCM")
        self._draw_text(f"Địa chỉ: {address[:50]}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._advance_y(15)

        # Billing period
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        month = random.choice(months)
        year = "2026"
        self._draw_text(f"Kỳ thanh toán: Tháng {month}/{year}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._advance_y(10)

        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Usage details
        self._draw_text("CHI TIẾT SỬ DỤNG", self.margin, self.y_cursor, header_font, blue)
        self._advance_y(font=header_font)
        self._advance_y(10)

        # Generate realistic usage
        if "ĐIỆN" in bill_title:
            old_reading = random.randint(10000, 50000)
            new_reading = old_reading + random.randint(100, 500)
            usage = new_reading - old_reading
            unit = "kWh"
            rate = random.choice([2000, 2500, 3000])
        elif "NƯỚC" in bill_title:
            old_reading = random.randint(1000, 5000)
            new_reading = old_reading + random.randint(10, 50)
            usage = new_reading - old_reading
            unit = "m³"
            rate = random.choice([8000, 10000, 12000])
        else:
            old_reading = 0
            new_reading = 0
            usage = 1
            unit = "tháng"
            rate = random.choice([150000, 200000, 300000, 500000])

        if old_reading > 0:
            self._draw_text(f"Chỉ số cũ: {old_reading:,}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._draw_text(f"Chỉ số mới: {new_reading:,}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._draw_text(f"Tiêu thụ: {usage:,} {unit}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        amount = usage * rate
        vat = int(amount * 0.1)
        total = amount + vat

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Totals
        # Totals
        self._draw_text(f"Tiền sử dụng:", self.margin, self.y_cursor, body_font, black)
        amt_str = self._format_currency(amount, "VND") # Utility is usually VND
        tw, _ = self._text_size(amt_str, body_font)
        self._draw_text(amt_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text(f"Thuế VAT (10%):", self.margin, self.y_cursor, body_font, gray)
        vat_str = self._format_currency(vat, "VND")
        tw, _ = self._text_size(vat_str, body_font)
        self._draw_text(vat_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        self._draw_text("TỔNG CỘNG:", self.margin, self.y_cursor, header_font, black)
        tot_str = self._format_currency(total, "VND")
        tw, _ = self._text_size(tot_str, header_font)
        self._draw_text(tot_str, self.width - self.margin - tw, self.y_cursor, header_font, blue)
        self._advance_y(font=header_font)

        # Payment deadline
        self._advance_y(20)
        deadline = f"25/{month}/{year}"
        self._draw_text(f"Hạn thanh toán: {deadline}", self.margin, self.y_cursor, body_font, (180, 0, 0))

        return self.img


class EcommerceReceiptLayout(BaseLayout):
    """E-commerce order confirmation layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.ECOMMERCE_RECEIPT,
            width_range=(400, 500),
            height_range=(700, 1100),
            margin=25,
            line_spacing=1.3,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render e-commerce receipt."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 20, bold=True)
        header_font = FontManager.get_font("sans", 14, bold=True)
        body_font = FontManager.get_font("sans", 12)
        small_font = FontManager.get_font("sans", 10)

        black = (30, 30, 30)
        gray = (120, 120, 120)

        # E-commerce platforms
        platforms_vocab = VIETNAMESE_BRANDS.get("ecommerce_platforms", [])
        if platforms_vocab:
            platform_name = random.choice(platforms_vocab)
            # Map colors
            if "Shopee" in platform_name: brand_color = (238, 77, 45)
            elif "Lazada" in platform_name: brand_color = (15, 0, 107)
            elif "Tiki" in platform_name: brand_color = (27, 168, 255)
            elif "Sendo" in platform_name: brand_color = (238, 28, 37)
            elif "TikTok" in platform_name: brand_color = (0, 0, 0)
            else: brand_color = (50, 50, 50)
            platform = platform_name
        else:
            platforms = [
                ("Shopee", (238, 77, 45)),
                ("Lazada", (15, 0, 107)),
                ("Tiki", (27, 168, 255)),
                ("Sendo", (238, 28, 37)),
                ("TikTok Shop", (0, 0, 0)),
            ]
            platform, brand_color = random.choice(platforms)

        # Header
        tw, _ = self._text_size(platform, title_font)
        self._draw_text(platform, (self.width - tw) // 2, self.y_cursor, title_font, brand_color)
        self._advance_y(font=title_font)

        tw, _ = self._text_size("Xác nhận đơn hàng", header_font)
        self._draw_text("Xác nhận đơn hàng", (self.width - tw) // 2, self.y_cursor, header_font, gray)
        self._advance_y(font=header_font)
        self._advance_y(15)

        # Order info
        order_id = f"{random.randint(100000000, 999999999)}"
        self._draw_text(f"Mã đơn hàng: #{order_id}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        date = data.get("date", "16/01/2026")
        self._draw_text(f"Ngày đặt: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # Shipping info
        self._draw_text("Địa chỉ giao hàng:", self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        names = ["Nguyễn Văn A", "Trần Thị B", "Lê C"]
        phones = ["0901234567", "0912345678", "0923456789"]
        self._draw_text(f"{random.choice(names)} • {random.choice(phones)}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        address = data.get("address", "123 Nguyễn Huệ, Q.1, TP.HCM")
        self._draw_text(address[:45], self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Items
        self._draw_text("Sản phẩm", self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(5)

        items = data.get("items", [])
        currency = data.get("currency", "VND")

        for item in items:
            desc = item.get("desc", "Sản phẩm")[:35]
            qty = item.get("qty", 1)
            total = item.get("total", 0)

            self._draw_text(f"{desc}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

            qty_line = f"  Số lượng: {qty}"
            self._draw_text(qty_line, self.margin, self.y_cursor, small_font, gray)
            if currency == "VND":
                price_str = f"{int(total):,}đ"
            else:
                price_str = f"${total:.2f}"
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._advance_y(5)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Costs breakdown
        subtotal = data.get("subtotal", 0)
        shipping = random.choice([0, 15000, 20000, 25000, 30000])
        discount = random.choice([0, 0, -10000, -20000, -30000, -50000])

        def draw_cost_line(label, value, color=gray):
            self._draw_text(label, self.margin, self.y_cursor, small_font, gray)
            self._draw_text(label, self.margin, self.y_cursor, small_font, gray)
            if currency == "VND":
                 val_str = self._format_currency(value, currency)
            else:
                if value < 0:
                     val_str = f"-${int(abs(value)):.2f}"
                else:
                     val_str = f"${value:.2f}"
            tw, _ = self._text_size(val_str, body_font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, body_font, color)
            self._advance_y(font=body_font)

        draw_cost_line("Tạm tính", subtotal, black)
        draw_cost_line("Phí vận chuyển", shipping, black)
        if shipping == 0:
            self._draw_text("  Miễn phí vận chuyển", self.margin, self.y_cursor, small_font, brand_color)
            self._advance_y(font=small_font)
        if discount < 0:
            draw_cost_line("Giảm giá / Voucher", discount, brand_color)

        total = subtotal + shipping + discount

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Total
        self._draw_text("Tổng thanh toán", self.margin, self.y_cursor, header_font, black)
        if currency == "VND":
            total_str = self._format_currency(total, currency)
        else:
            total_str = f"${total:.2f}"
        tw, _ = self._text_size(total_str, header_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        # Payment method
        self._advance_y(15)
        payments = ["COD", "Ví ShopeePay", "Ví MoMo", "Thẻ tín dụng", "Chuyển khoản"]
        payment = random.choice(payments)
        self._draw_text(f"Phương thức: {payment}", self.margin, self.y_cursor, small_font, gray)

        return self.img


class TaxiReceiptLayout(BaseLayout):                

    """Taxi/ride-hailing receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.TAXI_RECEIPT,
            width_range=(300, 380),
            height_range=(450, 650),
            margin=20,
            line_spacing=1.3,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render taxi receipt."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 18, bold=True)
        header_font = FontManager.get_font("sans", 13, bold=True)
        body_font = FontManager.get_font("sans", 11)
        small_font = FontManager.get_font("sans", 9)

        black = (30, 30, 30)
        gray = (120, 120, 120)

        # Ride-hailing apps
        taxi_vocab = VIETNAMESE_BRANDS.get("taxi_services", [])
        if taxi_vocab:
             app_name = random.choice(taxi_vocab)
             if "Grab" in app_name: brand_color = (0, 171, 102)
             elif "Be" in app_name: brand_color = (255, 201, 60)
             elif "Gojek" in app_name: brand_color = (0, 170, 90)
             elif "Xanh SM" in app_name: brand_color = (0, 160, 100)
             elif "Mai Linh" in app_name: brand_color = (0, 128, 0)
             elif "Vinasun" in app_name: brand_color = (0, 100, 0)
             else: brand_color = (200, 200, 0) # Taxi yellow
        else:
            apps = [
                ("Grab", (0, 171, 102)),
                ("Be", (255, 201, 60)),
                ("Gojek", (0, 170, 90)),
                ("Xanh SM", (0, 160, 100)),
                ("Mai Linh", (0, 128, 0)),
                ("Vinasun", (255, 255, 255)),
            ]
            app_name, brand_color = random.choice(apps)
            if app_name == "Vinasun":
                brand_color = (0, 100, 0)

        # Header
        tw, _ = self._text_size(app_name, title_font)
        self._draw_text(app_name, (self.width - tw) // 2, self.y_cursor, title_font, brand_color)
        self._advance_y(font=title_font)

        trip_types = ["GrabCar", "GrabBike", "Be Car", "Be Bike", "GoRide", "GoCar", "Xe Taxi", "Xe Ôm"]
        trip_type = random.choice(trip_types)
        tw, _ = self._text_size(trip_type, body_font)
        self._draw_text(trip_type, (self.width - tw) // 2, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)
        self._advance_y(15)

        # Trip ID
        trip_id = f"TRIP{random.randint(100000, 999999)}"
        date = data.get("date", "16/01/2026 10:30")
        self._draw_text(f"Mã chuyến: {trip_id}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(f"Thời gian: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # Route
        pickup_points = ["123 Nguyễn Huệ, Q.1", "Vincom Center, Q.1", "Aeon Mall Tân Phú", "Sân bay Tân Sơn Nhất"]
        dropoff_points = ["456 Lê Lợi, Q.1", "Landmark 81, Bình Thạnh", "Big C Gò Vấp", "Bến xe Miền Đông"]

        self._draw_text("Điểm đón:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(random.choice(pickup_points), self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_text("Điểm đến:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(random.choice(dropoff_points), self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Trip details
        distance = round(random.uniform(2, 25), 1)
        duration = random.randint(10, 60)

        self._draw_text(f"Quãng đường: {distance} km", self.margin, self.y_cursor, body_font, black)
        tw, _ = self._text_size(f"{duration} phút", body_font)
        self._draw_text(f"{duration} phút", self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Fare breakdown
        base_fare = random.choice([10000, 12000, 15000])
        per_km = random.choice([8000, 10000, 12000, 15000])
        distance_fare = int(distance * per_km)
        promo = random.choice([0, 0, -10000, -15000, -20000])
        total = base_fare + distance_fare + promo

        self._draw_text("Giá mở cửa:", self.margin, self.y_cursor, small_font, gray)
        bf_str = self._format_currency(base_fare, "VND")
        tw, _ = self._text_size(bf_str, body_font)
        self._draw_text(bf_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text(f"Cước ({distance} km):", self.margin, self.y_cursor, small_font, gray)
        df_str = self._format_currency(distance_fare, "VND")
        tw, _ = self._text_size(df_str, body_font)
        self._draw_text(df_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        if promo < 0:
            self._draw_text("Khuyến mãi:", self.margin, self.y_cursor, small_font, gray)
            p_str = self._format_currency(promo, "VND")
            tw, _ = self._text_size(p_str, body_font)
            self._draw_text(p_str, self.width - self.margin - tw, self.y_cursor, body_font, brand_color)
            self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Total
        self._draw_text("Tổng cộng", self.margin, self.y_cursor, header_font, black)
        tot_str = self._format_currency(total, "VND")
        tw, _ = self._text_size(tot_str, header_font)
        self._draw_text(tot_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        # Payment
        self._advance_y(15)
        payments = ["Tiền mặt", "GrabPay", "MoMo", "ZaloPay", "Thẻ"]
        payment = random.choice(payments)
        self._draw_text(f"Thanh toán: {payment}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        # Driver info
        self._advance_y(10)
        driver_names = ["Minh", "Hùng", "Tuấn", "Nam", "Dũng"]
        plate = f"59A-{random.randint(100, 999)}.{random.randint(10, 99)}"
        self._draw_text(f"Tài xế: {random.choice(driver_names)} • {plate}", self.margin, self.y_cursor, small_font, gray)

        return self.img


class LayoutFactory:
    """Factory for creating layout instances."""

    LAYOUTS = {
        LayoutType.SUPERMARKET_THERMAL: ThermalReceiptLayout,
        LayoutType.FORMAL_VAT: FormalVATLayout,
        LayoutType.HANDWRITTEN: HandwrittenLayout,
        LayoutType.CAFE_MINIMAL: CafeMinimalLayout,
        LayoutType.RESTAURANT_BILL: RestaurantBillLayout,
        LayoutType.MODERN_POS: ModernPOSLayout,
        LayoutType.DELIVERY_RECEIPT: DeliveryReceiptLayout,
        LayoutType.HOTEL_BILL: HotelBillLayout,
        LayoutType.UTILITY_BILL: UtilityBillLayout,
        LayoutType.ECOMMERCE_RECEIPT: EcommerceReceiptLayout,
        LayoutType.TAXI_RECEIPT: TaxiReceiptLayout,
    }

    @classmethod
    def create(cls, layout_type: LayoutType = None) -> BaseLayout:
        """Create a layout instance."""
        if layout_type is None:
            layout_type = random.choice(list(cls.LAYOUTS.keys()))

        layout_class = cls.LAYOUTS.get(layout_type)
        if layout_class:
            return layout_class()

        # Default to thermal
        return ThermalReceiptLayout()

    @classmethod
    def create_random(cls, weights: Dict[LayoutType, float] = None) -> BaseLayout:
        """Create a random layout with optional weights."""
        if weights is None:
            weights = {
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

        types = list(weights.keys())
        probs = [weights[t] for t in types]
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(types, weights=probs, k=1)[0]
        return cls.create(chosen)
