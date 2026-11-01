"""Draw the AudtoText icon with Qt and write assets/audtotext.ico.

The .ico holds 16-256 px frames as 32-bit BMP entries, so Windows picks a
crisp frame for the taskbar, title bar and desktop shortcut.

Usage:
    python tools/make_icon.py
"""
import ctypes
import os
import struct
import sys
from pathlib import Path

if sys.platform == "win32":
    # Same conda ICU clash as in audtotext_gui.py: load Windows' icuuc.dll first.
    ctypes.WinDLL(os.path.join(os.environ["SystemRoot"], "System32", "icuuc.dll"))

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath

SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parent.parent / "assets" / "audtotext.ico"


def draw(size: int) -> QImage:
    """Rounded tile with a waveform on the left and text lines on the right (256-unit grid)."""
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    k = size / 256.0
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor("#2563eb"))
    grad.setColorAt(1.0, QColor("#0d9488"))
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(8 * k, 8 * k, 240 * k, 240 * k), 56 * k, 56 * k)
    p.fillPath(tile, grad)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("white"))
    for i, height in enumerate((60, 128, 88)):
        x = 44 * k + i * 38 * k
        p.drawRoundedRect(QRectF(x, 128 * k - height * k / 2, 24 * k, height * k), 12 * k, 12 * k)
    for i, width in enumerate((64, 48, 64)):
        p.drawRoundedRect(QRectF(160 * k, (86 + i * 34) * k, width * k, 18 * k), 9 * k, 9 * k)
    p.end()
    return img


def dib_entry(img: QImage) -> bytes:
    """BITMAPINFOHEADER + bottom-up BGRA pixels + 1-bit AND mask (set where transparent)."""
    w, h = img.width(), img.height()
    raw = bytes(img.constBits())
    stride = img.bytesPerLine()
    rows = [raw[y * stride: y * stride + w * 4] for y in range(h)]
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    mask_stride = ((w + 31) // 32) * 4
    mask = bytearray()
    for row in reversed(rows):
        bits = bytearray(mask_stride)
        for x in range(w):
            if row[x * 4 + 3] == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        mask += bits
    return header + b"".join(reversed(rows)) + bytes(mask)


def main() -> None:
    QGuiApplication([])
    entries = [dib_entry(draw(s)) for s in SIZES]
    offset = 6 + 16 * len(SIZES)
    out = bytearray(struct.pack("<HHH", 0, 1, len(SIZES)))
    for size, data in zip(SIZES, entries):
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    for data in entries:
        out += data
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(out)
    print(f"wrote {OUT} ({len(out):,} bytes, sizes {SIZES})")


if __name__ == "__main__":
    main()
