"""สร้างไอคอนและภาพประกอบตัวติดตั้งทั้งหมดจากโค้ด (ตราสินค้าสีเขียว #3DD68C — ภาพวิซาร์ดโทนสว่างเรียบ)

    python installer/make_assets.py        # เขียนทับ installer/assets/* (ไฟล์ผลลัพธ์ commit ไว้ ตัวสร้างไม่ต้องรันสคริปต์นี้)

ต้องมี Pillow · ฟอนต์ไทย (Loma จาก tlwg) ใช้เฉพาะตอนสร้างภาพ — ภาพที่ได้ฝังตัวอักษรไว้แล้ว
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "assets"
GROUND, PANEL, LINE = (8, 11, 16), (14, 18, 26), (48, 59, 76)
GREEN, AMBER, TXT, DIM = (61, 214, 140), (242, 179, 75), (244, 247, 252), (140, 153, 174)
S = 4                                                    # วาดใหญ่ S เท่าแล้วย่อ — ขอบเนียน

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
THAI = "/usr/share/fonts/opentype/tlwg/Loma.otf"


def font(path, px):
    return ImageFont.truetype(path, int(px))


def mark(img, cx, cy, size, fill=GREEN):
    """สี่เหลี่ยมตะแคงมุมมน — เครื่องหมายสถานี เหมือนมุมซ้ายบนของแอป
    วาดสี่เหลี่ยมมุมมนบนมาสก์แล้วหมุน 45° (ความกว้างทแยงมุม = size)"""
    side = size / 2 ** 0.5
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], radius=side * 0.16, fill=255)
    m = m.rotate(45, center=(cx, cy), resample=Image.BICUBIC)
    img.paste(fill, mask=m)


def ruler(d, x, y0, y1, step, long_every=5, length=6, scale=1, color=LINE):
    """ขีดสเกลแบบขอบเครื่องวัด"""
    for i, y in enumerate(range(int(y0), int(y1), int(step))):
        d.line([(x, y), (x + (length * 2 if i % long_every == 0 else length) * scale, y)], fill=color, width=max(1, scale))


def spaced(d, xy, text, f, fill, track, anchor="la"):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor=anchor)
        x += d.textlength(ch, font=f) + track


def gradient(w, h, top, bottom):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return img


def icon(size=1024):
    """ไอคอนแอป: สี่เหลี่ยมมุมมนพื้นเข้ม + เครื่องหมายเขียว + ติ๊กถูก"""
    w = size * S // 4
    img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(w * 0.06)
    d.rounded_rectangle([pad, pad, w - pad, w - pad], radius=int(w * 0.22), fill=PANEL, outline=LINE, width=max(2, w // 160))
    tick = int(w * 0.04)                                   # ขีดสเกลมุมซ้าย — ภาษาเดียวกับแอป
    ruler(d, pad + int(w * 0.05), pad + int(w * 0.18), w - pad - int(w * 0.15), int(w * 0.04), 5, int(w * 0.012), 1, LINE)
    mark(img, w / 2, w / 2, w * 0.5)
    k = w * 0.5 / 2
    pts = [(w / 2 - k * 0.42, w / 2 + k * 0.02), (w / 2 - k * 0.1, w / 2 + k * 0.34), (w / 2 + k * 0.46, w / 2 - k * 0.3)]
    d.line(pts, fill=GROUND, width=int(w * 0.055), joint="curve")
    for x, y in (pts[0], pts[-1]):
        r = w * 0.0275
        d.ellipse([x - r, y - r, x + r, y + r], fill=GROUND)
    return img.resize((size, size), Image.LANCZOS)


def wizard(w, h):
    """ภาพแถบซ้ายของวิซาร์ด Windows (164x314 ที่ 100%) — โทนสว่างเรียบ ตราสินค้ากับชื่อ ไม่มีข้อความที่ผูกกับภาษา"""
    W, H = w * S, h * S
    img = gradient(W, H, (250, 251, 252), (232, 236, 241))
    d = ImageDraw.Draw(img)
    cx, cy = W / 2, H * 0.36
    mark(img, cx, cy, W * 0.5)
    k = W * 0.5 / 2
    d.line([(cx - k * 0.42, cy + k * 0.02), (cx - k * 0.1, cy + k * 0.34), (cx + k * 0.46, cy - k * 0.3)], fill=(250, 251, 252), width=int(W * 0.03), joint="curve")
    for x, y in ((cx - k * 0.42, cy + k * 0.02), (cx + k * 0.46, cy - k * 0.3)):
        r = W * 0.015
        d.ellipse([x - r, y - r, x + r, y + r], fill=(250, 251, 252))
    f = font(BOLD, W * 0.105)
    for i, word in enumerate(("Inspection", "Station")):
        d.text((cx, H * 0.58 + i * W * 0.14), word, font=f, fill=(28, 36, 48), anchor="ma")
    return img.resize((w, h), Image.LANCZOS)


def wizard_small(w):
    """ตราเล็กมุมขวาบนของหน้าตัวติดตั้ง (55x55) บนพื้นขาว"""
    W = w * S
    img = Image.new("RGB", (W, W), (255, 255, 255))
    d = ImageDraw.Draw(img)
    mark(img, W / 2, W / 2, W * 0.62)
    k = W * 0.62 / 2
    d.line([(W / 2 - k * 0.42, W / 2 + k * 0.02), (W / 2 - k * 0.1, W / 2 + k * 0.34), (W / 2 + k * 0.46, W / 2 - k * 0.3)], fill=(255, 255, 255), width=int(W * 0.06), joint="curve")
    return img.resize((w, w), Image.LANCZOS)


def pkg_logo(size=256):
    """โลโก้โปร่งใสสำหรับพื้นหลังหน้าต่างตัวติดตั้ง macOS (.pkg) — ใช้ได้ทั้งโหมดสว่าง/มืด"""
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mark(img, W / 2, W / 2, W * 0.8)
    k = W * 0.8 / 2
    d.line([(W / 2 - k * 0.42, W / 2 + k * 0.02), (W / 2 - k * 0.1, W / 2 + k * 0.34), (W / 2 + k * 0.46, W / 2 - k * 0.3)], fill=(0, 0, 0, 0), width=int(W * 0.05), joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ic = icon(1024)
    ic.save(OUT / "icon.png")
    ic.save(OUT / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    ic.save(OUT / "icon.icns")
    wizard(164, 314).save(OUT / "wizard_large.bmp")
    wizard(328, 628).save(OUT / "wizard_large_2x.bmp")
    wizard_small(55).save(OUT / "wizard_small.bmp")
    wizard_small(110).save(OUT / "wizard_small_2x.bmp")
    pkg_logo(256).save(OUT / "pkg_logo.png")
    print("wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
