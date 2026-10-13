"""สร้างไอคอนและภาพประกอบตัวติดตั้งทั้งหมดจากโค้ด — โลโก้คือรูปแก้ว (มัคมีไอน้ำ) สีเขียวของแอป

    python installer/make_assets.py        # เขียนทับ installer/assets/* (ไฟล์ผลลัพธ์ commit ไว้ ตัวสร้างไม่ต้องรันสคริปต์นี้)

ต้องมี Pillow · ฟอนต์ DejaVu (ตัวหนังสือบนภาพวิซาร์ด) — ภาพที่ได้ฝังตัวอักษรไว้แล้ว
ภาพ Windows (.bmp) มาจาก cup() ตัวเดียวกับไอคอน; โลโก้ในหัวแอป (app/app.py: cup_mark) วาดรูปทรงเดียวกันด้วย Qt
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "assets"
GROUND, PANEL, LINE = (8, 11, 16), (14, 18, 26), (48, 59, 76)
GREEN = (61, 214, 140)
S = 4                                                    # วาดใหญ่ S เท่าแล้วย่อ — ขอบเนียน

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PRODUCT = "Cuptick"


def font(path, px):
    return ImageFont.truetype(path, int(px))


def gradient(w, h, top, bottom):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return img


def cup(img, cx, cy, size, fill=GREEN):
    """แก้วมัคมองด้านข้าง: ตัวแก้ว + ขอบปาก + หู + ไอน้ำสองเส้น — กว้างประมาณ 0.8*size สูงประมาณ 0.95*size
    วาดรูปทรงลงมาสก์แล้วระบายสี จึงใช้ได้กับพื้นทึบและพื้นโปร่งใสเหมือนกัน (รูของหูเป็นพื้นหลังจริง)"""
    s = size
    x0 = cx - 0.405 * s                                  # ชิดซ้ายของตัวแก้ว (ขยับให้ทั้งรูป รวมหู อยู่กึ่งกลาง)
    bw, bh = 0.52 * s, 0.52 * s
    x1, y0 = x0 + bw, cy - 0.10 * s
    y1 = y0 + bh
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    # หู: วงแหวน (ครึ่งซ้ายถูกตัวแก้วทับ เหลือครึ่งขวาที่เห็น)
    hr = 0.19 * s
    hx, hy = x1 + 0.10 * s, y0 + 0.25 * s
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], outline=255, width=max(2, int(0.085 * s)))
    # ตัวแก้ว: ก้นโค้ง ปากตรง + ขอบปาก
    d.rounded_rectangle([x0, y0, x1, y1], radius=0.15 * s, fill=255, corners=(False, False, True, True))
    d.rounded_rectangle([x0 - 0.035 * s, y0 - 0.05 * s, x1 + 0.035 * s, y0 + 0.035 * s], radius=0.035 * s, fill=255)
    # ไอน้ำ: เส้นคดสองเส้นเหนือแก้ว
    w = max(2, int(0.055 * s))
    for frac, phase in ((0.28, 0.0), (0.68, math.pi)):
        px = x0 + bw * frac
        pts = [(px + 0.035 * s * math.sin(t * 2 * math.pi + phase), y0 - 0.11 * s - t * 0.27 * s) for t in [i / 24 for i in range(25)]]
        d.line(pts, fill=255, width=w, joint="curve")
        for x, y in (pts[0], pts[-1]):
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
    img.paste(fill, mask=m)


def icon(size=1024):
    """ไอคอนแอป: สี่เหลี่ยมมุมมนพื้นเข้ม + แก้วสีเขียว"""
    w = size * S // 4
    img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(w * 0.06)
    d.rounded_rectangle([pad, pad, w - pad, w - pad], radius=int(w * 0.22), fill=PANEL, outline=LINE, width=max(2, w // 160))
    cup(img, w / 2, w * 0.5, w * 0.62)
    return img.resize((size, size), Image.LANCZOS)


def wizard(w, h):
    """ภาพแถบซ้ายของวิซาร์ด Windows (164x314 ที่ 100%) — โทนสว่างเรียบ แก้ว + ชื่อสินค้า"""
    W, H = w * S, h * S
    img = gradient(W, H, (250, 251, 252), (232, 236, 241))
    d = ImageDraw.Draw(img)
    cup(img, W / 2, H * 0.34, W * 0.62)
    f = font(BOLD, W * 0.105)
    for i, word in enumerate(PRODUCT.split()):
        d.text((W / 2, H * 0.56 + i * W * 0.14), word, font=f, fill=(28, 36, 48), anchor="ma")
    return img.resize((w, h), Image.LANCZOS)


def wizard_small(w):
    """ตราเล็กมุมขวาบนของหน้าตัวติดตั้ง (55x55) บนพื้นขาว"""
    W = w * S
    img = Image.new("RGB", (W, W), (255, 255, 255))
    cup(img, W / 2, W * 0.52, W * 0.7)
    return img.resize((w, w), Image.LANCZOS)


def pkg_logo(size=256):
    """โลโก้โปร่งใสสำหรับพื้นหลังหน้าต่างตัวติดตั้ง macOS (.pkg) — ใช้ได้ทั้งโหมดสว่าง/มืด"""
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cup(img, W / 2, W * 0.52, W * 0.78)
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
