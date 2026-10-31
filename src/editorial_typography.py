"""Stable local typography assets; bundled licensed fonts, no network or timestamps."""
import hashlib
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OPENING_NAME = "Hi, I’m David"
OPENING_PERSPECTIVE = "An image of the times—well, it’s all a matter of perspective."
OPENING_GAME = "Connect Four looks easy enough. Stay for a move, or just a moment."
ATTRIBUTION = "A daily interpretation of selected public feeds, produced by a custom AI system.\nOutputs do not necessarily reflect my views.\n"
IMAGE_ASSETS = tuple(f"assets/{kind}_{theme}.webp" for kind in ("editorial_opening", "connect_four_title") for theme in ("light", "dark"))
MANIFEST = "assets/editorial_typography.json"
ASSETS = (*IMAGE_ASSETS, MANIFEST)

@lru_cache(maxsize=1)
def input_signature():
    fonts = Path(__file__).resolve().parents[1]/"assets/fonts"
    payload = {"renderer": 2, "copy": [OPENING_NAME, OPENING_PERSPECTIVE, OPENING_GAME],
               "fonts": {name: hashlib.sha256((fonts/name).read_bytes()).hexdigest() for name in ("CormorantGaramond.ttf", "CormorantGaramond-Italic.ttf")}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=4)
def _render(kind, theme):
    scale = 2
    height = 140 if kind == "editorial_opening" else 46
    image = Image.new("RGBA", (740*scale, height*scale))
    draw = ImageDraw.Draw(image)
    fonts = Path(__file__).resolve().parents[1]/"assets/fonts"
    lines = [(OPENING_NAME, 36, 8, False), (OPENING_PERSPECTIVE, 24, 57, True), (OPENING_GAME, 24, 94, False)] if kind == "editorial_opening" else [("Connect Four", 28, 6, False)]
    for text, size, y, italic in lines:
        font = ImageFont.truetype(str(fonts/("CormorantGaramond-Italic.ttf" if italic else "CormorantGaramond.ttf")), size*scale)
        font.set_variation_by_axes([400])
        draw.text((370*scale,y*scale), text, font=font, anchor="mt", fill="#DAD4CC" if theme=="dark" else "#38322D")
    output = BytesIO()
    image.save(output, format="WEBP", lossless=True, exact=True)
    return output.getvalue()

def write_typography(root):
    root = Path(root)
    signature = input_signature()
    manifest = root/MANIFEST
    try:
        saved = json.loads(manifest.read_text())
        if (saved["input_sha256"] == signature and set(saved["files"]) == set(IMAGE_ASSETS)
                and all(hashlib.sha256((root/p).read_bytes()).hexdigest()==digest for p,digest in saved["files"].items())):
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    hashes = {}
    for relative in IMAGE_ASSETS:
        path = root/relative
        kind, theme = path.stem.rsplit("_",1)
        raw = _render(kind,theme)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        if not path.exists() or path.read_bytes()!=raw:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
    manifest.write_text(json.dumps({"input_sha256": signature, "files": hashes},sort_keys=True,indent=2)+"\n")

def picture(kind, prefix, alt, width=740):
    from html import escape
    version = input_signature()[:12]
    # GitHub's theme-only fragments follow its saved theme even when it differs
    # from the OS preference; picture media queries follow the OS instead.
    return (f'<img src="{prefix}/{kind}_light.webp?v={version}#gh-light-mode-only" width="{width}" alt="{escape(alt,quote=True)}">'
            f'<img src="{prefix}/{kind}_dark.webp?v={version}#gh-dark-mode-only" width="{width}" alt="{escape(alt,quote=True)}">')
