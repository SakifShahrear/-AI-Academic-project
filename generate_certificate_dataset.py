#!/usr/bin/env python3
"""Generate synthetic real and manipulated certificate datasets.

Usage:
    python generate_certificate_dataset.py --input_dir Certificates

Dependencies:
    pip install pillow faker
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

COMPETITIONS = [
    "Coding Contest",
    "Debate Championship",
    "Math Olympiad",
    "Hackathon",
    "Robotics Challenge",
    "Programming Contest",
    "Data Science Challenge",
    "Cyber Security Competition",
    "Research Poster Competition",
    "Innovation Challenge",
]

POSITIONS = [
    "Champion",
    "1st Runner-up",
    "2nd Runner-up",
    "Finalist",
    "Participant",
    "Best Presenter",
    "Best Innovator",
]

FONT_CANDIDATES = {
    "sans": ["arial.ttf", "calibri.ttf", "segoeui.ttf"],
    "serif": ["times.ttf", "timesbd.ttf", "georgia.ttf", "cambria.ttf"],
    "script": ["seguisbi.ttf", "gabriola.ttf"],
}


@dataclass
class TemplateConfig:
    name_xy: Tuple[int, int]
    competition_xy: Tuple[int, int]
    position_xy: Tuple[int, int]
    date_xy: Tuple[int, int]
    logo_box: Tuple[int, int, int, int]
    font_family: str = "sans"
    font_size_name: int = 52
    font_size_body: int = 30
    font_size_date: int = 28
    color_name: Tuple[int, int, int] = (20, 20, 20)
    color_body: Tuple[int, int, int] = (30, 30, 30)
    color_date: Tuple[int, int, int] = (30, 30, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic certificate datasets.")
    parser.add_argument("--input_dir", type=str, default="certificates", help="Folder with original templates.")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output dataset directory.")
    parser.add_argument("--coords_file", type=str, default="template_coords.json", help="Template coordinate config file.")
    parser.add_argument("--real_count", type=int, default=100, help="Number of synthetic real images.")
    parser.add_argument("--fake_count", type=int, default=50, help="Number of manipulated fake images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def find_input_dir(user_input_dir: Path) -> Path:
    if user_input_dir.exists() and user_input_dir.is_dir():
        return user_input_dir

    # Convenience fallback for folder name case differences.
    candidates = [Path("certificates"), Path("Certificates")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Template folder not found: {user_input_dir}. "
        "Please pass a valid folder with --input_dir."
    )


def list_template_images(template_dir: Path) -> List[Path]:
    templates = [
        p for p in sorted(template_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Ignore obvious temporary artifacts.
    templates = [p for p in templates if not p.name.lower().startswith("_tmp")]

    if not templates:
        raise RuntimeError(f"No image templates found in {template_dir}")

    return templates


def default_template_config(width: int, height: int) -> TemplateConfig:
    # These are placeholders; edit them in template_coords.json after first run.
    return TemplateConfig(
        name_xy=(int(width * 0.32), int(height * 0.44)),
        competition_xy=(int(width * 0.20), int(height * 0.60)),
        position_xy=(int(width * 0.20), int(height * 0.66)),
        date_xy=(int(width * 0.12), int(height * 0.14)),
        logo_box=(int(width * 0.02), int(height * 0.02), int(width * 0.22), int(height * 0.22)),
        font_family="sans",
        font_size_name=max(22, int(height * 0.06)),
        font_size_body=max(18, int(height * 0.035)),
        font_size_date=max(16, int(height * 0.03)),
        color_name=(20, 20, 20),
        color_body=(35, 35, 35),
        color_date=(45, 45, 45),
    )


def config_to_dict(cfg: TemplateConfig) -> Dict[str, object]:
    return {
        "name_xy": list(cfg.name_xy),
        "competition_xy": list(cfg.competition_xy),
        "position_xy": list(cfg.position_xy),
        "date_xy": list(cfg.date_xy),
        "logo_box": list(cfg.logo_box),
        "font_family": cfg.font_family,
        "font_size_name": cfg.font_size_name,
        "font_size_body": cfg.font_size_body,
        "font_size_date": cfg.font_size_date,
        "color_name": list(cfg.color_name),
        "color_body": list(cfg.color_body),
        "color_date": list(cfg.color_date),
    }


def dict_to_config(data: Dict[str, object]) -> TemplateConfig:
    return TemplateConfig(
        name_xy=tuple(data["name_xy"]),
        competition_xy=tuple(data["competition_xy"]),
        position_xy=tuple(data["position_xy"]),
        date_xy=tuple(data["date_xy"]),
        logo_box=tuple(data["logo_box"]),
        font_family=str(data.get("font_family", "sans")),
        font_size_name=int(data.get("font_size_name", 48)),
        font_size_body=int(data.get("font_size_body", 26)),
        font_size_date=int(data.get("font_size_date", 24)),
        color_name=tuple(data.get("color_name", [20, 20, 20])),
        color_body=tuple(data.get("color_body", [30, 30, 30])),
        color_date=tuple(data.get("color_date", [35, 35, 35])),
    )


def build_or_load_coords(coords_path: Path, templates: List[Path]) -> Dict[str, TemplateConfig]:
    if coords_path.exists():
        raw = json.loads(coords_path.read_text(encoding="utf-8"))
        return {name: dict_to_config(cfg) for name, cfg in raw.items()}

    generated: Dict[str, TemplateConfig] = {}
    for template in templates:
        with Image.open(template) as img:
            generated[template.name] = default_template_config(img.width, img.height)

    serialized = {name: config_to_dict(cfg) for name, cfg in generated.items()}
    coords_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    print(f"Created placeholder coordinate file: {coords_path}")
    return generated


def get_font(font_family: str, size: int, fallback_family: str = "sans") -> ImageFont.FreeTypeFont:
    family = font_family if font_family in FONT_CANDIDATES else fallback_family
    candidates = FONT_CANDIDATES[family]

    # Try local names first, then Windows font directory.
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass

        windows_font = Path("C:/Windows/Fonts") / candidate
        try:
            return ImageFont.truetype(str(windows_font), size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def random_record(fake: Faker) -> Dict[str, str]:
    dt = fake.date_between(start_date="-4y", end_date="today")
    return {
        "name": fake.name().upper(),
        "competition": random.choice(COMPETITIONS),
        "position": random.choice(POSITIONS),
        "date": dt.strftime("%d %B %Y"),
    }


def draw_certificate_text(image: Image.Image, cfg: TemplateConfig, data: Dict[str, str]) -> Dict[str, Tuple[int, int, int, int]]:
    draw = ImageDraw.Draw(image)

    name_font = get_font(cfg.font_family, cfg.font_size_name)
    body_font = get_font(cfg.font_family, cfg.font_size_body)
    date_font = get_font(cfg.font_family, cfg.font_size_date)

    draw.text(cfg.name_xy, data["name"], fill=cfg.color_name, font=name_font)
    draw.text(cfg.competition_xy, f"Competition: {data['competition']}", fill=cfg.color_body, font=body_font)
    draw.text(cfg.position_xy, f"Position: {data['position']}", fill=cfg.color_body, font=body_font)
    draw.text(cfg.date_xy, f"Date: {data['date']}", fill=cfg.color_date, font=date_font)

    name_bbox = draw.textbbox(cfg.name_xy, data["name"], font=name_font)
    date_bbox = draw.textbbox(cfg.date_xy, f"Date: {data['date']}", font=date_font)
    return {"name_bbox": name_bbox, "date_bbox": date_bbox}


def overlay_forgery(image: Image.Image, cfg: TemplateConfig, fake_name: str, name_bbox: Tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = name_bbox

    pad = 10
    overlay_box = (max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad))
    patch_color = (
        random.randint(232, 250),
        random.randint(232, 250),
        random.randint(232, 250),
        255,
    )
    draw.rectangle(overlay_box, fill=patch_color)

    # A tiny blur helps the patch look like a manual clone/edit.
    patch = image.crop(overlay_box).filter(ImageFilter.GaussianBlur(radius=0.8))
    image.paste(patch, overlay_box)

    name_font = get_font(cfg.font_family, cfg.font_size_name)
    draw.text((overlay_box[0] + 4, overlay_box[1] + 2), fake_name, fill=cfg.color_name, font=name_font)


def date_manipulation(image: Image.Image, cfg: TemplateConfig, original_date: str) -> None:
    draw = ImageDraw.Draw(image)

    forced_year = random.choice(["2026", "2027"])
    date_parts = original_date.split(" ")
    if len(date_parts) >= 3:
        date_parts[-1] = forced_year
    manipulated = " ".join(date_parts)

    # Slight alignment difference and font-family mismatch to mimic tampering.
    mismatched_family = "serif" if cfg.font_family != "serif" else "sans"
    mismatched_font = get_font(mismatched_family, max(12, cfg.font_size_date - 1))

    # Light wipe then redraw to emulate edited date field.
    date_text = f"Date: {original_date}"
    bbox = draw.textbbox(cfg.date_xy, date_text, font=get_font(cfg.font_family, cfg.font_size_date))
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 4, bbox[3] + 2), fill=(245, 245, 245, 255))

    offset_xy = (cfg.date_xy[0] + random.randint(2, 6), cfg.date_xy[1] + random.randint(-1, 2))
    draw.text(offset_xy, f"Date: {manipulated}", fill=cfg.color_date, font=mismatched_font)


def logo_degradation(image: Image.Image, logo_box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = logo_box
    x0 = max(0, min(x0, image.width - 2))
    y0 = max(0, min(y0, image.height - 2))
    x1 = max(x0 + 1, min(x1, image.width))
    y1 = max(y0 + 1, min(y1, image.height))

    region = image.crop((x0, y0, x1, y1))

    # Pixelate then blur slightly to simulate low-quality patch edits.
    pixel_w = max(1, region.width // 7)
    pixel_h = max(1, region.height // 7)
    region = region.resize((pixel_w, pixel_h), Image.Resampling.NEAREST)
    region = region.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    region = region.filter(ImageFilter.GaussianBlur(radius=0.6))

    image.paste(region, (x0, y0, x1, y1))


def alignment_error(image: Image.Image, cfg: TemplateConfig, name: str, name_bbox: Tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = name_bbox

    # Cover original render then redraw shifted by a few pixels.
    draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(245, 245, 245, 255))

    shift_x = random.choice([-8, -5, 5, 8])
    shift_y = random.choice([-4, 4])
    font = get_font(cfg.font_family, cfg.font_size_name)
    draw.text((cfg.name_xy[0] + shift_x, cfg.name_xy[1] + shift_y), name, fill=cfg.color_name, font=font)


def save_metadata(metadata_rows: List[Dict[str, str]], output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "label", "manipulation_type"])
        writer.writeheader()
        writer.writerows(metadata_rows)


def make_dirs(output_dir: Path) -> Tuple[Path, Path]:
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    return real_dir, fake_dir


def generate_real_samples(
    templates: List[Path],
    template_cfgs: Dict[str, TemplateConfig],
    real_dir: Path,
    real_count: int,
    fake: Faker,
) -> List[Dict[str, str]]:
    metadata: List[Dict[str, str]] = []

    for i in range(real_count):
        template_path = templates[i % len(templates)]
        cfg = template_cfgs.get(template_path.name)
        if cfg is None:
            with Image.open(template_path) as img_ref:
                cfg = default_template_config(img_ref.width, img_ref.height)

        with Image.open(template_path).convert("RGB") as image:
            data = random_record(fake)
            draw_certificate_text(image, cfg, data)

            out_name = f"real_{i + 1:03d}{template_path.suffix.lower()}"
            image.save(real_dir / out_name, quality=95)

        metadata.append({
            "filename": f"real/{out_name}",
            "label": "1",
            "manipulation_type": "none",
        })

    return metadata


def generate_fake_samples(
    templates: List[Path],
    template_cfgs: Dict[str, TemplateConfig],
    fake_dir: Path,
    fake_count: int,
    fake: Faker,
) -> List[Dict[str, str]]:
    metadata: List[Dict[str, str]] = []
    manipulations = ["overlay", "date", "logo", "alignment"]

    for i in range(fake_count):
        template_path = templates[i % len(templates)]
        cfg = template_cfgs.get(template_path.name)
        if cfg is None:
            with Image.open(template_path) as img_ref:
                cfg = default_template_config(img_ref.width, img_ref.height)

        manipulation = manipulations[i % len(manipulations)]

        with Image.open(template_path).convert("RGB") as image:
            data = random_record(fake)
            boxes = draw_certificate_text(image, cfg, data)

            if manipulation == "overlay":
                overlay_forgery(image, cfg, fake.name().upper(), boxes["name_bbox"])
            elif manipulation == "date":
                date_manipulation(image, cfg, data["date"])
            elif manipulation == "logo":
                logo_degradation(image, cfg.logo_box)
            elif manipulation == "alignment":
                alignment_error(image, cfg, data["name"], boxes["name_bbox"])

            out_name = f"fake_{i + 1:03d}{template_path.suffix.lower()}"
            image.save(fake_dir / out_name, quality=95)

        metadata.append({
            "filename": f"fake/{out_name}",
            "label": "0",
            "manipulation_type": manipulation,
        })

    return metadata


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    input_dir = find_input_dir(Path(args.input_dir))
    output_dir = Path(args.output_dir)
    coords_path = Path(args.coords_file)

    templates = list_template_images(input_dir)
    template_cfgs = build_or_load_coords(coords_path, templates)

    real_dir, fake_dir = make_dirs(output_dir)

    metadata_rows: List[Dict[str, str]] = []
    metadata_rows.extend(generate_real_samples(templates, template_cfgs, real_dir, args.real_count, fake))
    metadata_rows.extend(generate_fake_samples(templates, template_cfgs, fake_dir, args.fake_count, fake))

    save_metadata(metadata_rows, output_dir)

    print(f"Templates used: {len(templates)}")
    print(f"Real samples: {args.real_count} -> {real_dir}")
    print(f"Fake samples: {args.fake_count} -> {fake_dir}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
