"""Поиск пар «изображение + разметка» в источниках YOLO-формата."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path | None
    prefix: str          # префикс источника — разводит одинаковые имена файлов

    @property
    def key(self) -> str:
        return f"{self.prefix}_{self.image.stem}"


def source_prefix(source: Path) -> str:
    """Короткий стабильный префикс источника.

    Считается от имени каталога, а не от полного пути: перенос источника
    в другое место не перемешивает сплит train/val.
    """
    digest = hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:6]
    return f"{source.name[:24]}-{digest}"


def label_for(image: Path) -> Path | None:
    """Разметка рядом с изображением либо в соседнем labels/."""
    candidates = [
        image.parent.parent / "labels" / f"{image.stem}.txt",
        image.parent / "labels" / f"{image.stem}.txt",
        image.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_pairs(source: Path) -> list[Pair]:
    prefix = source_prefix(source)
    pairs: list[Pair] = []
    for image in sorted(source.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS or not image.is_file():
            continue
        label = label_for(image)
        if label is not None:
            pairs.append(Pair(image=image, label=label, prefix=prefix))
    return pairs


def parse_label(path: Path, keep_classes: set[int] | None,
                target_class: int = 0) -> tuple[list[str], int]:
    """Строки разметки, приведённые к одному классу. Возвращает (строки, отброшено)."""
    lines: list[str] = []
    dropped = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            dropped += 1 if parts else 0
            continue
        try:
            source_class = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:5]]
        except ValueError:
            dropped += 1
            continue
        if keep_classes is not None and source_class not in keep_classes:
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            dropped += 1
            continue
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            dropped += 1
            continue
        lines.append(f"{target_class} " + " ".join(f"{v:.6f}" for v in coordinates))
    return lines, dropped


def split_of(key: str, val_ratio: float) -> str:
    """Детерминированный сплит по хэшу имени — стабилен между сборками."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"
